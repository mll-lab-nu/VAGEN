"""
Benchmark: parallel WebArenaEnv instances with random actions.

Each env runs in its own **process** (via ProcessPoolExecutor) to get
full isolation of Playwright's internal asyncio loop and greenlets.

Uses --max_concurrent to cap the number of simultaneous browser
processes, avoiding system resource exhaustion.

Usage:
    python -m vagen.envs.webarena.parallel_env_test --num_envs 40 --steps_per_env 5 --max_concurrent 8
    python -m vagen.envs.webarena.parallel_env_test --num_envs 4 --steps_per_env 5
"""

import concurrent.futures
import random
import time
from typing import List

import fire


# Random actions that don't require valid element ids
RANDOM_ACTIONS = [
    "scroll [down]",
    "scroll [up]",
    "press [Enter]",
    "press [Escape]",
    "click [1]",
    "click [2]",
    "click [5]",
    "go_back",
]


def _run_env_in_process(
    env_id: int,
    config_file: str,
    steps: int,
    headless: bool,
    timeout_ms: int = 60000,
) -> dict:
    """
    Run one env in a standalone process (sync Playwright, fully isolated).

    Args:
        timeout_ms: Playwright navigation/action timeout in ms.
    """
    from vagen.envs.webarena.browser_env import ScriptBrowserEnv, create_id_based_action

    timings = {"env_id": env_id, "reset": 0.0, "steps": [], "total": 0.0}
    t_total = time.perf_counter()

    env = ScriptBrowserEnv(
        headless=headless,
        observation_type="accessibility_tree",
        current_viewport_only=True,
        viewport_size={"width": 1280, "height": 720},
        sleep_after_execution=0.0,
    )

    # --- reset ---
    t0 = time.perf_counter()
    try:
        obs, info = env.reset(options={"config_file": config_file})
        env.page.set_default_timeout(timeout_ms)
        env.page.set_default_navigation_timeout(timeout_ms)
        timings["reset"] = time.perf_counter() - t0
        print(f"  [env {env_id}] reset done in {timings['reset']:.2f}s", flush=True)
    except Exception as e:
        timings["reset"] = time.perf_counter() - t0
        timings["reset_error"] = str(e)
        print(f"  [env {env_id}] reset FAILED in {timings['reset']:.2f}s: {e}", flush=True)
        # ScriptBrowserEnv.close() skips __exit__() when reset_finished=False,
        # leaving Playwright's internal asyncio loop running and polluting the
        # worker process for future tasks. Force cleanup here.
        try:
            if hasattr(env, "context_manager"):
                env.context_manager.__exit__()
        except Exception:
            pass
        timings["total"] = time.perf_counter() - t_total
        return timings

    # --- random steps ---
    for step_i in range(steps):
        action_str = random.choice(RANDOM_ACTIONS)
        t0 = time.perf_counter()
        try:
            action = create_id_based_action(action_str)
            obs, reward, terminated, truncated, step_info = env.step(action)
            dt = time.perf_counter() - t0
            timings["steps"].append({"step": step_i, "time": dt, "action": action_str})
            print(f"  [env {env_id}] step {step_i} done in {dt:.2f}s ({action_str})", flush=True)
            if terminated:
                print(f"  [env {env_id}] episode ended at step {step_i}", flush=True)
                break
        except Exception as e:
            dt = time.perf_counter() - t0
            timings["steps"].append({"step": step_i, "time": dt, "action": action_str, "error": str(e)})
            print(f"  [env {env_id}] step {step_i} FAILED in {dt:.2f}s ({action_str}): {e}", flush=True)
            continue

    try:
        env.close()
    except Exception:
        pass
    timings["total"] = time.perf_counter() - t_total
    return timings


def benchmark(
    config_file: str = "/work/nvme/bgig/ryu4/VAGEN/vagen/envs/webarena/config_files/0.json",
    num_envs: int = 4,
    steps_per_env: int = 5,
    headless: bool = True,
    max_concurrent: int = 8,
    skip_sequential: bool = False,
    timeout: int = 120,
):
    """
    Args:
        timeout: Per-env timeout in seconds (default 120s).
                 Controls both Playwright internal timeouts and process-level timeout.
    """
    pw_timeout_ms = timeout * 1000
    workers = min(max_concurrent, num_envs)
    print(f"=== Parallel benchmark: {num_envs} envs x {steps_per_env} steps ===")
    print(f"Config: {config_file}")
    print(f"Max concurrent browsers: {workers}")
    print(f"Per-env timeout: {timeout}s\n")

    # --- Parallel run (each env in its own process) ---
    t_parallel = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_run_env_in_process, i, config_file, steps_per_env, headless, pw_timeout_ms)
            for i in range(num_envs)
        ]
        results_parallel: List[dict] = []
        for i, f in enumerate(futures):
            try:
                results_parallel.append(f.result(timeout=timeout))
            except concurrent.futures.TimeoutError:
                f.cancel()
                print(f"  [env {i}] TIMED OUT after {timeout}s")
                results_parallel.append({
                    "env_id": i, "reset": timeout, "steps": [],
                    "total": timeout, "reset_error": f"timed out after {timeout}s",
                })
            except Exception as e:
                print(f"  [env {i}] PROCESS ERROR: {e}")
                results_parallel.append({
                    "env_id": i, "reset": 0, "steps": [],
                    "total": 0, "reset_error": str(e),
                })
    wall_parallel = time.perf_counter() - t_parallel

    # --- Sequential run (optional, for comparison) ---
    results_sequential: List[dict] = []
    wall_sequential = 0.0
    if not skip_sequential:
        print(f"\n=== Sequential benchmark: {num_envs} envs x {steps_per_env} steps ===\n")
        t_sequential = time.perf_counter()
        for i in range(num_envs):
            r = _run_env_in_process(i + num_envs, config_file, steps_per_env, headless, pw_timeout_ms)
            results_sequential.append(r)
        wall_sequential = time.perf_counter() - t_sequential

    # --- Report ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    def summarize(results, label, wall_time):
        reset_times = [r["reset"] for r in results]
        step_times = [s["time"] for r in results for s in r["steps"]]
        env_totals = [r["total"] for r in results]
        errors = sum(1 for r in results if "reset_error" in r)
        step_errors = sum(1 for r in results for s in r["steps"] if "error" in s)
        ok_count = len(results) - errors
        print(f"\n--- {label} ---")
        print(f"  Wall clock:        {wall_time:.2f}s")
        print(f"  Envs OK/FAIL:      {ok_count}/{errors}")
        print(f"  Avg reset time:    {sum(reset_times)/max(len(reset_times),1):.2f}s")
        if step_times:
            print(f"  Avg step time:     {sum(step_times)/len(step_times):.2f}s")
            print(f"  Total steps:       {len(step_times)} ({step_errors} errors)")
        print(f"  Sum of env totals: {sum(env_totals):.2f}s")
        if wall_time > 0 and step_times:
            print(f"  Throughput:        {len(step_times) / wall_time:.2f} steps/s")

    summarize(results_parallel, f"PARALLEL ({num_envs} envs, {workers} workers)", wall_parallel)

    if results_sequential:
        summarize(results_sequential, f"SEQUENTIAL ({num_envs} envs)", wall_sequential)
        if wall_parallel > 0 and wall_sequential > 0:
            speedup = wall_sequential / wall_parallel
            print(f"\n  Speedup: {speedup:.2f}x")

    print()


if __name__ == "__main__":
    fire.Fire(benchmark)
