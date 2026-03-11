"""
Benchmark: parallel WebArenaEnv instances across DIFFERENT sites/ports.

Tests concurrent access to different backend services to measure
cross-site parallelism (each site is a separate server, so they
shouldn't contend with each other unlike the same-env test).

Sites and their ports:
    shopping_admin  :7780   (config 0.json)
    shopping        :7770   (config 300.json)
    reddit          :9999   (config 399.json)
    gitlab          :8023   (config 339.json)
    map             :3000   (config 100.json)

Usage:
    python -m vagen.envs.webarena.parallel_diff_env_test
    python -m vagen.envs.webarena.parallel_diff_env_test --repeat 3 --steps_per_env 10
    python -m vagen.envs.webarena.parallel_diff_env_test --sites shopping_admin,shopping,reddit
"""

import concurrent.futures
import json
import os
import random
import time
from typing import Dict, List, Optional

import fire


CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config_files")

# One representative config per site (different backend ports)
DEFAULT_SITE_CONFIGS = {
    "shopping_admin": "0.json",     # :7780
    "shopping":       "300.json",   # :7770
    "reddit":         "399.json",   # :9999
    "gitlab":         "339.json",   # :8023
}

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
    site_name: str,
    steps: int,
    headless: bool,
    timeout_ms: int = 60000,
) -> dict:
    """Run one env in a standalone process (sync Playwright, fully isolated)."""
    from vagen.envs.webarena.browser_env import ScriptBrowserEnv, create_id_based_action

    timings = {
        "env_id": env_id,
        "site": site_name,
        "config": os.path.basename(config_file),
        "reset": 0.0,
        "steps": [],
        "total": 0.0,
    }
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
        print(f"  [env {env_id}] ({site_name}) reset done in {timings['reset']:.2f}s", flush=True)
    except Exception as e:
        timings["reset"] = time.perf_counter() - t0
        timings["reset_error"] = str(e)
        print(f"  [env {env_id}] ({site_name}) reset FAILED in {timings['reset']:.2f}s: {e}", flush=True)
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
            print(f"  [env {env_id}] ({site_name}) step {step_i} done in {dt:.2f}s ({action_str})", flush=True)
            if terminated:
                print(f"  [env {env_id}] ({site_name}) episode ended at step {step_i}", flush=True)
                break
        except Exception as e:
            dt = time.perf_counter() - t0
            timings["steps"].append({"step": step_i, "time": dt, "action": action_str, "error": str(e)})
            print(f"  [env {env_id}] ({site_name}) step {step_i} FAILED in {dt:.2f}s ({action_str}): {e}", flush=True)
            continue

    try:
        env.close()
    except Exception:
        pass
    timings["total"] = time.perf_counter() - t_total
    return timings


def _build_task_list(
    sites: Optional[str],
    repeat: int,
) -> List[dict]:
    """Build list of (config_file, site_name) tasks."""
    if sites:
        selected = [s.strip() for s in sites.split(",")]
    else:
        selected = list(DEFAULT_SITE_CONFIGS.keys())

    tasks = []
    for site_name in selected:
        cfg_name = DEFAULT_SITE_CONFIGS.get(site_name)
        if cfg_name is None:
            print(f"  WARNING: unknown site '{site_name}', skipping. "
                  f"Available: {list(DEFAULT_SITE_CONFIGS.keys())}")
            continue
        cfg_path = os.path.join(CONFIG_DIR, cfg_name)
        if not os.path.isfile(cfg_path):
            print(f"  WARNING: config not found: {cfg_path}, skipping.")
            continue
        for _ in range(repeat):
            tasks.append({"config_file": cfg_path, "site_name": site_name})

    return tasks


def benchmark(
    sites: Optional[str] = None,
    repeat: int = 1,
    steps_per_env: int = 5,
    headless: bool = True,
    max_concurrent: int = 8,
    skip_sequential: bool = False,
    timeout: int = 120,
):
    """
    Args:
        sites:          Comma-separated site names to test (default: all).
                        Options: shopping_admin, shopping, reddit, gitlab, map
        repeat:         How many instances per site (default: 1).
        steps_per_env:  Random steps per env after reset.
        max_concurrent: Max simultaneous browser processes.
        skip_sequential: Skip the sequential comparison run.
        timeout:        Per-env timeout in seconds.
    """
    pw_timeout_ms = timeout * 1000
    tasks = _build_task_list(sites, repeat)
    if not tasks:
        print("No valid tasks to run. Check your --sites parameter.")
        return

    num_envs = len(tasks)
    workers = min(max_concurrent, num_envs)
    site_summary = {}
    for t in tasks:
        site_summary[t["site_name"]] = site_summary.get(t["site_name"], 0) + 1

    print(f"=== Diff-env parallel benchmark ===")
    print(f"Sites: {site_summary}")
    print(f"Total envs: {num_envs}, Max concurrent: {workers}")
    print(f"Steps per env: {steps_per_env}, Timeout: {timeout}s\n")

    # --- Parallel run ---
    t_parallel = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _run_env_in_process, i,
                task["config_file"], task["site_name"],
                steps_per_env, headless, pw_timeout_ms,
            )
            for i, task in enumerate(tasks)
        ]
        results_parallel: List[dict] = []
        for i, f in enumerate(futures):
            try:
                results_parallel.append(f.result(timeout=timeout))
            except concurrent.futures.TimeoutError:
                f.cancel()
                site = tasks[i]["site_name"]
                print(f"  [env {i}] ({site}) TIMED OUT after {timeout}s")
                results_parallel.append({
                    "env_id": i, "site": site,
                    "config": os.path.basename(tasks[i]["config_file"]),
                    "reset": timeout, "steps": [],
                    "total": timeout, "reset_error": f"timed out after {timeout}s",
                })
            except Exception as e:
                site = tasks[i]["site_name"]
                print(f"  [env {i}] ({site}) PROCESS ERROR: {e}")
                results_parallel.append({
                    "env_id": i, "site": site,
                    "config": os.path.basename(tasks[i]["config_file"]),
                    "reset": 0, "steps": [],
                    "total": 0, "reset_error": str(e),
                })
    wall_parallel = time.perf_counter() - t_parallel

    # --- Sequential run (optional) ---
    results_sequential: List[dict] = []
    wall_sequential = 0.0
    if not skip_sequential:
        print(f"\n=== Sequential run ===\n")
        t_sequential = time.perf_counter()
        for i, task in enumerate(tasks):
            r = _run_env_in_process(
                i + num_envs, task["config_file"], task["site_name"],
                steps_per_env, headless, pw_timeout_ms,
            )
            results_sequential.append(r)
        wall_sequential = time.perf_counter() - t_sequential

    # --- Report ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    def summarize(results, label, wall_time):
        reset_times = [r["reset"] for r in results]
        step_times = [s["time"] for r in results for s in r["steps"]]
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
        if wall_time > 0 and step_times:
            print(f"  Throughput:        {len(step_times) / wall_time:.2f} steps/s")

        # Per-site breakdown
        by_site: Dict[str, List[dict]] = {}
        for r in results:
            by_site.setdefault(r.get("site", "?"), []).append(r)
        if len(by_site) > 1:
            print(f"\n  Per-site breakdown:")
            for site, site_results in sorted(by_site.items()):
                s_reset = [r["reset"] for r in site_results]
                s_steps = [s["time"] for r in site_results for s in r["steps"]]
                s_errors = sum(1 for r in site_results if "reset_error" in r)
                s_ok = len(site_results) - s_errors
                line = f"    {site:20s}  OK/FAIL: {s_ok}/{s_errors}  reset: {sum(s_reset)/max(len(s_reset),1):.2f}s"
                if s_steps:
                    line += f"  step: {sum(s_steps)/len(s_steps):.2f}s ({len(s_steps)} steps)"
                print(line)

    summarize(results_parallel, f"PARALLEL ({num_envs} envs, {workers} workers)", wall_parallel)

    if results_sequential:
        summarize(results_sequential, f"SEQUENTIAL ({num_envs} envs)", wall_sequential)
        if wall_parallel > 0 and wall_sequential > 0:
            speedup = wall_sequential / wall_parallel
            print(f"\n  Speedup: {speedup:.2f}x")

    print()


if __name__ == "__main__":
    fire.Fire(benchmark)
