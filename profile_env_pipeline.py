#!/usr/bin/env python3
"""
WebArena Environment Pipeline Profiler
=======================================

Profiles each layer of the training pipeline to find bottlenecks:

  Layer 1: Docker containers (shopping, gitlab, ...) — raw HTTP latency
  Layer 2: Training Server → Docker — HTTP round-trip from this machine
  Layer 3: Playwright browser — launch, page load, DOM access, a11y tree
  Layer 4: WebArenaEnv (Python) — full reset() and step() including all overhead

Usage:
    # Full profiling (all layers)
    python profile_env_pipeline.py

    # Quick mode (fewer rounds)
    python profile_env_pipeline.py --rounds 1

    # Skip specific layers
    python profile_env_pipeline.py --skip-docker --skip-playwright

    # Test remote browser server
    python profile_env_pipeline.py --remote-url http://localhost:5100

    # Control concurrency test
    python profile_env_pipeline.py --concurrency 1,2,4,8

    # Only test specific seeds
    python profile_env_pipeline.py --seeds 0,1,2
"""

import argparse
import asyncio
import concurrent.futures
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

# ─── Utility ────────────────────────────────────────────────────────

SERVICES = {
    "SHOPPING":       os.environ.get("SHOPPING",       "http://localhost:7770"),
    "SHOPPING_ADMIN": os.environ.get("SHOPPING_ADMIN", "http://localhost:7780/admin"),
    "GITLAB":         os.environ.get("GITLAB",         "http://localhost:8023"),
    "REDDIT":         os.environ.get("REDDIT",         "http://localhost:9999"),
    "WIKIPEDIA":      os.environ.get("WIKIPEDIA",      "http://localhost:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"),
    "MAP":            os.environ.get("MAP",             "http://localhost:3000"),
    "HOMEPAGE":       os.environ.get("HOMEPAGE",        "http://localhost:4399"),
}


def fmt_time(t: float) -> str:
    if t < 0.001:
        return f"{t*1e6:.0f}µs"
    if t < 1.0:
        return f"{t*1e3:.1f}ms"
    return f"{t:.2f}s"


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_stats(label: str, times: List[float], indent: int = 2):
    prefix = " " * indent
    if not times:
        print(f"{prefix}{label}: NO DATA")
        return
    avg = statistics.mean(times)
    med = statistics.median(times)
    mn, mx = min(times), max(times)
    std = statistics.stdev(times) if len(times) > 1 else 0
    print(f"{prefix}{label}:")
    print(f"{prefix}  avg={fmt_time(avg)}  med={fmt_time(med)}  "
          f"min={fmt_time(mn)}  max={fmt_time(mx)}  std={fmt_time(std)}  n={len(times)}")


# ─── Layer 1: Docker Container HTTP Latency ─────────────────────────

def profile_docker_http(rounds: int = 3) -> Dict[str, List[float]]:
    """Measure raw HTTP GET latency to each Docker service."""
    import requests

    print_header("Layer 1: Docker Container HTTP Latency")
    print(f"  (Direct HTTP GET to each service, {rounds} rounds)\n")

    results: Dict[str, List[float]] = {}

    for name, url in SERVICES.items():
        times = []
        errors = 0
        for r in range(rounds):
            try:
                t0 = time.perf_counter()
                resp = requests.get(url, timeout=30, allow_redirects=True)
                dt = time.perf_counter() - t0
                times.append(dt)
                status = resp.status_code
            except Exception as e:
                dt = time.perf_counter() - t0
                times.append(dt)
                errors += 1
                status = f"ERR({e.__class__.__name__})"
            print(f"    {name:<18} round {r+1}: {fmt_time(dt):>10}  status={status}", flush=True)
        results[name] = times
        print_stats(name, times, indent=4)
        if errors:
            print(f"      errors: {errors}/{rounds}")
        print()

    # Summary ranking
    print("  Ranking (avg latency):")
    ranked = sorted(results.items(), key=lambda kv: statistics.mean(kv[1]))
    for i, (name, times) in enumerate(ranked, 1):
        print(f"    {i}. {name:<18} avg={fmt_time(statistics.mean(times))}")

    return results


# ─── Layer 2: TCP/Network Latency ───────────────────────────────────

def profile_network(rounds: int = 3) -> Dict[str, Dict]:
    """Measure TCP connect time and DNS resolution to each service."""
    import socket
    from urllib.parse import urlparse

    print_header("Layer 2: Network (TCP Connect) Latency")
    print(f"  (TCP SYN/ACK only, {rounds} rounds)\n")

    results: Dict[str, Dict] = {}

    for name, url in SERVICES.items():
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or 80

        connect_times = []
        for r in range(rounds):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            try:
                t0 = time.perf_counter()
                sock.connect((host, port))
                dt = time.perf_counter() - t0
                connect_times.append(dt)
                print(f"    {name:<18} round {r+1}: TCP connect {fmt_time(dt)}", flush=True)
            except Exception as e:
                print(f"    {name:<18} round {r+1}: FAILED ({e})", flush=True)
            finally:
                sock.close()

        results[name] = {"host": host, "port": port, "times": connect_times}
        if connect_times:
            print_stats(f"{name} ({host}:{port})", connect_times, indent=4)
        print()

    return results


# ─── Layer 3: Playwright Browser Profiling ──────────────────────────

def profile_playwright(rounds: int = 1) -> Dict[str, Any]:
    """Profile Playwright browser operations in detail."""
    from playwright.sync_api import sync_playwright

    print_header("Layer 3: Playwright Browser Profiling")
    print(f"  (Browser launch, context, page load, a11y tree, {rounds} rounds)\n")

    results: Dict[str, Any] = {
        "launch": [], "context_create": [], "page_create": [],
        "cdp_enable": [], "goto": {}, "get_obs": {}, "close": [],
    }

    # Pick a few representative URLs to test
    test_urls = {}
    for name in ["SHOPPING", "REDDIT", "GITLAB", "HOMEPAGE"]:
        if name in SERVICES:
            test_urls[name] = SERVICES[name]
    if not test_urls:
        test_urls["HOMEPAGE"] = SERVICES.get("HOMEPAGE", "http://localhost:4399")

    for rnd in range(rounds):
        print(f"  --- Round {rnd + 1}/{rounds} ---")

        # 3a: Browser launch
        t0 = time.perf_counter()
        cm = sync_playwright()
        pw = cm.__enter__()
        browser = pw.chromium.launch(
            headless=True,
            args=["--blink-settings=imagesEnabled=false"],
        )
        dt_launch = time.perf_counter() - t0
        results["launch"].append(dt_launch)
        print(f"    browser launch:     {fmt_time(dt_launch)}")

        for name, url in test_urls.items():
            # 3b: Context creation
            t0 = time.perf_counter()
            context = browser.new_context(
                viewport={"width": 1280, "height": 720},
                device_scale_factor=1,
            )
            dt_ctx = time.perf_counter() - t0
            results["context_create"].append(dt_ctx)

            # 3c: Page creation + CDP
            t0 = time.perf_counter()
            page = context.new_page()
            dt_page = time.perf_counter() - t0
            results["page_create"].append(dt_page)

            t0 = time.perf_counter()
            client = page.context.new_cdp_session(page)
            client.send("Accessibility.enable")
            dt_cdp = time.perf_counter() - t0
            results["cdp_enable"].append(dt_cdp)

            print(f"    [{name}] context={fmt_time(dt_ctx)}  "
                  f"page={fmt_time(dt_page)}  cdp_enable={fmt_time(dt_cdp)}")

            # 3d: Page navigation (goto)
            page.set_default_timeout(30000)
            page.set_default_navigation_timeout(30000)
            try:
                t0 = time.perf_counter()
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                dt_goto = time.perf_counter() - t0
                results["goto"].setdefault(name, []).append(dt_goto)
                print(f"    [{name}] goto:      {fmt_time(dt_goto)}")
            except Exception as e:
                dt_goto = time.perf_counter() - t0
                results["goto"].setdefault(name, []).append(dt_goto)
                print(f"    [{name}] goto:      {fmt_time(dt_goto)} FAILED: {e}")

            # 3e: Get accessibility tree observation
            try:
                from vagen.envs.webarena.browser_env.processors import ObservationHandler

                handler = ObservationHandler(
                    "text", "accessibility_tree", "",
                    current_viewport_only=True,
                    viewport_size={"width": 1280, "height": 720},
                )

                t0 = time.perf_counter()
                obs = handler.get_observation(page, client)
                dt_obs = time.perf_counter() - t0
                obs_text = obs.get("text", "")
                obs_len = len(obs_text) if isinstance(obs_text, str) else 0
                results["get_obs"].setdefault(name, []).append(dt_obs)
                print(f"    [{name}] get_obs:   {fmt_time(dt_obs)}  (a11y tree len={obs_len})")
            except Exception as e:
                print(f"    [{name}] get_obs:   FAILED: {e}")

            # Cleanup context
            try:
                client.detach()
            except Exception:
                pass
            context.close()

        # 3f: Browser close
        t0 = time.perf_counter()
        browser.close()
        cm.__exit__()
        dt_close = time.perf_counter() - t0
        results["close"].append(dt_close)
        print(f"    browser close:      {fmt_time(dt_close)}")
        print()

    # Summary
    print("  --- Playwright Summary ---")
    print_stats("browser launch", results["launch"])
    print_stats("context create", results["context_create"])
    print_stats("page create", results["page_create"])
    print_stats("CDP enable a11y", results["cdp_enable"])
    print_stats("browser close", results["close"])
    for name in test_urls:
        if name in results["goto"]:
            print_stats(f"goto [{name}]", results["goto"][name])
        if name in results["get_obs"]:
            print_stats(f"get_obs [{name}]", results["get_obs"][name])

    return results


# ─── Layer 3b: Remote Browser Server Profiling ─────────────────────

def profile_remote_browser(remote_url: str, seeds: List[int], rounds: int = 1) -> Dict[str, Any]:
    """Profile remote browser server HTTP API latency."""
    import requests as req

    print_header("Layer 3b: Remote Browser Server Profiling")
    print(f"  Server: {remote_url}")
    print(f"  Seeds: {seeds}, {rounds} round(s)\n")

    # Health check
    try:
        resp = req.get(f"{remote_url}/health", timeout=5)
        print(f"  Health: {resp.json()}")
    except Exception as e:
        print(f"  Health check FAILED: {e}")
        return {}

    results: Dict[str, List[float]] = {
        "reset_total": [], "step_total": [], "close_total": [],
        "reset_network": [], "step_network": [],
    }

    # Load a config file
    base = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base, "vagen/envs/webarena/config_files/wa")
    train_config = os.path.join(config_dir, "train_webarena_lite.json")
    if not os.path.exists(train_config):
        print(f"  Config not found: {train_config}")
        return results

    with open(train_config) as f:
        all_tasks = json.load(f)

    for rnd in range(rounds):
        print(f"\n  --- Round {rnd + 1}/{rounds} ---")
        for seed in seeds:
            task = all_tasks[seed % len(all_tasks)]
            sites = task.get("sites", ["?"])

            # Load storage state if needed
            storage_state = None
            ss_path = task.get("storage_state")
            if ss_path:
                if not os.path.isabs(ss_path):
                    webarena_root = os.path.join(base, "vagen/envs/webarena")
                    ss_path = os.path.join(webarena_root, ss_path)
                if os.path.exists(ss_path):
                    with open(ss_path) as f:
                        storage_state = json.load(f)
                    task_copy = dict(task)
                    del task_copy["storage_state"]
                    task = task_copy

            # RESET
            payload = {
                "config": task,
                "observation_type": "accessibility_tree",
                "current_viewport_only": True,
                "viewport_width": 1280,
                "viewport_height": 720,
                "storage_state": storage_state,
            }
            t0 = time.perf_counter()
            resp = req.post(f"{remote_url}/reset", json=payload, timeout=120)
            dt_reset = time.perf_counter() - t0
            resp.raise_for_status()
            data = resp.json()
            session_id = data["session_id"]
            server_reset_time = data.get("info", {}).get("reset_time", 0)
            network_overhead = dt_reset - server_reset_time if server_reset_time else 0

            results["reset_total"].append(dt_reset)
            results["reset_network"].append(network_overhead)
            obs_len = len(data.get("observation", ""))
            print(f"    seed={seed} [{','.join(sites)}] reset: "
                  f"total={fmt_time(dt_reset)}  server={fmt_time(server_reset_time)}  "
                  f"network={fmt_time(network_overhead)}  obs_len={obs_len}")

            # STEP (scroll down)
            t0 = time.perf_counter()
            resp = req.post(
                f"{remote_url}/step/{session_id}",
                json={"action": "scroll [down]"},
                timeout=30,
            )
            dt_step = time.perf_counter() - t0
            resp.raise_for_status()
            step_data = resp.json()
            server_step_time = step_data.get("info", {}).get("step_time", 0)
            step_net = dt_step - server_step_time if server_step_time else 0

            results["step_total"].append(dt_step)
            results["step_network"].append(step_net)
            print(f"    seed={seed} [{','.join(sites)}] step:  "
                  f"total={fmt_time(dt_step)}  server={fmt_time(server_step_time)}  "
                  f"network={fmt_time(step_net)}")

            # CLOSE
            t0 = time.perf_counter()
            req.delete(f"{remote_url}/session/{session_id}", timeout=10)
            dt_close = time.perf_counter() - t0
            results["close_total"].append(dt_close)

    print("\n  --- Remote Browser Summary ---")
    print_stats("reset (total)", results["reset_total"])
    print_stats("reset (network overhead)", results["reset_network"])
    print_stats("step (total)", results["step_total"])
    print_stats("step (network overhead)", results["step_network"])
    print_stats("close", results["close_total"])

    return results


# ─── Layer 4: Full WebArenaEnv Profiling ────────────────────────────

def profile_webarena_env(
    seeds: List[int],
    rounds: int = 1,
    remote_url: Optional[str] = None,
    steps_per_episode: int = 3,
) -> Dict[str, Any]:
    """Profile the full WebArenaEnv reset/step cycle with detailed breakdown."""

    print_header("Layer 4: Full WebArenaEnv (Python) Profiling")
    mode = "REMOTE" if remote_url else "LOCAL (Playwright)"
    print(f"  Mode: {mode}")
    print(f"  Seeds: {seeds}, {rounds} round(s), {steps_per_episode} steps/episode\n")

    from vagen.envs.webarena.webarena_env import WebArenaEnv

    base = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base, "vagen/envs/webarena/config_files/wa")

    config = {
        "task_config_file": os.path.join(config_dir, "train_webarena_lite.json"),
        "headless": True,
        "observation_type": "accessibility_tree",
        "current_viewport_only": True,
        "max_steps": 30,
        "skip_sites": ["map"],
        "reset_timeout": 120.0,
        "playwright_timeout": 30000,
        "nav_timeout": 30000,
    }
    if remote_url:
        config["remote_browser_url"] = remote_url

    results: Dict[str, List[float]] = {
        "env_init": [], "reset_total": [], "step_total": [],
        "close_total": [], "episode_total": [],
    }

    async def run():
        for rnd in range(rounds):
            print(f"\n  --- Round {rnd + 1}/{rounds} ---")
            for seed in seeds:
                # Init
                t0 = time.perf_counter()
                env = WebArenaEnv(config)
                dt_init = time.perf_counter() - t0
                results["env_init"].append(dt_init)

                cfg_file = env._config_files[seed % len(env._config_files)]
                with open(cfg_file) as f:
                    task = json.load(f)
                sites = task.get("sites", ["?"])

                episode_t0 = time.perf_counter()

                # Reset
                try:
                    t0 = time.perf_counter()
                    obs, info = await env.reset(seed=seed)
                    dt_reset = time.perf_counter() - t0
                    results["reset_total"].append(dt_reset)
                    obs_len = len(obs.get("obs_str", ""))
                    print(f"    seed={seed} [{','.join(sites)}] init={fmt_time(dt_init)}  "
                          f"reset={fmt_time(dt_reset)}  obs_len={obs_len}")
                except Exception as e:
                    dt_reset = time.perf_counter() - t0
                    print(f"    seed={seed} [{','.join(sites)}] reset FAILED in {fmt_time(dt_reset)}: {e}")
                    await env.close()
                    continue

                # Steps
                step_times = []
                for i in range(steps_per_episode):
                    action = "<action>scroll [down]</action>"
                    t0 = time.perf_counter()
                    try:
                        obs, reward, done, step_info = await env.step(action)
                        dt_step = time.perf_counter() - t0
                        step_times.append(dt_step)
                        print(f"      step {i+1}: {fmt_time(dt_step)}  "
                              f"reward={reward:.3f}  done={done}")
                        if done:
                            break
                    except Exception as e:
                        dt_step = time.perf_counter() - t0
                        step_times.append(dt_step)
                        print(f"      step {i+1}: {fmt_time(dt_step)} FAILED: {e}")
                        break

                results["step_total"].extend(step_times)

                # Close
                t0 = time.perf_counter()
                await env.close()
                dt_close = time.perf_counter() - t0
                results["close_total"].append(dt_close)

                dt_episode = time.perf_counter() - episode_t0
                results["episode_total"].append(dt_episode)
                print(f"    seed={seed} episode_total={fmt_time(dt_episode)}  "
                      f"close={fmt_time(dt_close)}")

    asyncio.run(run())

    print("\n  --- WebArenaEnv Summary ---")
    print_stats("env __init__", results["env_init"])
    print_stats("reset (total)", results["reset_total"])
    print_stats("step (total)", results["step_total"])
    print_stats("close", results["close_total"])
    print_stats("episode (total)", results["episode_total"])

    # Breakdown
    if results["reset_total"] and results["step_total"]:
        avg_reset = statistics.mean(results["reset_total"])
        avg_step = statistics.mean(results["step_total"])
        avg_episode = statistics.mean(results["episode_total"]) if results["episode_total"] else 0
        print(f"\n  Breakdown (avg per episode, {steps_per_episode} steps):")
        print(f"    reset:  {fmt_time(avg_reset):>10}  ({100*avg_reset/avg_episode:.0f}% of episode)" if avg_episode else "")
        total_step = avg_step * steps_per_episode
        print(f"    steps:  {fmt_time(total_step):>10}  ({100*total_step/avg_episode:.0f}% of episode)" if avg_episode else "")
        print(f"    other:  {fmt_time(avg_episode - avg_reset - total_step):>10}" if avg_episode else "")

    return results


# ─── Layer 4b: Concurrency Scaling Test ─────────────────────────────

def profile_concurrency(
    concurrency_levels: List[int],
    n_tasks: int = 8,
    remote_url: Optional[str] = None,
) -> Dict[int, Dict]:
    """Test how reset/step performance scales with parallel envs."""

    print_header("Layer 4b: Concurrency Scaling Test")
    mode = "REMOTE" if remote_url else "LOCAL"
    print(f"  Mode: {mode}")
    print(f"  Concurrency levels: {concurrency_levels}")
    print(f"  Tasks per level: {n_tasks}\n")

    from vagen.envs.webarena.webarena_env import WebArenaEnv

    base = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base, "vagen/envs/webarena/config_files/wa")

    config = {
        "task_config_file": os.path.join(config_dir, "train_webarena_lite.json"),
        "headless": True,
        "observation_type": "accessibility_tree",
        "current_viewport_only": True,
        "max_steps": 30,
        "skip_sites": ["map"],
        "reset_timeout": 120.0,
    }
    if remote_url:
        config["remote_browser_url"] = remote_url

    results: Dict[int, Dict] = {}

    async def run_one(seed: int) -> Dict:
        env = WebArenaEnv(config)
        t0 = time.perf_counter()
        try:
            await env.reset(seed=seed)
            dt_reset = time.perf_counter() - t0

            t1 = time.perf_counter()
            await env.step("<action>scroll [down]</action>")
            dt_step = time.perf_counter() - t1

            await env.close()
            return {"reset": dt_reset, "step": dt_step, "error": None}
        except Exception as e:
            await env.close()
            return {"reset": time.perf_counter() - t0, "step": 0, "error": str(e)}

    async def run_level(conc: int):
        sem = asyncio.Semaphore(conc)
        wall_t0 = time.perf_counter()

        async def bounded(seed):
            async with sem:
                return await run_one(seed)

        tasks = [bounded(i) for i in range(n_tasks)]
        task_results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - wall_t0

        resets = [r["reset"] for r in task_results if not r["error"]]
        steps = [r["step"] for r in task_results if not r["error"]]
        errors = sum(1 for r in task_results if r["error"])

        return {
            "wall_time": wall_time,
            "resets": resets,
            "steps": steps,
            "errors": errors,
            "throughput": n_tasks / wall_time if wall_time > 0 else 0,
        }

    for conc in concurrency_levels:
        print(f"  Concurrency={conc}:")
        res = asyncio.run(run_level(conc))
        results[conc] = res

        print(f"    wall_time={fmt_time(res['wall_time'])}  "
              f"throughput={res['throughput']:.2f} tasks/s  "
              f"errors={res['errors']}/{n_tasks}")
        if res["resets"]:
            print_stats("reset", res["resets"], indent=4)
        if res["steps"]:
            print_stats("step", res["steps"], indent=4)
        print()

    # Summary table
    print("  --- Concurrency Summary ---")
    print(f"  {'Conc':>6} {'Wall':>10} {'Throughput':>12} {'Avg Reset':>12} {'Avg Step':>12} {'Errors':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for conc in concurrency_levels:
        r = results[conc]
        avg_r = fmt_time(statistics.mean(r["resets"])) if r["resets"] else "N/A"
        avg_s = fmt_time(statistics.mean(r["steps"])) if r["steps"] else "N/A"
        print(f"  {conc:>6} {fmt_time(r['wall_time']):>10} {r['throughput']:>10.2f}/s "
              f"{avg_r:>12} {avg_s:>12} {r['errors']:>8}")

    return results


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WebArena Environment Pipeline Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rounds", type=int, default=2,
                        help="Number of rounds for each test (default: 2)")
    parser.add_argument("--seeds", type=str, default="0,5,10",
                        help="Comma-separated seed list (default: 0,5,10)")
    parser.add_argument("--remote-url", type=str, default=None,
                        help="Remote browser server URL (e.g. http://localhost:5100)")
    parser.add_argument("--steps", type=int, default=3,
                        help="Steps per episode in Layer 4 test (default: 3)")

    # Skip flags
    parser.add_argument("--skip-docker", action="store_true",
                        help="Skip Layer 1 (Docker HTTP)")
    parser.add_argument("--skip-network", action="store_true",
                        help="Skip Layer 2 (TCP connect)")
    parser.add_argument("--skip-playwright", action="store_true",
                        help="Skip Layer 3 (Playwright)")
    parser.add_argument("--skip-env", action="store_true",
                        help="Skip Layer 4 (WebArenaEnv)")
    parser.add_argument("--skip-concurrency", action="store_true",
                        help="Skip concurrency scaling test")

    # Concurrency
    parser.add_argument("--concurrency", type=str, default="1,2,4",
                        help="Concurrency levels to test (default: 1,2,4)")
    parser.add_argument("--concurrency-tasks", type=int, default=8,
                        help="Number of tasks for concurrency test (default: 8)")

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    conc_levels = [int(c) for c in args.concurrency.split(",")]

    print("=" * 70)
    print("  WebArena Pipeline Profiler")
    print(f"  seeds={seeds}  rounds={args.rounds}  steps/episode={args.steps}")
    print(f"  remote_url={args.remote_url or '(local)'}")
    print("=" * 70)

    all_results: Dict[str, Any] = {}
    total_t0 = time.perf_counter()

    # Layer 1
    if not args.skip_docker:
        all_results["docker_http"] = profile_docker_http(rounds=args.rounds)

    # Layer 2
    if not args.skip_network:
        all_results["network"] = profile_network(rounds=args.rounds)

    # Layer 3
    if not args.skip_playwright:
        if args.remote_url:
            all_results["remote_browser"] = profile_remote_browser(
                args.remote_url, seeds, rounds=args.rounds,
            )
        else:
            all_results["playwright"] = profile_playwright(rounds=args.rounds)

    # Layer 4
    if not args.skip_env:
        all_results["webarena_env"] = profile_webarena_env(
            seeds, rounds=args.rounds, remote_url=args.remote_url,
            steps_per_episode=args.steps,
        )

    # Concurrency test
    if not args.skip_concurrency:
        all_results["concurrency"] = profile_concurrency(
            conc_levels, n_tasks=args.concurrency_tasks,
            remote_url=args.remote_url,
        )

    total_time = time.perf_counter() - total_t0

    # ─── Grand Summary ──────────────────────────────────────────
    print_header("GRAND SUMMARY — Where is the time going?")

    summary_items: List[tuple] = []

    if "docker_http" in all_results:
        for name, times in all_results["docker_http"].items():
            avg = statistics.mean(times)
            summary_items.append((f"L1 Docker HTTP [{name}]", avg))

    if "network" in all_results:
        for name, info in all_results["network"].items():
            if info["times"]:
                avg = statistics.mean(info["times"])
                summary_items.append((f"L2 TCP connect [{name}]", avg))

    if "playwright" in all_results:
        pw = all_results["playwright"]
        if pw.get("launch"):
            summary_items.append(("L3 Playwright launch", statistics.mean(pw["launch"])))
        if pw.get("context_create"):
            summary_items.append(("L3 context create", statistics.mean(pw["context_create"])))
        for name, times in pw.get("goto", {}).items():
            summary_items.append((f"L3 goto [{name}]", statistics.mean(times)))
        for name, times in pw.get("get_obs", {}).items():
            summary_items.append((f"L3 get_obs [{name}]", statistics.mean(times)))

    if "webarena_env" in all_results:
        env = all_results["webarena_env"]
        if env.get("reset_total"):
            summary_items.append(("L4 WebArenaEnv reset", statistics.mean(env["reset_total"])))
        if env.get("step_total"):
            summary_items.append(("L4 WebArenaEnv step", statistics.mean(env["step_total"])))

    # Sort by time descending
    summary_items.sort(key=lambda x: x[1], reverse=True)

    if summary_items:
        max_label = max(len(s[0]) for s in summary_items)
        max_time = summary_items[0][1] if summary_items else 1.0
        bar_width = 40

        print(f"\n  {'Operation':<{max_label}}  {'Time':>10}  Bar")
        print(f"  {'-'*max_label}  {'-'*10}  {'-'*bar_width}")
        for label, t in summary_items:
            bar_len = int(bar_width * t / max_time) if max_time > 0 else 0
            bar = "█" * bar_len
            print(f"  {label:<{max_label}}  {fmt_time(t):>10}  {bar}")

    print(f"\n  Total profiling time: {fmt_time(total_time)}")
    print()


if __name__ == "__main__":
    main()
