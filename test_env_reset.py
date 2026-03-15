#!/usr/bin/env python3
"""Test WebArena environment reset for all train and test tasks via actual Env.reset().
Uses parallel workers for speed."""
import asyncio
import json
import os
import sys
import logging
import time

logging.basicConfig(level=logging.WARNING)

CONCURRENCY = int(os.environ.get("CONCURRENCY", 8))

# Check env vars first
required_vars = ["SHOPPING", "SHOPPING_ADMIN", "GITLAB", "REDDIT", "WIKIPEDIA", "MAP", "HOMEPAGE"]
print("=== Environment Variables ===")
for var in required_vars:
    val = os.environ.get(var, "")
    print(f"  {var}={val}")
    if not val:
        print(f"ERROR: {var} not set!")
        sys.exit(1)

from vagen.envs.webarena.webarena_env import WebArenaEnv


N_TEST_STEPS = int(os.environ.get("TEST_STEPS", 0))

async def test_one_seed(config, seed, n_tasks):
    """Test a single seed with its own env instance."""
    env = WebArenaEnv(config)
    cfg_file = os.path.basename(env._config_files[seed % n_tasks])
    # Read site info from config
    config_path = env._config_files[seed % n_tasks]
    with open(config_path) as f:
        task_cfg = json.load(f)
    sites = task_cfg.get("sites", ["unknown"])
    site_str = ",".join(sites)

    t0 = time.time()
    step_times = []
    try:
        obs, info = await env.reset(seed=seed)
        reset_elapsed = time.time() - t0

        # Run a few test steps if requested
        for i in range(N_TEST_STEPS):
            action = "<action>scroll [down]</action>"
            st0 = time.time()
            obs, reward, done, info = await env.step(action)
            step_times.append(time.time() - st0)
            if done:
                break

        result = ("OK", seed, cfg_file, None, reset_elapsed, step_times, site_str)
    except Exception as e:
        elapsed = time.time() - t0
        import traceback
        tb = traceback.format_exc()
        result = ("FAIL", seed, cfg_file, tb[-500:], elapsed, step_times, site_str)
    finally:
        await env.close()
    return result


async def test_split(task_config_file: str, label: str, skip_sites=None):
    """Test reset for all tasks in a config file, running CONCURRENCY in parallel."""
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"Config:  {task_config_file}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"{'='*60}")

    config = {
        "task_config_file": task_config_file,
        "headless": True,
        "observation_type": "accessibility_tree",
        "current_viewport_only": True,
        "max_steps": 30,
        "skip_sites": skip_sites or ["map"],
        "reset_timeout": 60.0,
        "playwright_timeout": 30000,
        "nav_timeout": 30000,
    }

    # Get task count
    tmp_env = WebArenaEnv(config)
    n_tasks = len(tmp_env._config_files)
    await tmp_env.close()
    print(f"Tasks (after filtering): {n_tasks}")

    t0 = time.time()
    sem = asyncio.Semaphore(CONCURRENCY)
    ok, fail = 0, 0
    failed_seeds = []
    reset_times: list[float] = []
    done_count = 0

    all_step_times = []

    async def bounded_test(seed):
        nonlocal ok, fail, done_count
        async with sem:
            result = await test_one_seed(config, seed, n_tasks)
        status, s, cfg, err, elapsed, step_times, site = result
        done_count += 1
        reset_times.append(elapsed)
        all_step_times.extend(step_times)
        if status == "OK":
            ok += 1
        else:
            fail += 1
            failed_seeds.append((s, err))
        tag = f"[{done_count}/{n_tasks}]"
        step_info = ""
        if step_times:
            avg_st = sum(step_times) / len(step_times)
            step_info = f" steps={len(step_times)} avg={avg_st:.2f}s"
        if status == "OK":
            print(f"  {tag} seed={s:>4} [{site:20s}] {cfg}: OK (reset={elapsed:.1f}s{step_info})")
        else:
            print(f"  {tag} seed={s:>4} [{site:20s}] {cfg}: FAIL ({elapsed:.1f}s) - {err}")

    # Process seeds in batches of CONCURRENCY to avoid creating too many env instances
    for batch_start in range(0, n_tasks, CONCURRENCY):
        batch_seeds = range(batch_start, min(batch_start + CONCURRENCY, n_tasks))
        await asyncio.gather(*[bounded_test(seed) for seed in batch_seeds])

    wall_time = time.time() - t0
    avg_reset = sum(reset_times) / len(reset_times) if reset_times else 0
    print(f"\n--- {label} Results ---")
    print(f"  Wall clock:     {wall_time:.1f}s")
    print(f"  OK:             {ok}/{n_tasks}")
    print(f"  FAIL:           {fail}/{n_tasks}")
    print(f"  Avg reset time: {avg_reset:.2f}s")
    if reset_times:
        print(f"  Min/Max reset:  {min(reset_times):.2f}s / {max(reset_times):.2f}s")
    if all_step_times:
        avg_st = sum(all_step_times) / len(all_step_times)
        print(f"  Avg step time:  {avg_st:.2f}s ({len(all_step_times)} steps)")
        print(f"  Min/Max step:   {min(all_step_times):.2f}s / {max(all_step_times):.2f}s")
    if failed_seeds:
        print(f"  Failed seeds:")
        for s, err in sorted(failed_seeds):
            print(f"    seed={s}: {err}")

    return fail


async def main():
    base = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base, "vagen/envs/webarena/config_files/wa")
    skip_sites = ["map"]

    total_fails = 0

    # Test validation split (smaller, test first)
    test_path = os.path.join(config_dir, "test_webarena_lite.json")
    if os.path.exists(test_path):
        total_fails += await test_split(test_path, "TEST (test_webarena_lite)", skip_sites)
    else:
        print(f"Test config not found: {test_path}")

    # Test train split
    train_path = os.path.join(config_dir, "train_webarena_lite.json")
    if os.path.exists(train_path):
        total_fails += await test_split(train_path, "TRAIN (train_webarena_lite)", skip_sites)
    else:
        print(f"Train config not found: {train_path}")

    print(f"\n{'='*60}")
    if total_fails == 0:
        print("ALL RESETS OK!")
    else:
        print(f"TOTAL FAILURES: {total_fails}")
    return total_fails


if __name__ == "__main__":
    fails = asyncio.run(main())
    sys.exit(min(fails, 1))
