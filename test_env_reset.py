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


async def test_one_seed(config, seed, n_tasks):
    """Test a single seed with its own env instance."""
    env = WebArenaEnv(config)
    cfg_file = os.path.basename(env._config_files[seed % n_tasks])
    try:
        obs, info = await env.reset(seed=seed)
        result = ("OK", seed, cfg_file, None)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        result = ("FAIL", seed, cfg_file, tb[-500:])
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
        "reset_timeout": 30.0,
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
    done_count = 0

    async def bounded_test(seed):
        nonlocal ok, fail, done_count
        async with sem:
            result = await test_one_seed(config, seed, n_tasks)
        status, s, cfg, err = result
        done_count += 1
        if status == "OK":
            ok += 1
        else:
            fail += 1
            failed_seeds.append((s, err))
        tag = f"[{done_count}/{n_tasks}]"
        if status == "FAIL":
            print(f"  {tag} seed={s:>4} {cfg}: FAIL - {err}")
        elif done_count % 20 == 0 or done_count == n_tasks:
            print(f"  {tag} progress... ({ok} ok, {fail} fail)")

    tasks = [bounded_test(seed) for seed in range(n_tasks)]
    await asyncio.gather(*tasks)

    elapsed = time.time() - t0
    print(f"\n--- {label} Results ({elapsed:.1f}s) ---")
    print(f"  OK:   {ok}/{n_tasks}")
    print(f"  FAIL: {fail}/{n_tasks}")
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
