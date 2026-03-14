#!/usr/bin/env python3
"""Test WebArena environment reset for all train and test tasks via actual Env.reset()."""
import asyncio
import json
import os
import sys
import logging

logging.basicConfig(level=logging.WARNING)

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


async def test_split(task_config_file: str, label: str, skip_sites=None):
    """Test reset for all tasks in a config file."""
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"Config:  {task_config_file}")
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

    env = WebArenaEnv(config)
    n_tasks = len(env._config_files)
    print(f"Tasks (after filtering): {n_tasks}")

    ok, fail = 0, 0
    failed_seeds = []

    for seed in range(n_tasks):
        try:
            obs, info = await env.reset(seed=seed)
            ok += 1
            status = "OK"
        except Exception as e:
            fail += 1
            failed_seeds.append((seed, str(e)[:100]))
            status = f"FAIL: {e}"

        # Print progress every task
        cfg_file = os.path.basename(env._config_files[seed % n_tasks])
        print(f"  [{seed+1}/{n_tasks}] seed={seed} {cfg_file}: {status[:80]}")

    # Cleanup
    await env.close()

    print(f"\n--- {label} Results ---")
    print(f"  OK:   {ok}/{n_tasks}")
    print(f"  FAIL: {fail}/{n_tasks}")
    if failed_seeds:
        print(f"  Failed seeds:")
        for s, err in failed_seeds:
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
