"""End-to-end test: single WebArenaEnv session via a real browser pool.

Prerequisites:
    source vagen/envs/webarena/setup_vars.sh   # DATASET + URL env vars
    # WebArena docker services reachable on the ports in setup_vars.sh

Usage:
    python -m vagen.envs.webarena.tests.test_env_local \\
        --seed=0 \\
        --max_steps=3

This runs one full rollout with a canned action sequence (all 'Wait' then
'exit'), confirming reset/step/evaluate/close all work on a real browser.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from vagen.envs.webarena.browser_pool import BrowserPool
from vagen.envs.webarena.webarena_env import WebArenaEnv, load_tasks


async def run(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    task_config_file = os.path.abspath(args.task_config_file)
    auth_cache = os.path.abspath(args.auth_cache_dir)
    os.makedirs(auth_cache, exist_ok=True)

    tasks = load_tasks(task_config_file)
    print(f"Loaded {len(tasks)} tasks from {task_config_file}")
    print(f"Task[{args.seed}]: sites={tasks[args.seed]['sites']} "
          f"intent={tasks[args.seed]['intent'][:80]!r}")

    pool = BrowserPool(n_browsers=1, max_contexts_per_browser=1)
    await pool.start()

    slot = await pool.acquire_slot()
    env = WebArenaEnv(
        env_config={
            "task_config_file": task_config_file,
            "auth_cache_dir": auth_cache,
            "max_steps": args.max_steps,
        },
        browser_slot=slot,
        browser_pool=pool,
        auth_locks={},
        tasks=tasks,
    )

    try:
        print("\n=== system_prompt ===")
        sp = await env.system_prompt()
        print(sp["obs_str"][:300], "...")

        print("\n=== reset ===")
        obs, info = await env.reset(seed=args.seed)
        print(f"info: {info}")
        print(f"obs head: {obs['obs_str'][:500]}")

        print("\n=== step (Wait) ===")
        action = '<think>need to observe</think><answer>do(action="Wait")</answer>'
        obs, reward, done, info = await env.step(action)
        print(f"reward={reward} done={done} info_keys={list(info.keys())}")

        print("\n=== step (exit) ===")
        action = '<think>submitting</think><answer>exit(message="test answer")</answer>'
        obs, reward, done, info = await env.step(action)
        print(f"reward={reward} done={done} success={info.get('success')} "
              f"eval_score={info.get('eval_score')} error={info.get('error')}")

    finally:
        await env.close()
        await pool.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task_config_file", default="vagen/envs/webarena/config_files/normalized_test.json")
    p.add_argument("--auth_cache_dir", default="./.webarena_auth_cache")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
