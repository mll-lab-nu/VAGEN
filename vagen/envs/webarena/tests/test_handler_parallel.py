"""Parallel sanity test: spin up handler, run N sessions concurrently.

Verifies that BrowserPool correctly load-balances across browsers and that
concurrent reset/step/close don't deadlock.

Prerequisites:
    source vagen/envs/webarena/setup_vars.sh

Usage:
    python -m vagen.envs.webarena.tests.test_handler_parallel \\
        --n_browsers=2 --max_contexts_per_browser=4 --n_sessions=8
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time

from vagen.envs.webarena.handler import WebArenaHandler


async def one_rollout(handler: WebArenaHandler, idx: int, seed: int):
    t0 = time.monotonic()
    r = await handler.connect(env_config={"max_steps": 3}, seed=seed)
    sid = r.data["session_id"]
    try:
        await handler.call(sid, "step", {
            "action_str": '<think>wait</think><answer>do(action="Wait")</answer>'
        }, images=[])
        await handler.call(sid, "step", {
            "action_str": '<think>done</think><answer>exit(message="")</answer>'
        }, images=[])
    finally:
        await handler.call(sid, "close", {}, images=[])
    elapsed = time.monotonic() - t0
    print(f"[session {idx}] seed={seed} done in {elapsed:.1f}s stats={handler.pool.stats_str()}")


async def run(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    handler = WebArenaHandler(
        task_config_file=os.path.abspath(args.task_config_file),
        auth_cache_dir=os.path.abspath(args.auth_cache_dir),
        n_browsers=args.n_browsers,
        max_contexts_per_browser=args.max_contexts_per_browser,
    )
    await handler.start()

    try:
        t0 = time.monotonic()
        seeds = list(range(args.n_sessions))
        await asyncio.gather(*(
            one_rollout(handler, i, s) for i, s in enumerate(seeds)
        ))
        print(f"\nTotal: {args.n_sessions} sessions in {time.monotonic() - t0:.1f}s")
    finally:
        await handler.aclose()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task_config_file", default="vagen/envs/webarena/config_files/normalized_test.json")
    p.add_argument("--auth_cache_dir", default="./.webarena_auth_cache")
    p.add_argument("--n_browsers", type=int, default=2)
    p.add_argument("--max_contexts_per_browser", type=int, default=4)
    p.add_argument("--n_sessions", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
