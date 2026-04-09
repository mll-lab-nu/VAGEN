"""
Test env pool logic in EbAlfredHandler.

Uses a mock env to verify pool lifecycle without needing AI2-THOR.
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vagen.envs.eb_alfred.handler import EbAlfredHandler


class FakeEnv:
    """Mock env that tracks create/close calls."""
    _count = 0

    def __init__(self):
        FakeEnv._count += 1
        self.id = FakeEnv._count
        self.closed = False
        self._assigned_display = "0"

    async def close(self):
        self.closed = True

    async def reset(self, seed):
        return {"obs_str": f"obs-{seed}"}, {"seed": seed}

    async def system_prompt(self):
        return {"obs_str": "system prompt"}

    async def step(self, action):
        return {"obs_str": "step"}, 0.0, False, {}


async def test_pool_basic():
    """Verify envs are pooled on close and reused on connect."""
    FakeEnv._count = 0
    handler = EbAlfredHandler(
        x_displays=["0"],
        capacity=4,
        startup_concurrency=4,
        pool_size=4,
    )

    # Monkey-patch create_env to return FakeEnv
    async def fake_create(config):
        env = FakeEnv()
        env._assigned_display = "0"
        return env

    handler.create_env = fake_create

    # Create 4 sessions
    results = []
    for i in range(4):
        r = await handler.connect({"eval_set": "base"}, seed=i)
        results.append(r)

    # Wait for all envs to be ready
    for sid, ctx in handler._sessions.items():
        if hasattr(ctx, '_ready') and ctx._ready:
            await ctx._ready.wait()

    assert len(handler._sessions) == 4, f"Expected 4 sessions, got {len(handler._sessions)}"
    assert FakeEnv._count == 4, f"Expected 4 envs created, got {FakeEnv._count}"
    assert len(handler._env_pool) == 0

    # Close all 4 → should go to pool
    sids = list(handler._sessions.keys())
    for sid in sids:
        ctx = handler._sessions[sid]
        await handler._handle_close(ctx)

    assert len(handler._sessions) == 0
    assert len(handler._env_pool) == 4, f"Expected 4 pooled, got {len(handler._env_pool)}"
    assert FakeEnv._count == 4, "No new envs should be created"

    # Create 4 more sessions → should reuse from pool
    for i in range(4):
        await handler.connect({"eval_set": "base"}, seed=i + 100)

    # Wait for all envs to be ready
    for sid, ctx in handler._sessions.items():
        if hasattr(ctx, '_ready') and ctx._ready:
            await ctx._ready.wait()

    assert len(handler._sessions) == 4
    assert len(handler._env_pool) == 0, f"Pool should be empty, got {len(handler._env_pool)}"
    assert FakeEnv._count == 4, f"Should reuse, not create new! Got {FakeEnv._count}"

    print("PASS: test_pool_basic")


async def test_pool_overflow():
    """When pool is full, env should be actually closed."""
    FakeEnv._count = 0
    handler = EbAlfredHandler(
        x_displays=["0"],
        capacity=4,
        startup_concurrency=4,
        pool_size=2,  # Only keep 2 in pool
    )

    async def fake_create(config):
        env = FakeEnv()
        env._assigned_display = "0"
        return env

    handler.create_env = fake_create

    # Create 4 sessions
    for i in range(4):
        await handler.connect({"eval_set": "base"}, seed=i)

    for sid, ctx in handler._sessions.items():
        if hasattr(ctx, '_ready') and ctx._ready:
            await ctx._ready.wait()

    # Close all 4 → first 2 pooled, last 2 actually closed
    sids = list(handler._sessions.keys())
    closed_envs = []
    for sid in sids:
        ctx = handler._sessions[sid]
        if ctx.env:
            closed_envs.append(ctx.env)
        await handler._handle_close(ctx)

    assert len(handler._env_pool) == 2, f"Expected 2 pooled, got {len(handler._env_pool)}"
    actually_closed = sum(1 for e in closed_envs if e.closed)
    assert actually_closed == 2, f"Expected 2 closed, got {actually_closed}"

    print("PASS: test_pool_overflow")


async def test_no_deadlock_with_queuing():
    """With batch_size > capacity, verify no deadlock: queued sessions
    should be served as envs are pooled and permits released."""
    FakeEnv._count = 0
    handler = EbAlfredHandler(
        x_displays=["0"],
        capacity=2,
        startup_concurrency=2,
        pool_size=2,
    )

    async def fake_create(config):
        await asyncio.sleep(0.01)  # Simulate short startup
        env = FakeEnv()
        env._assigned_display = "0"
        return env

    handler.create_env = fake_create

    # Create 4 sessions (capacity=2, so 2 will queue)
    connect_tasks = []
    for i in range(4):
        connect_tasks.append(handler.connect({"eval_set": "base"}, seed=i))
    await asyncio.gather(*connect_tasks)

    assert len(handler._sessions) == 4

    # Wait for first 2 to be ready
    await asyncio.sleep(0.1)
    ready_count = sum(1 for ctx in handler._sessions.values() if ctx.env is not None)
    assert ready_count == 2, f"Expected 2 ready, got {ready_count}"

    # Close 1 session → frees permit → queued session should get env from pool
    first_sid = None
    for sid, ctx in handler._sessions.items():
        if ctx.env is not None:
            first_sid = sid
            break
    await handler._handle_close(handler._sessions[first_sid])

    await asyncio.sleep(0.05)  # Let queued task run

    # Now should have 2 ready (1 original + 1 newly unblocked that reused pool)
    ready_count = sum(1 for ctx in handler._sessions.values() if ctx.env is not None)
    assert ready_count == 2, f"Expected 2 ready after close+reuse, got {ready_count}"
    # Pool should have been used (one env went in, one came out)
    assert FakeEnv._count <= 3, f"Should reuse pool, only created {FakeEnv._count}"

    # Close another → unblock last queued session too
    second_sid = None
    for sid, ctx in handler._sessions.items():
        if ctx.env is not None:
            second_sid = sid
            break
    await handler._handle_close(handler._sessions[second_sid])
    await asyncio.sleep(0.05)

    # Both remaining sessions should now be ready (last queued got unblocked)
    ready_count = sum(1 for ctx in handler._sessions.values() if ctx.env is not None)
    assert ready_count == 2, f"Expected 2 ready, got {ready_count}"

    # Cleanup
    for sid in list(handler._sessions.keys()):
        ctx = handler._sessions[sid]
        await handler._handle_close(ctx)

    total_created = FakeEnv._count
    print(f"PASS: test_no_deadlock_with_queuing (created {total_created} envs for 4 sessions)")


async def test_batch_cycle():
    """Simulate 2 training batches: batch_size=4, capacity=2.
    Second batch should reuse all pooled envs."""
    FakeEnv._count = 0
    handler = EbAlfredHandler(
        x_displays=["0"],
        capacity=2,
        startup_concurrency=2,
        pool_size=2,
    )

    async def fake_create(config):
        await asyncio.sleep(0.05)  # Simulate startup
        env = FakeEnv()
        env._assigned_display = "0"
        return env

    handler.create_env = fake_create

    async def run_episode(handler, seed):
        """Simulate one episode: connect → wait ready → close."""
        result = await handler.connect({"eval_set": "base"}, seed=seed)
        sid = result.data["session_id"]
        ctx = handler._sessions[sid]
        if hasattr(ctx, '_ready') and ctx._ready:
            await ctx._ready.wait()
        # Simulate some work
        await asyncio.sleep(0.02)
        await handler._handle_close(ctx)

    # Batch 1: 4 episodes
    t0 = time.time()
    await asyncio.gather(*[run_episode(handler, i) for i in range(4)])
    batch1_time = time.time() - t0
    batch1_created = FakeEnv._count

    assert len(handler._env_pool) == 2, f"Expected 2 pooled after batch 1, got {len(handler._env_pool)}"

    # Batch 2: 4 more episodes → should reuse pool
    t0 = time.time()
    await asyncio.gather(*[run_episode(handler, i + 100) for i in range(4)])
    batch2_time = time.time() - t0
    batch2_created = FakeEnv._count - batch1_created

    print(f"  Batch 1: created {batch1_created} envs, took {batch1_time:.3f}s")
    print(f"  Batch 2: created {batch2_created} envs, took {batch2_time:.3f}s")
    assert batch2_created == 0, f"Batch 2 should create 0 new envs, created {batch2_created}"

    # Cleanup
    await handler.aclose()
    print("PASS: test_batch_cycle")


async def test_preload():
    """Verify preload fills the pool before any client connects."""
    FakeEnv._count = 0
    handler = EbAlfredHandler(
        x_displays=["0"],
        capacity=4,
        startup_concurrency=4,
        pool_size=4,
    )

    async def fake_create(config):
        await asyncio.sleep(0.01)
        env = FakeEnv()
        env._assigned_display = "0"
        return env

    handler.create_env = fake_create

    # Preload 4 envs
    await handler.preload(4, {"eval_set": "base"})
    assert len(handler._env_pool) == 4, f"Expected 4 preloaded, got {len(handler._env_pool)}"
    assert FakeEnv._count == 4

    # Connect 4 sessions → all should reuse from pool instantly
    handler._ensure_semaphore()
    for i in range(4):
        await handler.connect({"eval_set": "base"}, seed=i)

    for ctx in handler._sessions.values():
        if hasattr(ctx, '_ready') and ctx._ready:
            await ctx._ready.wait()

    assert FakeEnv._count == 4, f"Should reuse preloaded, got {FakeEnv._count}"
    assert len(handler._env_pool) == 0

    await handler.aclose()
    print("PASS: test_preload")


async def test_preload_capped_by_pool_size():
    """Preload(n) should be capped at pool_size."""
    FakeEnv._count = 0
    handler = EbAlfredHandler(
        x_displays=["0"],
        capacity=8,
        startup_concurrency=8,
        pool_size=3,
    )

    async def fake_create(config):
        env = FakeEnv()
        env._assigned_display = "0"
        return env

    handler.create_env = fake_create

    await handler.preload(10, {"eval_set": "base"})  # Request 10, capped to 3
    assert len(handler._env_pool) == 3, f"Expected 3 (capped), got {len(handler._env_pool)}"

    await handler.aclose()
    print("PASS: test_preload_capped_by_pool_size")


async def main():
    await test_pool_basic()
    await test_pool_overflow()
    await test_no_deadlock_with_queuing()
    await test_batch_cycle()
    await test_preload()
    await test_preload_capped_by_pool_size()
    print("\nAll tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
