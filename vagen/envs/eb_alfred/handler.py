"""
EB-ALFRED handler for the remote gym environment service.

This is the only component that needs customization.
It implements create_env() to instantiate EB-ALFRED environments
with automatic multi-GPU load balancing.

Capacity control:
    When ``capacity`` is set (> 0), at most that many Unity environments
    run concurrently.  Extra ``/connect`` requests are accepted immediately
    (returning session_id so the client does NOT retry) and queued.
    Environments are created in the background as slots free up -- each
    independently, not in batches.
"""

import asyncio
import logging
import random
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vagen.envs_remote.handler import BaseGymHandler, HandlerResult, SessionContext
from .eb_alfred_env import EbAlfred

LOGGER = logging.getLogger(__name__)


@dataclass
class _DeferredSessionContext(SessionContext):
    """SessionContext with extra fields for capacity-based deferred env creation."""

    env_config: Dict[str, Any] = field(default_factory=dict)
    _ready: Optional[asyncio.Event] = field(default=None, repr=False)
    _holds_slot: bool = field(default=False, repr=False)
    _error: Optional[str] = field(default=None, repr=False)


def detect_gpu_displays() -> List[str]:
    """Auto-detect available GPUs via nvidia-smi, return display list.

    Assumes display :i maps to GPU i (standard convention for
    multi-GPU X server setups with Xvfb or xinit per GPU).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            indices = [
                line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            if indices:
                return indices
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ["0"]


class EbAlfredHandler(BaseGymHandler):
    """Handler for EB-ALFRED with capacity-based queuing, multi-GPU load
    balancing, and **persistent env pooling**.

    Env pooling (enabled by default):
      - When a session closes, its Unity process is returned to an idle
        pool instead of being destroyed.
      - When a new session connects, an idle env is taken from the pool
        (near-instant) instead of spawning a new Unity process (~90 s).
      - The pool implicitly holds capacity-semaphore permits: a pooled
        env still counts against ``capacity`` because the Unity process
        is alive and consuming GPU memory.
      - Set ``pool_size=0`` together with ``capacity=0`` to disable
        pooling entirely (original behaviour).

    When capacity > 0:
      - /connect returns session_id immediately (env creation is deferred)
      - A background task waits for a capacity slot, then creates the env
      - /call (reset/step) blocks until the env is ready
      - /call (close) releases the slot so the next queued env can start
      - Each env is independent: no batch waiting

    When capacity = 0 (default):
      - Original behaviour: env is created synchronously on /connect

    startup_concurrency controls how many Unity processes may be in the
    startup phase simultaneously, independent of capacity.  This prevents
    a "startup storm" when many sessions are queued and capacity slots
    become available at the same time.  E.g. capacity=64,
    startup_concurrency=8 means up to 64 envs run concurrently but at
    most 8 are initialising at any given moment.
    """

    def __init__(
        self,
        x_displays: Optional[List[str]] = None,
        capacity: int = 16,
        startup_concurrency: int = 8,
        pool_size: int = -1,
        **kwargs,
    ):
        """
        Args:
            x_displays: List of X display IDs to use (e.g. ["0", "1"]).
                         None = auto-detect GPUs via nvidia-smi.
            capacity: Max concurrently running Unity environments (0 = unlimited).
            startup_concurrency: Max Unity processes that may be starting up at
                once (0 = unlimited).  Prevents CPU spikes when many capacity
                slots open simultaneously.  Ignored when capacity = 0.
            pool_size: Max idle envs kept alive in the pool.  -1 (default) =
                same as capacity (keep every env alive).  0 = disable pooling.
            **kwargs: Passed to BaseGymHandler (session_timeout, max_sessions).
        """
        super().__init__(**kwargs)
        self._x_displays = x_displays if x_displays is not None else detect_gpu_displays()
        self._pending_counts: Dict[str, int] = {d: 0 for d in self._x_displays}
        self._capacity = capacity
        self._startup_concurrency = startup_concurrency
        # Env pool: idle Unity processes available for immediate reuse.
        # Each pooled env implicitly holds one capacity-semaphore permit.
        self._pool_size = pool_size if pool_size >= 0 else max(capacity, 1)
        self._env_pool: List[Any] = []
        # Defer semaphore creation: it must be created on the running event loop,
        # not during __init__ (which runs before uvicorn starts the loop).
        self._capacity_sem: Optional[asyncio.Semaphore] = None
        self._startup_sem: Optional[asyncio.Semaphore] = None
        LOGGER.info(
            f"[Handler] Using X displays: {self._x_displays}, "
            f"capacity={capacity if capacity > 0 else 'unlimited'}, "
            f"startup_concurrency={startup_concurrency if startup_concurrency > 0 else 'unlimited'}, "
            f"pool_size={self._pool_size}"
        )

    def _ensure_semaphore(self) -> None:
        """Lazily create the capacity and startup semaphores on the running event loop."""
        if self._capacity_sem is None and self._capacity > 0:
            self._capacity_sem = asyncio.Semaphore(self._capacity)
        if self._startup_sem is None and self._startup_concurrency > 0 and self._capacity > 0:
            self._startup_sem = asyncio.Semaphore(self._startup_concurrency)

    async def preload(self, n: int, env_config: Dict[str, Any]) -> None:
        """Pre-create *n* environments and place them in the idle pool.

        Called once during server startup so that the first training batch
        gets instant env assignment instead of waiting ~90 s per Unity
        process.  Envs are created with ``startup_concurrency`` throttling.

        Args:
            n: Number of envs to pre-create (capped at pool_size).
            env_config: Config dict forwarded to ``create_env()``.
        """
        n = min(n, self._pool_size)
        if n <= 0:
            return

        sem = asyncio.Semaphore(self._startup_concurrency or n)

        async def _create_one(idx: int):
            async with sem:
                LOGGER.info(f"[Preload] Creating env {idx + 1}/{n} ...")
                env = await self.create_env(env_config)
                return env

        t0 = time.time()
        LOGGER.info(f"[Preload] Pre-creating {n} envs (concurrency={self._startup_concurrency or n}) ...")
        envs = await asyncio.gather(*[_create_one(i) for i in range(n)])
        self._env_pool.extend(envs)
        elapsed = time.time() - t0
        LOGGER.info(f"[Preload] {n} envs ready in {elapsed:.1f}s (pool: {len(self._env_pool)})")

    def _least_loaded_display(self) -> str:
        """Pick the display with the fewest active + pending sessions.

        On ties, randomly choose among the least-loaded displays to
        avoid always funnelling to the first GPU.
        """
        counts = {d: self._pending_counts.get(d, 0) for d in self._x_displays}
        for ctx in self._sessions.values():
            d = getattr(ctx.env, "_assigned_display", None)
            if d in counts:
                counts[d] += 1
        min_count = min(counts.values())
        candidates = [d for d, c in counts.items() if c == min_count]
        chosen = random.choice(candidates)
        LOGGER.debug(f"[Handler] GPU load: {counts}, assigning display :{chosen}")
        return chosen

    async def create_env(self, env_config: Dict[str, Any]) -> Any:
        """
        Create an EbAlfred environment on the least-loaded GPU.

        AI2-THOR startup is blocking, so we offload to a thread.
        """
        display = self._least_loaded_display()
        self._pending_counts[display] = self._pending_counts.get(display, 0) + 1
        env_config = {**env_config, "x_display": display}

        try:
            env = await asyncio.to_thread(EbAlfred, env_config)
        finally:
            self._pending_counts[display] = max(0, self._pending_counts.get(display, 1) - 1)

        env._assigned_display = display
        LOGGER.info(
            f"[Handler] Created env on display :{display} "
            f"(GPU load: { {d: sum(1 for c in self._sessions.values() if getattr(c.env, '_assigned_display', None) == d) for d in self._x_displays} })"
        )
        return env

    # ------------------------------------------------------------------
    # Capacity-aware connect / call / close
    # ------------------------------------------------------------------

    async def connect(
        self, env_config: Dict[str, Any], seed: Optional[int] = None
    ) -> HandlerResult:
        """Accept session immediately; defer env creation if capacity-limited."""
        self._ensure_semaphore()
        # No capacity limit → use original behaviour
        if self._capacity_sem is None:
            return await super().connect(env_config, seed=seed)

        # Check total session limit
        if self.max_sessions > 0 and len(self._sessions) >= self.max_sessions:
            raise RuntimeError(
                f"Max sessions limit reached ({self.max_sessions}). "
                f"Please try again later or close existing sessions."
            )

        session_id = uuid.uuid4().hex
        ready_event = asyncio.Event()
        ctx = _DeferredSessionContext(
            session_id=session_id,
            env=None,
            created_at=time.time(),
            last_access=time.time(),
            env_config=env_config,
            _ready=ready_event,
        )
        self._sessions[session_id] = ctx

        # Start cleanup task if not running
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Fire-and-forget: wait for slot → create env
        asyncio.create_task(self._deferred_create(ctx))

        n_active = sum(1 for s in self._sessions.values() if s.env is not None)
        n_queued = len(self._sessions) - n_active
        # Estimate wait: (queued_ahead / capacity) * avg_episode_time
        # Use a rough estimate of 15s per env creation cycle
        estimated_wait = max(0, (n_queued - 1)) / max(1, self._capacity) * 15

        LOGGER.info(
            f"[Handler] Session {session_id} queued "
            f"(active={n_active}, queued={n_queued}, capacity={self._capacity}, "
            f"est_wait={estimated_wait:.0f}s)"
        )

        return HandlerResult(data={
            "session_id": session_id,
            "status": "queued",
            "estimated_wait_s": estimated_wait,
        })

    async def _deferred_create(self, ctx: _DeferredSessionContext) -> None:
        """Background task: acquire capacity slot, then reuse pooled env or create new.

        Always acquires a capacity permit first (so queued sessions unblock
        as soon as any session releases its permit via close/pool).  After
        acquiring the permit, checks the pool for an idle env:
          - Pool hit  → instant reuse, skip Unity startup
          - Pool miss → two-phase creation with startup_sem throttle
        """
        try:
            LOGGER.info(f"[Handler] Session {ctx.session_id} waiting for capacity slot...")
            await self._capacity_sem.acquire()
            ctx._holds_slot = True

            # ---- fast path: reuse from pool ----
            if self._env_pool:
                ctx.env = self._env_pool.pop()
                LOGGER.info(
                    f"[Handler] Session {ctx.session_id} reused pooled env "
                    f"(pool: {len(self._env_pool)} remaining)"
                )
                return

            # ---- slow path: create new Unity process ----
            LOGGER.info(f"[Handler] Session {ctx.session_id} acquired capacity slot, waiting for startup slot...")

            if self._startup_sem is not None:
                await self._startup_sem.acquire()

            LOGGER.info(f"[Handler] Session {ctx.session_id} starting Unity...")
            try:
                ctx.env = await self.create_env(ctx.env_config)
            finally:
                if self._startup_sem is not None:
                    self._startup_sem.release()

            LOGGER.info(f"[Handler] Session {ctx.session_id} env ready")
        except Exception as e:
            LOGGER.error(f"[Handler] Session {ctx.session_id} env creation failed: {e}")
            ctx._error = str(e)
            if ctx._holds_slot:
                self._capacity_sem.release()
                ctx._holds_slot = False
        finally:
            ctx._ready.set()

    async def _wait_env_ready(self, ctx: _DeferredSessionContext) -> None:
        """Block until env is created (called by call() before dispatching)."""
        if ctx._ready is not None and not ctx._ready.is_set():
            LOGGER.info(f"[Handler] Session {ctx.session_id} caller waiting for env...")
            await ctx._ready.wait()
        if ctx.env is None:
            error = ctx._error or "Environment creation failed"
            raise RuntimeError(f"Session {ctx.session_id}: {error}")

    async def call(
        self,
        session_id: str,
        method: str,
        params: Dict[str, Any],
        images,
    ) -> HandlerResult:
        """Dispatch method call; wait for env if still queued."""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")

        ctx = self._sessions[session_id]
        ctx.last_access = time.time()

        # Wait for env to be ready (no-op if capacity=0 / already ready)
        if ctx.env is None and method != "close":
            await self._wait_env_ready(ctx)

        return await super().call(session_id, method, params, images)

    async def _release_env(self, ctx: SessionContext) -> None:
        """Pool or close the env, then always release the capacity permit.

        The pool is a pure cache — it does NOT hold capacity permits.
        This ensures queued sessions always unblock when a session closes,
        regardless of whether the env was pooled or destroyed.

        When a new session later acquires a permit, it checks the pool
        first (fast path) before creating a new Unity process (slow path).
        """
        try:
            if ctx.env is not None and len(self._env_pool) < self._pool_size:
                # Return to pool — keep Unity alive for reuse
                self._env_pool.append(ctx.env)
                LOGGER.info(
                    f"[Handler] Session {ctx.session_id} returned env to pool "
                    f"(pool: {len(self._env_pool)}/{self._pool_size})"
                )
                ctx.env = None
            else:
                # Pool full (or no env) — actually close Unity
                if ctx.env is not None:
                    await ctx.env.close()
                    ctx.env = None
        except Exception as e:
            LOGGER.error(f"[Handler] Error releasing env for session {ctx.session_id}: {e}")
        finally:
            # Always release capacity permit so queued sessions can proceed
            if ctx._holds_slot and self._capacity_sem is not None:
                self._capacity_sem.release()
                ctx._holds_slot = False

    async def _handle_close(self, ctx: SessionContext) -> HandlerResult:
        """Close session: return env to pool or destroy it."""
        await self._release_env(ctx)
        self._sessions.pop(ctx.session_id, None)

        n_active = sum(1 for s in self._sessions.values() if s.env is not None)
        n_queued = len(self._sessions) - n_active
        LOGGER.info(
            f"[Handler] Closed session {ctx.session_id} "
            f"(active={n_active}, queued={n_queued}, "
            f"pool={len(self._env_pool)}, capacity={self._capacity})"
        )
        return HandlerResult(data={"closed": True})

    def get_session_stats(self) -> Dict[str, Any]:
        """Session stats with active/queued/pool breakdown."""
        stats = super().get_session_stats()
        n_active = sum(1 for s in self._sessions.values() if s.env is not None)
        stats["active"] = n_active
        stats["queued"] = len(self._sessions) - n_active
        stats["capacity"] = self._capacity if self._capacity > 0 else "unlimited"
        stats["pool_size"] = len(self._env_pool)
        stats["pool_max"] = self._pool_size
        for s in stats.get("sessions", []):
            sid = s["session_id"]
            ctx = self._sessions.get(sid)
            s["status"] = "active" if (ctx and ctx.env is not None) else "queued"
        return stats

    async def _cleanup_loop(self):
        """Cleanup timed-out sessions, releasing capacity slots or pooling envs."""
        while True:
            try:
                await asyncio.sleep(60)
                now = time.time()
                to_remove = []
                for session_id, ctx in self._sessions.items():
                    if now - ctx.last_access > self.session_timeout:
                        to_remove.append(session_id)
                        LOGGER.warning(f"[Handler] Session {session_id} timed out")

                for session_id in to_remove:
                    ctx = self._sessions.get(session_id)
                    if ctx is None:
                        continue
                    await self._release_env(ctx)
                    self._sessions.pop(session_id, None)
            except asyncio.CancelledError:
                break
            except Exception as e:
                LOGGER.error(f"[Handler] Cleanup loop error: {e}")

    async def aclose(self):
        """Shutdown: close all sessions and pooled envs, release all capacity slots."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        async def _close_one(sid: str, ctx: SessionContext):
            try:
                if ctx.env is not None:
                    await ctx.env.close()
            except Exception as e:
                LOGGER.error(f"[Handler] Shutdown close error {sid}: {e}")
            finally:
                if ctx._holds_slot and self._capacity_sem is not None:
                    self._capacity_sem.release()
                    ctx._holds_slot = False

        if self._sessions:
            await asyncio.gather(
                *(_close_one(sid, ctx) for sid, ctx in self._sessions.items())
            )
        self._sessions.clear()

        # Close all pooled envs (they don't hold capacity permits)
        n_pooled = len(self._env_pool)
        for env in self._env_pool:
            try:
                await env.close()
            except Exception as e:
                LOGGER.error(f"[Handler] Shutdown pool close error: {e}")
        self._env_pool.clear()

        LOGGER.info(f"[Handler] All sessions closed, {n_pooled} pooled envs released")
