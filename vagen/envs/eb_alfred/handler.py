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
    """Handler for EB-ALFRED with capacity-based queuing and multi-GPU load balancing.

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
            **kwargs: Passed to BaseGymHandler (session_timeout, max_sessions).
        """
        super().__init__(**kwargs)
        self._x_displays = x_displays if x_displays is not None else detect_gpu_displays()
        self._pending_counts: Dict[str, int] = {d: 0 for d in self._x_displays}
        self._capacity = capacity
        self._startup_concurrency = startup_concurrency
        # Defer semaphore creation: it must be created on the running event loop,
        # not during __init__ (which runs before uvicorn starts the loop).
        self._capacity_sem: Optional[asyncio.Semaphore] = None
        self._startup_sem: Optional[asyncio.Semaphore] = None
        LOGGER.info(
            f"[Handler] Using X displays: {self._x_displays}, "
            f"capacity={capacity if capacity > 0 else 'unlimited'}, "
            f"startup_concurrency={startup_concurrency if startup_concurrency > 0 else 'unlimited'}"
        )

    def _ensure_semaphore(self) -> None:
        """Lazily create the capacity and startup semaphores on the running event loop."""
        if self._capacity_sem is None and self._capacity > 0:
            self._capacity_sem = asyncio.Semaphore(self._capacity)
        if self._startup_sem is None and self._startup_concurrency > 0 and self._capacity > 0:
            self._startup_sem = asyncio.Semaphore(self._startup_concurrency)

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
        """Background task: acquire capacity slot, then create env.

        Two-phase acquisition:
          1. capacity_sem  – limits total running envs (held for env lifetime)
          2. startup_sem   – limits concurrent Unity startups (held only during
                             EbAlfred.__init__, released as soon as the process
                             is running)
        This prevents a "startup storm" when many capacity slots open at once.
        """
        try:
            LOGGER.info(f"[Handler] Session {ctx.session_id} waiting for capacity slot...")
            await self._capacity_sem.acquire()
            ctx._holds_slot = True
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

    async def _handle_close(self, ctx: SessionContext) -> HandlerResult:
        """Close env and release capacity slot."""
        try:
            if ctx.env is not None:
                await ctx.env.close()
        except Exception as e:
            LOGGER.error(f"[Handler] Error closing env for session {ctx.session_id}: {e}")
        finally:
            if ctx._holds_slot and self._capacity_sem is not None:
                self._capacity_sem.release()
                ctx._holds_slot = False
            self._sessions.pop(ctx.session_id, None)

        n_active = sum(1 for s in self._sessions.values() if s.env is not None)
        n_queued = len(self._sessions) - n_active
        LOGGER.info(
            f"[Handler] Closed session {ctx.session_id} "
            f"(active={n_active}, queued={n_queued}, capacity={self._capacity})"
        )
        return HandlerResult(data={"closed": True})

    def get_session_stats(self) -> Dict[str, Any]:
        """Session stats with active/queued breakdown."""
        stats = super().get_session_stats()
        n_active = sum(1 for s in self._sessions.values() if s.env is not None)
        stats["active"] = n_active
        stats["queued"] = len(self._sessions) - n_active
        stats["capacity"] = self._capacity if self._capacity > 0 else "unlimited"
        for s in stats.get("sessions", []):
            sid = s["session_id"]
            ctx = self._sessions.get(sid)
            s["status"] = "active" if (ctx and ctx.env is not None) else "queued"
        return stats

    async def _cleanup_loop(self):
        """Cleanup timed-out sessions, releasing capacity slots."""
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
                    try:
                        if ctx.env is not None:
                            await ctx.env.close()
                    except Exception as e:
                        LOGGER.error(f"[Handler] Cleanup error {session_id}: {e}")
                    finally:
                        if ctx._holds_slot and self._capacity_sem is not None:
                            self._capacity_sem.release()
                            ctx._holds_slot = False
                        self._sessions.pop(session_id, None)
            except asyncio.CancelledError:
                break
            except Exception as e:
                LOGGER.error(f"[Handler] Cleanup loop error: {e}")

    async def aclose(self):
        """Shutdown: close all sessions, release all capacity slots."""
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
        LOGGER.info("[Handler] All sessions closed")
