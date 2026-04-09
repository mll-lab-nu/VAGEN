"""
Handler for the primitive_skill (ManiSkill) environment.

Resource model:
    max_envs = total alive envs (active sessions + cached idle envs).
    This bounds GPU memory usage. When at capacity with no cached envs
    available, new requests wait until an env is freed.

Caching:
    On client close, the env is NOT destroyed — it stays alive in a
    per-GPU, per-env_id cache. On next connect:
    - Same env_id: reuse cached env, just reset with new seed (~fast)
    - Different env_id: destroy cached env, create new one
    - No cache available: create new env
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from vagen.envs_remote.handler import BaseGymHandler, HandlerResult, SessionContext
from .primitive_skill_env import PrimitiveSkillEnv

LOGGER = logging.getLogger(__name__)


class PrimitiveSkillHandler(BaseGymHandler):
    """
    Handler with GPU load balancing and env caching for ManiSkill environments.

    Usage:
        handler = PrimitiveSkillHandler(devices=[0, 1], max_envs=64)
        app = GymService(handler).build()
    """

    def __init__(
        self,
        devices: Optional[List[int]] = None,
        session_timeout: float = 3600.0,
        max_envs: int = 64,
        acquire_timeout: float = 300.0,
    ):
        super().__init__(session_timeout=session_timeout, max_sessions=0)
        self.devices = devices or [0]
        self.max_envs = max_envs
        self.acquire_timeout = acquire_timeout

        self._env_slots = asyncio.Semaphore(max_envs)
        self._active: Dict[int, int] = {d: 0 for d in self.devices}
        # Per-GPU cache: list of (env_id, PrimitiveSkillEnv)
        self._cache: Dict[int, List[Tuple[str, PrimitiveSkillEnv]]] = {d: [] for d in self.devices}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _total_active(self) -> int:
        return sum(self._active.values())

    def _total_cached(self) -> int:
        return sum(len(pool) for pool in self._cache.values())

    def _total_alive(self) -> int:
        return self._total_active() + self._total_cached()

    def _stats_str(self) -> str:
        active = dict(self._active)
        cached = {d: len(pool) for d, pool in self._cache.items()}
        return f"active={active} cached={cached} total={self._total_alive()}/{self.max_envs}"

    # ------------------------------------------------------------------
    # GPU assignment
    # ------------------------------------------------------------------

    def _pick_device(self) -> int:
        """Pick GPU with fewest active envs."""
        return min(self._active, key=self._active.get)

    def _pop_cached(self, device: int, env_id: str) -> Optional[PrimitiveSkillEnv]:
        """Pop cached env from device, preferring same env_id."""
        pool = self._cache[device]
        if not pool:
            return None
        # Prefer same env_id (just reset, no recreation)
        for i, (cached_id, env) in enumerate(pool):
            if cached_id == env_id:
                pool.pop(i)
                LOGGER.info(f"[PrimSkillHandler] Cache hit: env_id={env_id} GPU {device}")
                return env
        # Different env_id: destroy cached env, free the slot for a new one
        _, old_env = pool.pop()
        LOGGER.info(f"[PrimSkillHandler] Cache evict (diff env_id) GPU {device}")
        # Close the old env synchronously-ish via fire-and-forget
        asyncio.create_task(self._close_env_quiet(old_env))
        # Slot is freed by _close_env_quiet, but we re-acquire below
        return None

    def _pop_any_cached(self) -> Optional[Tuple[int, PrimitiveSkillEnv]]:
        """Pop any cached env from any device (to free a slot)."""
        for device, pool in self._cache.items():
            if pool:
                _, env = pool.pop()
                return device, env
        return None

    async def _close_env_quiet(self, env: PrimitiveSkillEnv):
        """Close an env and release its slot, swallowing errors."""
        try:
            await env.close()
        except Exception as e:
            LOGGER.warning(f"[PrimSkillHandler] Error closing env: {e}")
        self._env_slots.release()

    # ------------------------------------------------------------------
    # BaseGymHandler abstract method
    # ------------------------------------------------------------------

    async def create_env(self, env_config: Dict[str, Any]) -> PrimitiveSkillEnv:
        return PrimitiveSkillEnv(env_config)

    # ------------------------------------------------------------------
    # Env acquisition (cache → create)
    # ------------------------------------------------------------------

    async def _acquire_env(
        self, device: int, env_config: Dict[str, Any]
    ) -> PrimitiveSkillEnv:
        """Get an env: try cache first, then create new."""
        env_id = env_config.get("env_id", "AlignTwoCube")

        # Try cache on target device (same env_id = reuse, no new slot needed)
        env = self._pop_cached(device, env_id)
        if env is not None:
            return env

        # Need a new slot
        acquired = False
        try:
            await asyncio.wait_for(self._env_slots.acquire(), timeout=0.0)
            acquired = True
        except (asyncio.TimeoutError, ValueError):
            pass

        if not acquired:
            # At capacity — evict a cached env to free a slot
            evicted = self._pop_any_cached()
            if evicted is not None:
                evict_device, evict_env = evicted
                await evict_env.close()
                self._env_slots.release()
                LOGGER.info(f"[PrimSkillHandler] Evicted cached env GPU {evict_device}")

            try:
                await asyncio.wait_for(self._env_slots.acquire(), timeout=self.acquire_timeout)
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"No env slot available after {self.acquire_timeout}s ({self._stats_str()})"
                )

        # Create new env
        env_config_with_gpu = {**env_config, "gpu_device": device}
        env = PrimitiveSkillEnv(env_config_with_gpu)
        LOGGER.info(f"[PrimSkillHandler] New env {env_id} GPU {device} ({self._stats_str()})")
        return env

    # ------------------------------------------------------------------
    # Connect
    # ------------------------------------------------------------------

    async def connect(
        self, env_config: Dict[str, Any], seed: Optional[int] = None
    ) -> HandlerResult:
        device = self._pick_device()
        self._active[device] += 1

        try:
            env = await self._acquire_env(device, env_config)
        except BaseException:
            self._active[device] = max(0, self._active[device] - 1)
            raise

        session_id = uuid.uuid4().hex
        self._sessions[session_id] = SessionContext(
            session_id=session_id,
            env=env,
            created_at=time.time(),
            last_access=time.time(),
        )

        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        LOGGER.info(f"[PrimSkillHandler] Session {session_id[:8]} GPU {device} ({self._stats_str()})")

        result_data = {"session_id": session_id}
        if seed is not None:
            try:
                obs, info = await self._sessions[session_id].env.reset(seed)
            except BaseException:
                try:
                    await env.close()
                except Exception:
                    pass
                self._sessions.pop(session_id, None)
                self._active[device] = max(0, self._active[device] - 1)
                self._env_slots.release()
                raise
            result_data["obs"] = obs.get("obs_str", "")
            result_data["info"] = info
            images = self._extract_images(obs)
            return HandlerResult(data=result_data, images=images)

        return HandlerResult(data=result_data)

    # ------------------------------------------------------------------
    # Close (cache instead of destroy)
    # ------------------------------------------------------------------

    async def _handle_close(self, ctx: SessionContext) -> HandlerResult:
        env: PrimitiveSkillEnv = ctx.env
        device = env.cfg.gpu_device
        env_id = env.cfg.env_id

        # Move from active to cache (env stays alive, no slot change)
        self._active[device] = max(0, self._active[device] - 1)
        self._cache[device].append((env_id, env))

        self._sessions.pop(ctx.session_id, None)
        LOGGER.info(
            f"[PrimSkillHandler] Session {ctx.session_id[:8]} -> cached "
            f"env_id={env_id} GPU {device} ({self._stats_str()})"
        )
        return HandlerResult(data={"closed": True})

    # ------------------------------------------------------------------
    # Stats endpoint
    # ------------------------------------------------------------------

    def get_session_stats(self) -> Dict[str, Any]:
        stats = super().get_session_stats()
        stats["max_envs"] = self.max_envs
        stats["active_per_device"] = dict(self._active)
        stats["cached_per_device"] = {d: len(pool) for d, pool in self._cache.items()}
        stats["total_alive"] = self._total_alive()
        return stats

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def aclose(self):
        for device, entries in self._cache.items():
            for _, env in entries:
                try:
                    await env.close()
                except Exception as e:
                    LOGGER.error(f"[PrimSkillHandler] Error closing cached env GPU {device}: {e}")
            entries.clear()
        LOGGER.info("[PrimSkillHandler] All cached envs closed")
        await super().aclose()
