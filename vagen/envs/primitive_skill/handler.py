"""
Handler for the primitive_skill (ManiSkill) environment.

ManiSkill uses GPU rendering (render_backend="gpu"), so envs need
to be created on the server side. The handler manages session lifecycle
and GPU device assignment.
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
    Handler with GPU load balancing for ManiSkill environments.

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
        """
        Args:
            devices: GPU device IDs. Defaults to [0].
            session_timeout: Idle timeout before session cleanup.
            max_envs: Max concurrent envs across all GPUs.
            acquire_timeout: Max wait time for an env slot (seconds).
        """
        super().__init__(session_timeout=session_timeout, max_sessions=0)
        self.devices = devices or [0]
        self.max_envs = max_envs
        self.acquire_timeout = acquire_timeout

        self._env_slots = asyncio.Semaphore(max_envs)
        self._active: Dict[int, int] = {d: 0 for d in self.devices}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _total_active(self) -> int:
        return sum(self._active.values())

    def _stats_str(self) -> str:
        return f"active={dict(self._active)} total={self._total_active()}/{self.max_envs}"

    # ------------------------------------------------------------------
    # GPU assignment
    # ------------------------------------------------------------------

    def _pick_device(self) -> int:
        """Pick GPU with fewest active envs."""
        return min(self._active, key=self._active.get)

    # ------------------------------------------------------------------
    # BaseGymHandler abstract method
    # ------------------------------------------------------------------

    async def create_env(self, env_config: Dict[str, Any]) -> PrimitiveSkillEnv:
        return PrimitiveSkillEnv(env_config)

    # ------------------------------------------------------------------
    # Connect (override for GPU assignment + slot management)
    # ------------------------------------------------------------------

    async def connect(
        self, env_config: Dict[str, Any], seed: Optional[int] = None
    ) -> HandlerResult:
        device = self._pick_device()
        self._active[device] += 1

        try:
            await asyncio.wait_for(self._env_slots.acquire(), timeout=self.acquire_timeout)
        except asyncio.TimeoutError:
            self._active[device] = max(0, self._active[device] - 1)
            raise RuntimeError(
                f"No env slot available after {self.acquire_timeout}s ({self._stats_str()})"
            )

        try:
            env_config_with_gpu = {**env_config, "gpu_device": device}
            env = PrimitiveSkillEnv(env_config_with_gpu)
        except BaseException:
            self._active[device] = max(0, self._active[device] - 1)
            self._env_slots.release()
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
            obs, info = await self._sessions[session_id].env.reset(seed)
            result_data["obs"] = obs.get("obs_str", "")
            result_data["info"] = info
            images = self._extract_images(obs)
            return HandlerResult(data=result_data, images=images)

        return HandlerResult(data=result_data)

    # ------------------------------------------------------------------
    # Close (release slot)
    # ------------------------------------------------------------------

    async def _handle_close(self, ctx: SessionContext) -> HandlerResult:
        env: PrimitiveSkillEnv = ctx.env
        device = env.cfg.gpu_device

        await ctx.env.close()
        self._active[device] = max(0, self._active[device] - 1)
        self._env_slots.release()

        del self._sessions[ctx.session_id]
        LOGGER.info(
            f"[PrimSkillHandler] Session {ctx.session_id[:8]} closed "
            f"GPU {device} ({self._stats_str()})"
        )
        return HandlerResult(data={"closed": True})

    # ------------------------------------------------------------------
    # Stats endpoint
    # ------------------------------------------------------------------

    def get_session_stats(self) -> Dict[str, Any]:
        stats = super().get_session_stats()
        stats["max_envs"] = self.max_envs
        stats["active_per_device"] = dict(self._active)
        stats["total_active"] = self._total_active()
        return stats

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def aclose(self):
        await super().aclose()
        LOGGER.info("[PrimSkillHandler] All sessions closed")
