"""
Handler for the AI2-THOR navigation environment.

Resource model:
    max_envs = total alive envs (active sessions + cached idle envs).
    This bounds GPU memory usage. When at capacity with no cached envs
    available, new requests wait until an env is freed.

Caching:
    On client close, the env is NOT destroyed — it stays alive in a
    per-GPU cache. On next connect, a cached env is reused:
    - Same scene (~0.15s teleport-only)
    - Different scene (~0.2s scene reload, skip controller creation)
    - No cache available: create new env (~2.3s)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from vagen.envs_remote.handler import BaseGymHandler, HandlerResult, SessionContext
from .navigation_env import NavigationEnv

LOGGER = logging.getLogger(__name__)


class NavigationHandler(BaseGymHandler):
    """
    Handler with GPU load balancing and env caching.

    Usage:
        handler = NavigationHandler(devices=[0, 1], max_envs=128)
        app = GymService(handler).build()
    """

    def __init__(
        self,
        devices: Optional[List[int]] = None,
        session_timeout: float = 3600.0,
        max_envs: int = 128,
        acquire_timeout: float = 300.0,
    ):
        """
        Args:
            devices: GPU device IDs. Defaults to [0].
            session_timeout: Idle timeout before session cleanup.
            max_envs: Max alive envs across all GPUs (active + cached).
            acquire_timeout: Max wait time for an env slot (seconds).
        """
        # max_sessions=0 disables base class limit (we manage it ourselves)
        super().__init__(session_timeout=session_timeout, max_sessions=0)
        self.devices = devices or [0]
        self.max_envs = max_envs
        self.acquire_timeout = acquire_timeout

        # Semaphore: total alive env slots
        self._env_slots = asyncio.Semaphore(max_envs)
        # Per-GPU state
        self._active: Dict[int, int] = {d: 0 for d in self.devices}
        self._cache: Dict[int, List[Tuple[str, NavigationEnv]]] = {d: [] for d in self.devices}
        # seed → scene index (lazy loaded)
        self._scene_index: Dict[str, List[str]] = {}

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
    # Scene index
    # ------------------------------------------------------------------

    def _get_scene_for_seed(self, eval_set: str, seed: int) -> Optional[str]:
        if eval_set not in self._scene_index:
            path = os.path.join(os.path.dirname(__file__), "assets", f"{eval_set}.json")
            try:
                with open(path) as f:
                    self._scene_index[eval_set] = [t["scene"] for t in json.load(f)["tasks"]]
            except Exception:
                return None
        scenes = self._scene_index.get(eval_set)
        return scenes[seed % len(scenes)] if scenes else None

    # ------------------------------------------------------------------
    # GPU assignment
    # ------------------------------------------------------------------

    def _pick_device(self, preferred_scene: Optional[str] = None) -> int:
        """Pick GPU: prefer one with a matching cached scene, else fewest active."""
        if preferred_scene:
            for device, pool in self._cache.items():
                if any(scene == preferred_scene for scene, _ in pool):
                    return device
        return min(self._active, key=self._active.get)

    def _pop_cached(self, device: int, preferred_scene: Optional[str] = None) -> Optional[NavigationEnv]:
        """Pop cached env from device, preferring same scene."""
        pool = self._cache[device]
        if not pool:
            return None
        if preferred_scene:
            for i, (scene, env) in enumerate(pool):
                if scene == preferred_scene:
                    pool.pop(i)
                    LOGGER.info(f"[NavHandler] Cache hit: scene={preferred_scene} GPU {device}")
                    return env
        _, env = pool.pop()
        LOGGER.info(f"[NavHandler] Cache reuse (diff scene) GPU {device}")
        return env

    def _pop_any_cached(self) -> Optional[Tuple[int, NavigationEnv]]:
        """Pop any cached env from any device (to free a slot for new creation)."""
        for device, pool in self._cache.items():
            if pool:
                _, env = pool.pop()
                return device, env
        return None

    # ------------------------------------------------------------------
    # BaseGymHandler abstract method (not used directly, connect() overrides)
    # ------------------------------------------------------------------

    async def create_env(self, env_config: Dict[str, Any]) -> NavigationEnv:
        return NavigationEnv(env_config)

    # ------------------------------------------------------------------
    # Env acquisition (cache → create)
    # ------------------------------------------------------------------

    async def _acquire_env(
        self, device: int, env_config: Dict[str, Any], seed: Optional[int] = None
    ) -> NavigationEnv:
        """Get an env: try cache first, then create new (with slot management)."""
        preferred_scene = None
        if seed is not None:
            preferred_scene = self._get_scene_for_seed(env_config.get("eval_set", "base"), seed)

        # Try cache on target device (env already alive, no new slot needed)
        env = self._pop_cached(device, preferred_scene)
        if env is not None:
            LOGGER.info(f"[NavHandler] Reuse cached -> active ({self._stats_str()})")
            return env

        # Need a new slot for a new env
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
                LOGGER.info(f"[NavHandler] Evicted cached env GPU {evict_device} to make room")

            try:
                await asyncio.wait_for(self._env_slots.acquire(), timeout=self.acquire_timeout)
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"No env slot available after {self.acquire_timeout}s ({self._stats_str()})"
                )

        env_config_with_gpu = {**env_config, "gpu_device": device}
        env = NavigationEnv(env_config_with_gpu)
        LOGGER.info(f"[NavHandler] New env GPU {device} ({self._stats_str()})")
        return env

    # ------------------------------------------------------------------
    # Connect (override base class for cache-aware env creation)
    # ------------------------------------------------------------------

    async def connect(
        self, env_config: Dict[str, Any], seed: Optional[int] = None
    ) -> HandlerResult:
        # Pick device and increment IMMEDIATELY (before any await) to prevent
        # all concurrent connects from seeing the same counts and picking the same GPU.
        preferred_scene = None
        if seed is not None:
            preferred_scene = self._get_scene_for_seed(env_config.get("eval_set", "base"), seed)
        device = self._pick_device(preferred_scene)
        self._active[device] += 1

        try:
            env = await self._acquire_env(device, env_config, seed)
        except BaseException:
            # Roll back on failure
            self._active[device] = max(0, self._active[device] - 1)
            raise

        # Standard session setup
        import uuid
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = SessionContext(
            session_id=session_id,
            env=env,
            created_at=time.time(),
            last_access=time.time(),
        )

        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        LOGGER.info(f"[NavHandler] Session {session_id[:8]} created ({self._stats_str()})")

        result_data = {"session_id": session_id}
        if seed is not None:
            obs, info = await self._sessions[session_id].env.reset(seed)
            result_data["obs"] = obs.get("obs_str", "")
            result_data["info"] = info
            images = self._extract_images(obs)
            return HandlerResult(data=result_data, images=images)

        return HandlerResult(data=result_data)

    # ------------------------------------------------------------------
    # Close (cache instead of destroy)
    # ------------------------------------------------------------------

    async def _handle_close(self, ctx: SessionContext) -> HandlerResult:
        env: NavigationEnv = ctx.env
        device = env.cfg.gpu_device
        scene = env._episode_data["scene"] if env._episode_data else ""

        # Move from active to cached (env stays alive, no slot change)
        self._active[device] = max(0, self._active[device] - 1)
        self._cache[device].append((scene, env))

        del self._sessions[ctx.session_id]
        LOGGER.info(
            f"[NavHandler] Session {ctx.session_id[:8]} -> cached "
            f"scene={scene} GPU {device} ({self._stats_str()})"
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

    # ------------------------------------------------------------------
    # Cleanup: override base class to release semaphore + active count
    # ------------------------------------------------------------------

    async def _cleanup_loop(self):
        """Background task to cleanup timed-out sessions.

        Overrides base class to properly release semaphore slots and
        decrement _active counts when sessions are cleaned up.
        """
        while True:
            try:
                await asyncio.sleep(60)

                now = time.time()
                to_remove = []

                for session_id, ctx in self._sessions.items():
                    idle_time = now - ctx.last_access
                    if idle_time > self.session_timeout:
                        to_remove.append(session_id)
                        LOGGER.warning(
                            f"[NavHandler] Session {session_id[:8]} timed out "
                            f"after {idle_time:.1f}s idle"
                        )

                for session_id in to_remove:
                    ctx = self._sessions.get(session_id)
                    if ctx is None:
                        continue
                    try:
                        env: NavigationEnv = ctx.env
                        device = env.cfg.gpu_device
                        await env.close()
                        self._sessions.pop(session_id, None)
                        self._active[device] = max(0, self._active[device] - 1)
                        self._env_slots.release()
                        LOGGER.info(
                            f"[NavHandler] Cleaned up timed-out session "
                            f"{session_id[:8]} GPU {device} ({self._stats_str()})"
                        )
                    except Exception as e:
                        LOGGER.error(
                            f"[NavHandler] Error cleaning up session "
                            f"{session_id[:8]}: {e}"
                        )
                        self._sessions.pop(session_id, None)
                        try:
                            device = ctx.env.cfg.gpu_device
                            self._active[device] = max(0, self._active[device] - 1)
                        except Exception:
                            pass
                        self._env_slots.release()

            except asyncio.CancelledError:
                break
            except Exception as e:
                LOGGER.error(f"[NavHandler] Cleanup loop error: {e}")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def aclose(self):
        for device, entries in self._cache.items():
            for _, env in entries:
                try:
                    await env.close()
                except Exception as e:
                    LOGGER.error(f"[NavHandler] Error closing cached env GPU {device}: {e}")
            entries.clear()
        LOGGER.info("[NavHandler] All cached envs closed")
        await super().aclose()
