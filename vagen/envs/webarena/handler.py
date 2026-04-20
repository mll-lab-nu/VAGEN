"""Handler for the VAGEN WebArena environment.

Owns a Chromium browser pool (K browsers × M contexts each). One session
== one Playwright context, drawn from the pool.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from vagen.envs_remote.handler import BaseGymHandler, HandlerResult, SessionContext
from .browser_pool import BrowserPool, BrowserSlot
from .webarena_env import WebArenaEnv, load_tasks

LOGGER = logging.getLogger(__name__)


class WebArenaHandler(BaseGymHandler):
    """
    Resource model:
      K Chromium browsers, each backed by its own OS thread (Playwright
      sync API forces thread affinity). Each browser can host up to
      max_contexts_per_browser Playwright contexts concurrently. A session
      borrows one context from a (least-loaded) browser.

    Total capacity: K × M sessions.
    """

    def __init__(
        self,
        task_config_file: str,
        auth_cache_dir: str,
        n_browsers: int = 4,
        max_contexts_per_browser: int = 16,
        session_timeout: float = 3600.0,
        acquire_timeout: float = 300.0,
    ):
        # max_sessions=0 disables the base class session cap; capacity is
        # actually bounded by the browser pool's semaphore.
        super().__init__(session_timeout=session_timeout, max_sessions=0)

        if not task_config_file or not os.path.exists(task_config_file):
            raise FileNotFoundError(f"task_config_file not found: {task_config_file}")

        self.task_config_file = task_config_file
        self.auth_cache_dir = os.path.abspath(auth_cache_dir)
        self.acquire_timeout = acquire_timeout

        self.pool = BrowserPool(
            n_browsers=n_browsers,
            max_contexts_per_browser=max_contexts_per_browser,
        )
        self._tasks: List[Dict[str, Any]] = load_tasks(task_config_file)
        self._auth_locks: Dict[str, asyncio.Lock] = {}
        self._started = False
        self._start_lock = asyncio.Lock()
        LOGGER.info(
            f"[WebArenaHandler] init: {len(self._tasks)} tasks, "
            f"auth_cache={self.auth_cache_dir}, "
            f"pool={n_browsers}×{max_contexts_per_browser}={n_browsers * max_contexts_per_browser}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            os.makedirs(self.auth_cache_dir, exist_ok=True)
            await self.pool.start()
            self._started = True

    async def aclose(self) -> None:
        await super().aclose()
        await self.pool.close()

    # ------------------------------------------------------------------
    # create_env (unused directly — connect() overrides to inject slot)
    # ------------------------------------------------------------------

    async def create_env(self, env_config: Dict[str, Any]) -> Any:
        raise NotImplementedError("Use connect() instead.")

    # ------------------------------------------------------------------
    # connect: override to pass BrowserSlot into the env
    # ------------------------------------------------------------------

    async def connect(
        self, env_config: Dict[str, Any], seed: Optional[int] = None
    ) -> HandlerResult:
        if not self._started:
            await self.start()

        # Inject shared defaults into env_config
        env_config = {
            **env_config,
            "task_config_file": self.task_config_file,
            "auth_cache_dir": self.auth_cache_dir,
        }

        slot = await self.pool.acquire_slot(timeout=self.acquire_timeout)
        try:
            env = WebArenaEnv(
                env_config=env_config,
                browser_slot=slot,
                browser_pool=self.pool,
                auth_locks=self._auth_locks,
                tasks=self._tasks,
            )
        except BaseException:
            self.pool.release_slot(slot)
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

        LOGGER.info(
            f"[WebArenaHandler] session {session_id[:8]} created on browser {slot.browser_id} "
            f"({self.pool.stats_str()})"
        )

        result_data: Dict[str, Any] = {"session_id": session_id}
        if seed is not None:
            try:
                obs, info = await env.reset(seed)
            except BaseException:
                # Reset failed — drop the session, free the slot
                self._sessions.pop(session_id, None)
                await env.close()
                raise
            result_data["obs"] = obs.get("obs_str", "")
            result_data["info"] = info
            images = self._extract_images(obs)
            return HandlerResult(data=result_data, images=images)

        return HandlerResult(data=result_data)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_session_stats(self) -> Dict[str, Any]:
        stats = super().get_session_stats()
        stats["browser_pool"] = self.pool.stats()
        return stats
