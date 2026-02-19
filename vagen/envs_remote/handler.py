"""
Handler interface for gym environment service.

The handler is the ONLY component that needs to be customized for different environments.
Client and server are completely reusable.

Handler responsibilities:
- Manage session_id -> environment instance mapping
- Implement environment creation/destruction logic
- Handle method calls (system_prompt, reset, step)
- Manage resource cleanup and timeout logic
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass
class HandlerResult:
    """
    Standard result format returned by handler methods.

    Attributes:
        data: Result data dict (always present)
        images: Optional list of PIL images
    """

    data: Dict[str, Any]
    images: Optional[List[Image.Image]] = None


class BaseGymHandler(ABC):
    """
    Abstract base class for gym environment handlers.

    Subclass this to implement custom environment logic.
    The handler manages all business logic including:
    - Session lifecycle (create, destroy)
    - Environment instance management
    - Method dispatch (system_prompt, reset, step)
    - Resource cleanup and timeout handling
    """

    def __init__(
        self,
        session_timeout: float = 3600.0,
        max_sessions: int = 0,
    ):
        """
        Initialize handler.

        Args:
            session_timeout: Maximum idle time before session cleanup (seconds)
            max_sessions: Maximum number of concurrent sessions (0 = unlimited)
        """
        self.session_timeout = session_timeout
        self.max_sessions = max_sessions
        self._sessions: Dict[str, SessionContext] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

    @abstractmethod
    async def create_env(self, env_config: Dict[str, Any]) -> Any:
        """
        Create a new environment instance.

        Args:
            env_config: Environment configuration dict

        Returns:
            Environment instance (typically a GymImageEnv subclass)

        Note:
            This is the main method to customize for different environments.
        """
        raise NotImplementedError

    async def connect(
        self, env_config: Dict[str, Any], seed: Optional[int] = None
    ) -> HandlerResult:
        """
        Handle connect request: create new session.

        If seed is provided, also perform initial reset and return result.
        This reduces round-trips for the common case of connect + reset.

        Args:
            env_config: Environment configuration
            seed: Optional seed for initial reset

        Returns:
            HandlerResult with session_id
            If seed provided: also includes obs, info, and images from reset

        Raises:
            RuntimeError: If max_sessions limit reached
        """
        # Check session limit
        if self.max_sessions > 0 and len(self._sessions) >= self.max_sessions:
            raise RuntimeError(
                f"Max sessions limit reached ({self.max_sessions}). "
                f"Please try again later or close existing sessions."
            )

        session_id = uuid.uuid4().hex
        env = await self.create_env(env_config)

        self._sessions[session_id] = SessionContext(
            session_id=session_id,
            env=env,
            created_at=time.time(),
            last_access=time.time(),
        )

        LOGGER.info(
            f"[Handler] Created session {session_id} "
            f"({len(self._sessions)}/{self.max_sessions if self.max_sessions > 0 else 'unlimited'})"
            + (f" with initial seed {seed}" if seed is not None else "")
        )

        # Start cleanup task if not running
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        result_data = {"session_id": session_id}

        # If seed provided, perform initial reset
        if seed is not None:
            ctx = self._sessions[session_id]
            obs, info = await ctx.env.reset(seed)

            result_data["obs"] = obs.get("obs_str", "")
            result_data["info"] = info

            images = self._extract_images(obs)
            return HandlerResult(data=result_data, images=images)

        return HandlerResult(data=result_data)

    async def call(
        self,
        session_id: str,
        method: str,
        params: Dict[str, Any],
        images: List[Image.Image],
    ) -> HandlerResult:
        """
        Handle call request: execute method on session's environment.

        Args:
            session_id: Session ID
            method: Method name (system_prompt, reset, step, close)
            params: Method parameters
            images: Input images (if any)

        Returns:
            HandlerResult with method results

        Raises:
            ValueError: If session not found or method invalid
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")

        ctx = self._sessions[session_id]
        ctx.last_access = time.time()

        # Dispatch to appropriate method
        if method == "system_prompt":
            result = await self._handle_system_prompt(ctx)
        elif method == "reset":
            result = await self._handle_reset(ctx, params)
        elif method == "step":
            result = await self._handle_step(ctx, params)
        elif method == "close":
            result = await self._handle_close(ctx)
        else:
            raise ValueError(f"Unknown method: {method}")

        return result

    async def _handle_system_prompt(self, ctx: SessionContext) -> HandlerResult:
        """Handle system_prompt call."""
        obs = await ctx.env.system_prompt()
        return self._obs_to_result(obs)

    async def _handle_reset(self, ctx: SessionContext, params: Dict[str, Any]) -> HandlerResult:
        """Handle reset call."""
        seed = params.get("seed", 0)
        obs, info = await ctx.env.reset(seed)

        result_data = {
            "obs": obs.get("obs_str", ""),
            "info": info,
        }

        images = self._extract_images(obs)
        return HandlerResult(data=result_data, images=images)

    async def _handle_step(self, ctx: SessionContext, params: Dict[str, Any]) -> HandlerResult:
        """Handle step call."""
        action_str = params.get("action_str", "")
        obs, reward, done, info = await ctx.env.step(action_str)

        result_data = {
            "obs": obs.get("obs_str", ""),
            "reward": reward,
            "done": done,
            "info": info,
        }

        images = self._extract_images(obs)
        return HandlerResult(data=result_data, images=images)

    async def _handle_close(self, ctx: SessionContext) -> HandlerResult:
        """Handle close call and cleanup session."""
        await ctx.env.close()
        del self._sessions[ctx.session_id]
        LOGGER.info(
            f"[Handler] Closed session {ctx.session_id} "
            f"({len(self._sessions)}/{self.max_sessions if self.max_sessions > 0 else 'unlimited'} remaining)"
        )
        return HandlerResult(data={"closed": True})

    @staticmethod
    def _obs_to_result(obs: Dict[str, Any]) -> HandlerResult:
        """Convert observation dict to HandlerResult."""
        result_data = {
            "obs": obs.get("obs_str", ""),
        }
        images = BaseGymHandler._extract_images(obs)
        return HandlerResult(data=result_data, images=images)

    @staticmethod
    def _extract_images(obs: Dict[str, Any]) -> Optional[List[Image.Image]]:
        """Extract images from observation multi_modal_input."""
        if "multi_modal_input" not in obs:
            return None
        mmi = obs["multi_modal_input"]
        if "<image>" in mmi:
            return mmi["<image>"]
        return None

    async def _cleanup_loop(self):
        """Background task to cleanup timed-out sessions."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = time.time()
                to_remove = []

                for session_id, ctx in self._sessions.items():
                    idle_time = now - ctx.last_access
                    if idle_time > self.session_timeout:
                        to_remove.append(session_id)
                        LOGGER.warning(
                            f"[Handler] Session {session_id} timed out after {idle_time:.1f}s idle"
                        )

                # Cleanup timed-out sessions
                for session_id in to_remove:
                    ctx = self._sessions.get(session_id)
                    if ctx is None:
                        # Session may have been removed concurrently
                        continue
                    try:
                        await ctx.env.close()
                        # Use pop with default to avoid KeyError if concurrently removed
                        self._sessions.pop(session_id, None)
                    except Exception as e:
                        LOGGER.error(f"[Handler] Error cleaning up session {session_id}: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                LOGGER.error(f"[Handler] Cleanup loop error: {e}")

    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current sessions.

        Returns:
            Dict with session statistics:
            - num_sessions: Current number of active sessions
            - max_sessions: Maximum allowed sessions (0 = unlimited)
            - sessions: List of session info dicts
        """
        now = time.time()
        sessions_info = []

        for session_id, ctx in self._sessions.items():
            idle_time = now - ctx.last_access
            sessions_info.append({
                "session_id": session_id,
                "created_at": ctx.created_at,
                "last_access": ctx.last_access,
                "idle_seconds": idle_time,
                "will_timeout_in": max(0, self.session_timeout - idle_time),
            })

        return {
            "num_sessions": len(self._sessions),
            "max_sessions": self.max_sessions,
            "session_timeout": self.session_timeout,
            "sessions": sessions_info,
        }

    async def aclose(self):
        """Cleanup all sessions on shutdown."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all remaining sessions
        for session_id, ctx in list(self._sessions.items()):
            try:
                await ctx.env.close()
            except Exception as e:
                LOGGER.error(f"[Handler] Error closing session {session_id}: {e}")

        self._sessions.clear()
        LOGGER.info("[Handler] All sessions closed")


@dataclass
class SessionContext:
    """Context for a single session."""

    session_id: str
    env: Any  # Environment instance
    created_at: float
    last_access: float
