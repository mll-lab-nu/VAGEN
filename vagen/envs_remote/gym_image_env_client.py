"""
Generic HTTP client for remote gym environments.

This client is completely reusable and environment-agnostic.
It inherits from GymImageEnv and transparently forwards all method calls to a remote server.

Features:
- URL pool support with failover
- Automatic retry with exponential backoff
- Session management (connect on reset, cleanup on close)
- Full GymImageEnv interface compatibility

Usage:
    env = GymImageEnvClient(env_config={
        "base_urls": ["http://localhost:8000", "http://localhost:8001"],
        "timeout": 120.0,
        "retries": 8,
        "backoff": 2.0,
        # ... other config passed to remote environment
    })

    # Connection is established on first reset()
    obs, info = await env.reset(seed=42)

    # Session is locked to one server with one session_id
    obs, reward, done, info = await env.step("action")

    await env.close()
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import httpx
from PIL import Image

from vagen.envs.gym_image_env import GymImageEnv
from .multipart_codec import encode_multipart, decode_multipart

LOGGER = logging.getLogger(__name__)


class GymImageEnvClient(GymImageEnv):
    """
    Generic HTTP client for remote gym environments.

    Connection lifecycle:
    - __init__(): Create client (synchronous, no connection yet)
    - reset(): Establish connection and get session_id (first time only)
    - step(): Use established session
    - close(): Cleanup session and disconnect

    Design rationale:
    - One env instance = one session = one episode sequence (reset -> step* -> close)
    - Session_id is determined on first reset() and locked thereafter
    - This matches standard gym usage pattern and keeps __init__ synchronous
    """

    def __init__(self, env_config: Dict[str, Any]):
        """
        Initialize remote gym client (does not connect to server yet).

        Connection is established on the first reset() call.

        Args:
            env_config: Configuration dict with keys:
                - base_urls (str | List[str]): Server URL(s)
                - timeout (float): Request timeout in seconds (default: 120.0)
                - retries (int): Number of retries (default: 8)
                - backoff (float): Backoff multiplier (default: 2.0)
                - token (str, optional): API key for authentication
                - log_retries (bool): Whether to log retries (default: True)
                - failover_after_failures (int): Failover to next URL after N failures (default: 4)
                - ... (other keys passed to remote environment)
        """
        super().__init__(env_config)

        # Extract client config
        base_urls = env_config.get("base_urls", "http://localhost:8000")
        if isinstance(base_urls, str):
            base_urls = [base_urls]
        self.base_urls: List[str] = base_urls

        self.timeout = float(env_config.get("timeout", 120.0))
        self.retries = int(env_config.get("retries", 8))
        self.backoff = float(env_config.get("backoff", 2.0))
        self.backoff_jitter_min = float(env_config.get("backoff_jitter_min", 0.7))
        self.backoff_jitter_range = float(env_config.get("backoff_jitter_range", 0.6))
        self.token = env_config.get("token")
        self.log_retries = bool(env_config.get("log_retries", True))
        self.failover_after_failures = int(env_config.get("failover_after_failures", 4))

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None
        self._session_id: Optional[str] = None
        self._current_url_index: int = 0

        # Remove client-specific keys before passing to remote
        self._remote_env_config = {
            k: v
            for k, v in env_config.items()
            if k
            not in {
                "base_urls",
                "timeout",
                "retries",
                "backoff",
                "backoff_jitter_min",
                "backoff_jitter_range",
                "token",
                "log_retries",
                "failover_after_failures",
            }
        }

    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )

    async def _ensure_connected_for_reset(self, seed: int) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Ensure session is established (called by reset only).

        On first reset: Establishes connection, passes seed, and gets reset result.
        On subsequent resets: Returns None (caller should call reset normally).

        Args:
            seed: Reset seed (passed to server on first reset only)

        Returns:
            (obs, info) if this is first reset and server returns reset result
            None otherwise
        """
        if self._session_id is None:
            return await self._connect(seed=seed)
        return None

    def _check_connected(self, method_name: str):
        """
        Check if connected (called by methods other than reset).

        Raises:
            RuntimeError: If not connected (forgot to call reset first)
        """
        if self._session_id is None:
            raise RuntimeError(
                f"Cannot call {method_name}() before reset(). "
                f"The session is established on the first reset() call."
            )

    async def _connect(
        self, seed: Optional[int] = None
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Establish connection with server and obtain session_id.

        If seed is provided, the server will also perform reset and return
        the initial observation and info.

        Args:
            seed: Optional seed for initial reset

        Returns:
            (obs, info) if seed was provided and server returns reset result
            None otherwise
        """
        await self._ensure_client()

        request_data = {"env_config": self._remote_env_config}
        if seed is not None:
            request_data["seed"] = seed

        boundary, body = encode_multipart(request_data)

        headers = {"Content-Type": f'multipart/form-data; boundary="{boundary}"'}
        if self.token:
            headers["X-API-Key"] = self.token

        last_exc: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            url_index = self._pick_url_index(attempt)
            base_url = self.base_urls[url_index]
            url = f"{base_url}/connect"

            try:
                response = await self._client.post(url, content=body, headers=headers)

                if response.status_code == 503:
                    raise RuntimeError("server busy")

                response.raise_for_status()

                # Decode response
                content_type = response.headers.get("content-type", "")
                data, images = decode_multipart(content_type, response.content)

                self._session_id = data.get("session_id")
                if not self._session_id:
                    raise RuntimeError("Server did not return session_id")

                self._current_url_index = url_index
                LOGGER.info(
                    f"[Client] Connected to {base_url}, session_id={self._session_id}"
                    + (f", seed={seed}" if seed is not None else "")
                )

                # If seed was provided, return reset result
                if seed is not None and "obs" in data:
                    obs = {"obs_str": data.get("obs", "")}
                    if images:
                        obs["multi_modal_input"] = {"<image>": images}
                    info = data.get("info", {})
                    return obs, info

                return None

            except Exception as e:
                last_exc = e

                if attempt == self.retries:
                    LOGGER.error(
                        f"[Client] Connect failed after {self.retries} retries to {base_url}: {e}"
                    )
                    raise RuntimeError(f"Failed to connect to any server: {e}") from e

                jitter = self.backoff_jitter_min + self.backoff_jitter_range * random.random()
                delay = self.backoff * (2**attempt) * jitter

                if self.log_retries:
                    next_url_index = self._pick_url_index(attempt + 1)
                    next_url = self.base_urls[next_url_index]
                    LOGGER.warning(
                        f"[Client] Connect retry (current={base_url}, next={next_url}, "
                        f"attempt={attempt + 1}/{self.retries}, delay={delay:.2f}s, error={type(e).__name__})"
                    )

                await asyncio.sleep(delay)

        raise RuntimeError(f"Failed to connect: {last_exc}")

    def _pick_url_index(self, attempt: int) -> int:
        """
        Pick URL index based on attempt number.

        attempts 0..F -> url[0]
        attempt F+1 -> url[1]
        ...
        """
        if len(self.base_urls) == 1:
            return 0

        if attempt <= self.failover_after_failures:
            return self._current_url_index

        offset = attempt - self.failover_after_failures
        return offset % len(self.base_urls)

    async def _call(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        images: Optional[List[Image.Image]] = None,
    ) -> Tuple[Dict[str, Any], Optional[List[Image.Image]]]:
        """
        Generic method to call remote endpoint.

        Args:
            method: Method name (system_prompt, reset, step, close)
            params: Method parameters
            images: Input images

        Returns:
            (result_data, result_images)

        Note:
            Assumes session_id is already established. Caller should ensure this.
        """
        # Note: We assume _session_id is set. reset() calls _ensure_connected_for_reset()
        # before calling this. Other methods check via _check_connected().

        request_data = {
            "session_id": self._session_id,
            "method": method,
            "params": params or {},
        }

        boundary, body = encode_multipart(request_data, images)
        headers = {"Content-Type": f'multipart/form-data; boundary="{boundary}"'}
        if self.token:
            headers["X-API-Key"] = self.token

        last_exc: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            url_index = self._pick_url_index(attempt)
            base_url = self.base_urls[url_index]
            url = f"{base_url}/call"

            try:
                response = await self._client.post(url, content=body, headers=headers)

                if response.status_code == 503:
                    raise RuntimeError("server busy")

                response.raise_for_status()

                # Decode response
                content_type = response.headers.get("content-type", "")
                data, result_images = decode_multipart(content_type, response.content)

                # Update current URL on success
                if url_index != self._current_url_index:
                    LOGGER.info(f"[Client] Switched to {base_url}")
                    self._current_url_index = url_index

                return data, result_images if result_images else None

            except Exception as e:
                last_exc = e

                if attempt == self.retries:
                    LOGGER.error(
                        f"[Client] Call {method} failed after {self.retries} retries to {base_url}: {e}"
                    )
                    raise RuntimeError(f"Remote call failed: {e}") from e

                jitter = self.backoff_jitter_min + self.backoff_jitter_range * random.random()
                delay = self.backoff * (2**attempt) * jitter

                if self.log_retries:
                    next_url_index = self._pick_url_index(attempt + 1)
                    next_url = self.base_urls[next_url_index]
                    LOGGER.warning(
                        f"[Client] Call {method} retry (current={base_url}, next={next_url}, "
                        f"attempt={attempt + 1}/{self.retries}, delay={delay:.2f}s, error={type(e).__name__})"
                    )

                await asyncio.sleep(delay)

        raise RuntimeError(f"Remote call failed: {last_exc}")

    # =========================================================================
    # GymImageEnv interface implementation (transparent forwarding)
    # =========================================================================

    async def system_prompt(self) -> Dict[str, Any]:
        """
        Get system prompt from remote environment.

        Note: Should be called after reset() in typical usage.
        In gym_agent_loop, reset() is always called first, which establishes
        the connection.
        """
        # Check if connected (reset should be called first)
        self._check_connected("system_prompt")

        data, images = await self._call("system_prompt")

        obs = {"obs_str": data.get("obs", "")}
        if images:
            obs["multi_modal_input"] = {"<image>": images}

        return obs

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset remote environment.

        On first call: Establishes connection with server and obtains session_id.
        On subsequent calls: Uses existing session_id.

        This is where the session is created and locked to a specific server URL.
        """
        # Establish connection on first reset
        await self._ensure_connected_for_reset()

        data, images = await self._call("reset", params={"seed": seed})

        obs = {"obs_str": data.get("obs", "")}
        if images:
            obs["multi_modal_input"] = {"<image>": images}

        info = data.get("info", {})

        return obs, info

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute step on remote environment.

        Requires: reset() must be called first to establish session.
        """
        self._check_connected("step")

        data, images = await self._call("step", params={"action_str": action_str})

        obs = {"obs_str": data.get("obs", "")}
        if images:
            obs["multi_modal_input"] = {"<image>": images}

        reward = float(data.get("reward", 0.0))
        done = bool(data.get("done", False))
        info = data.get("info", {})

        return obs, reward, done, info

    async def close(self) -> None:
        """Close remote environment and cleanup session."""
        if self._session_id is not None:
            try:
                await self._call("close")
                LOGGER.info(f"[Client] Closed session {self._session_id}")
            except Exception as e:
                LOGGER.warning(f"[Client] Error during close: {e}")
            finally:
                self._session_id = None

        if self._client is not None:
            await self._client.aclose()
            self._client = None
