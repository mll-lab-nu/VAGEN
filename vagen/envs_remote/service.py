"""
Generic FastAPI service for gym environments.

This service is completely reusable and environment-agnostic.
It only handles:
- HTTP routing
- Session ID validation
- Request/response encoding/decoding
- Forwarding to handler

All business logic is delegated to the handler.

Customization:
    Subclass GymService and override individual methods to customize
    authentication, admission control, error handling, or add endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response

from .handler import BaseGymHandler, SessionNotFoundError
from .multipart_codec import encode_multipart, decode_multipart

LOGGER = logging.getLogger(__name__)


class GymService:
    """
    Generic HTTP service for remote gym environments.

    Wraps a BaseGymHandler with FastAPI routing, authentication,
    and concurrency control.

    Subclass to customize:
    - authenticate(): change auth mechanism
    - acquire() / release(): change admission / concurrency control
    - handle_connect_error() / handle_call_error(): change error mapping
    - register_routes(): add or replace endpoints

    Usage:
        # Basic
        app = GymService(MyHandler()).build()
        # uvicorn mymodule:app --host 0.0.0.0 --port 8000

        # With options
        app = GymService(MyHandler(), max_inflight=10, api_key="secret").build()

        # Subclass for custom behavior
        class MyService(GymService):
            def authenticate(self, request): ...
            def register_routes(self, app):
                super().register_routes(app)
                app.add_api_route("/gpu-status", self.gpu_status, methods=["GET"])
        app = MyService(MyHandler()).build()
    """

    def __init__(
        self,
        handler: BaseGymHandler,
        max_inflight: int = 0,
        admit_timeout: float = 5.0,
        api_key: str = "",
        image_format: str = "PNG",
        image_mime: str = "image/png",
    ):
        self.handler = handler
        self.max_inflight = max_inflight
        self.admit_timeout = admit_timeout
        self.api_key = api_key or os.getenv("GYM_API_KEY", "")
        self.image_format = image_format
        self.image_mime = image_mime
        self._sem: Optional[asyncio.Semaphore] = (
            asyncio.Semaphore(max_inflight) if max_inflight > 0 else None
        )

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def authenticate(self, request: Request) -> None:
        """
        Authenticate a request. Override for custom auth.

        Raises HTTPException(401) on failure.
        Default: checks GYM_API_KEY env var or api_key constructor arg.
        """
        if not self.api_key:
            return
        token = request.query_params.get("token") or request.headers.get("x-api-key")
        if token != self.api_key:
            raise HTTPException(status_code=401, detail="unauthorized")

    async def acquire(self) -> bool:
        """
        Acquire admission slot. Returns True if a slot was acquired
        (and must be released later), False if no semaphore is configured.

        Override to implement custom admission control (e.g., GPU capacity queues).
        """
        if self._sem is None:
            return False
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=self.admit_timeout)
            return True
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="server busy")

    def release(self) -> None:
        """Release admission slot. Override if acquire() is overridden."""
        if self._sem is not None:
            self._sem.release()

    def handle_connect_error(self, e: Exception) -> None:
        """Map connect exceptions to HTTP errors. Override to customize."""
        if isinstance(e, RuntimeError) and "Max sessions limit reached" in str(e):
            LOGGER.warning(f"[Service] Connect rejected: {e}")
            raise HTTPException(status_code=503, detail=str(e))
        LOGGER.error(f"[Service] Connect error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    def handle_call_error(self, e: Exception) -> None:
        """Map call exceptions to HTTP errors. Override to customize."""
        if isinstance(e, SessionNotFoundError):
            raise HTTPException(status_code=404, detail=str(e))
        if isinstance(e, RuntimeError) and "Max sessions limit reached" in str(e):
            raise HTTPException(status_code=503, detail=str(e))
        LOGGER.error(f"[Service] Call error: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    def _encode_result(self, result: Any) -> Response:
        """Encode a HandlerResult into an HTTP response."""
        boundary, body = encode_multipart(
            result.data,
            result.images,
            image_format=self.image_format,
            image_mime=self.image_mime,
        )
        return Response(
            content=body,
            media_type=f'multipart/mixed; boundary="{boundary}"',
        )

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    async def health(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "ok": True,
            "service": "gym-env-service",
            "max_inflight": self.max_inflight if self.max_inflight > 0 else "unlimited",
        }

    async def sessions(self, request: Request) -> Dict[str, Any]:
        """Get statistics about active sessions."""
        self.authenticate(request)
        return self.handler.get_session_stats()

    async def connect(self, request: Request) -> Response:
        """Create a new session, optionally with initial reset."""
        self.authenticate(request)

        acquired = await self.acquire()
        try:
            content_type = request.headers.get("content-type", "")
            body = await request.body()
            data_dict, _ = decode_multipart(content_type, body)

            env_config = data_dict.get("env_config", {})
            seed = data_dict.get("seed")

            result = await self.handler.connect(env_config, seed=seed)
            return self._encode_result(result)

        except HTTPException:
            raise
        except Exception as e:
            self.handle_connect_error(e)
        finally:
            if acquired:
                self.release()

    async def call(self, request: Request) -> Response:
        """Call a method on an existing session."""
        self.authenticate(request)

        acquired = await self.acquire()
        try:
            content_type = request.headers.get("content-type", "")
            body = await request.body()
            data_dict, img_list = decode_multipart(content_type, body)

            session_id = data_dict.get("session_id")
            method = data_dict.get("method")
            params = data_dict.get("params", {})

            if not session_id:
                raise HTTPException(status_code=400, detail="session_id required")
            if not method:
                raise HTTPException(status_code=400, detail="method required")

            result = await self.handler.call(session_id, method, params, img_list)
            return self._encode_result(result)

        except HTTPException:
            raise
        except Exception as e:
            self.handle_call_error(e)
        finally:
            if acquired:
                self.release()

    # ------------------------------------------------------------------
    # App construction
    # ------------------------------------------------------------------

    def register_routes(self, app: FastAPI) -> None:
        """
        Register routes on the app. Override to add or replace endpoints.

        Call super().register_routes(app) to keep the default routes,
        then add your own.
        """
        app.add_api_route("/health", self.health, methods=["GET"])
        app.add_api_route("/sessions", self.sessions, methods=["GET"])
        app.add_api_route("/connect", self.connect, methods=["POST"])
        app.add_api_route("/call", self.call, methods=["POST"])

    def build(
        self,
        startup_callback: Optional[callable] = None,
        shutdown_callback: Optional[callable] = None,
    ) -> FastAPI:
        """
        Build and return the FastAPI application.

        This is the main entry point. Call once, then run the returned app
        with uvicorn.

        Args:
            startup_callback: Optional sync callable invoked during app startup
                (inside the running event loop, before requests are served).
            shutdown_callback: Optional sync callable invoked during app shutdown.
        """
        handler = self.handler

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            if startup_callback is not None:
                startup_callback()
            try:
                yield
            finally:
                if shutdown_callback is not None:
                    shutdown_callback()
                await handler.aclose()

        app = FastAPI(
            title="Gym Environment Service",
            description="Generic HTTP service for remote gym environments",
            lifespan=lifespan,
        )
        app.state.handler = handler

        self.register_routes(app)
        return app


