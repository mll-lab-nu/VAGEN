"""
Generic FastAPI service for gym environments.

This service is completely reusable and environment-agnostic.
It only handles:
- HTTP routing
- Session ID validation
- Request/response encoding/decoding
- Forwarding to handler

All business logic is delegated to the handler.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException
from fastapi.responses import Response

from .handler import BaseGymHandler
from .multipart_codec import encode_multipart, parse_data_field, read_images, decode_multipart

LOGGER = logging.getLogger(__name__)

# ----------------------------
# Config (env-driven)
# ----------------------------
API_KEY = os.getenv("GYM_API_KEY", "")  # Empty => no auth
MAX_INFLIGHT = int(os.getenv("GYM_MAX_INFLIGHT", "0"))  # 0 => unlimited
ADMIT_TIMEOUT = float(os.getenv("GYM_ADMIT_TIMEOUT", "5.0"))

IMAGE_FORMAT = os.getenv("GYM_IMAGE_FORMAT", "PNG")
IMAGE_MIME = os.getenv("GYM_IMAGE_MIME", "image/png")

# Global concurrency limiter (optional)
_sem = asyncio.Semaphore(MAX_INFLIGHT) if MAX_INFLIGHT > 0 else None


def _auth(request: Request) -> None:
    """
    Optional API key authentication.

    Accepts:
      - Query param: ?token=...
      - Header: X-API-Key: ...
    """
    if not API_KEY:
        return
    token = request.query_params.get("token") or request.headers.get("x-api-key")
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")


def build_gym_service(handler: BaseGymHandler) -> FastAPI:
    """
    Build a generic gym environment service.

    This function creates a FastAPI app that routes requests to the provided handler.
    The service is completely reusable - just provide a different handler for
    different environments.

    Args:
        handler: Handler instance that implements environment logic

    Returns:
        FastAPI application ready to serve

    Usage:
        >>> handler = MyGymHandler()
        >>> app = build_gym_service(handler)
        >>> # Run with: uvicorn mymodule:app --host 0.0.0.0 --port 8000
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await handler.aclose()

    app = FastAPI(
        title="Gym Environment Service",
        description="Generic HTTP service for remote gym environments",
        lifespan=lifespan,
    )
    app.state.handler = handler

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "ok": True,
            "service": "gym-env-service",
            "max_inflight": MAX_INFLIGHT if MAX_INFLIGHT > 0 else "unlimited",
        }

    @app.post("/connect")
    async def connect(
        request: Request,
        data: Optional[str] = Form(default=None),
        images: Optional[List[UploadFile]] = File(default=None),
    ):
        """
        Create a new session.

        Request:
            Content-Type: multipart/form-data
            - data: JSON string with env_config and optional seed
                    {"env_config": {...}, "seed": 42}  (seed is optional)

        Response:
            Content-Type: multipart/mixed
            - data: {"session_id": "..."}
                    If seed provided: also {"obs": "...", "info": {...}}
            - images: (if seed provided and env returns images)

        If seed is provided, the server will create the session AND perform
        initial reset in one round-trip, returning both session_id and reset result.
        """
        _auth(request)

        acquired = False
        if _sem is not None:
            try:
                await asyncio.wait_for(_sem.acquire(), timeout=ADMIT_TIMEOUT)
                acquired = True
            except asyncio.TimeoutError:
                raise HTTPException(status_code=503, detail="server busy")

        try:
            data_dict = parse_data_field(data)
            env_config = data_dict.get("env_config", {})
            seed = data_dict.get("seed")  # Optional

            result = await handler.connect(env_config, seed=seed)

            boundary, body = encode_multipart(
                result.data,
                result.images,
                image_format=IMAGE_FORMAT,
                image_mime=IMAGE_MIME,
            )

            return Response(
                content=body,
                media_type=f'multipart/mixed; boundary="{boundary}"',
            )

        except Exception as e:
            LOGGER.error(f"[Service] Connect error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if acquired and _sem is not None:
                _sem.release()

    @app.post("/call")
    async def call(
        request: Request,
        data: Optional[str] = Form(default=None),
        images: Optional[List[UploadFile]] = File(default=None),
    ):
        """
        Call a method on an existing session.

        Request:
            Content-Type: multipart/form-data
            - data: JSON string with {session_id, method, params}
            - images: optional images

        Response:
            Content-Type: multipart/mixed
            - data: method result
            - images: optional result images
        """
        _auth(request)

        acquired = False
        if _sem is not None:
            try:
                await asyncio.wait_for(_sem.acquire(), timeout=ADMIT_TIMEOUT)
                acquired = True
            except asyncio.TimeoutError:
                raise HTTPException(status_code=503, detail="server busy")

        try:
            data_dict = parse_data_field(data)
            img_list = await read_images(images)

            session_id = data_dict.get("session_id")
            method = data_dict.get("method")
            params = data_dict.get("params", {})

            if not session_id:
                raise HTTPException(status_code=400, detail="session_id required")
            if not method:
                raise HTTPException(status_code=400, detail="method required")

            result = await handler.call(session_id, method, params, img_list)

            boundary, body = encode_multipart(
                result.data,
                result.images,
                image_format=IMAGE_FORMAT,
                image_mime=IMAGE_MIME,
            )

            return Response(
                content=body,
                media_type=f'multipart/mixed; boundary="{boundary}"',
            )

        except ValueError as e:
            # Handler raised ValueError (e.g., session not found)
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            LOGGER.error(f"[Service] Call error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if acquired and _sem is not None:
                _sem.release()

    return app
