"""
ManiSkill environment server.

Usage:
    python -m vagen.envs.maniskill.serve
    python -m vagen.envs.maniskill.serve --port=8003 --max_sessions=128
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging

import fire
import uvicorn

from vagen.envs_remote import GymService
from .handler import ManiSkillHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


def main(
    host: str = "0.0.0.0",
    port: int = 8003,
    max_sessions: int = 256,
    max_inflight: int = 128,
    thread_pool_size: int = 64,
    session_timeout: float = 3600.0,
    api_key: str = "",
    workers: int = 1,
):
    """Start the ManiSkill environment server."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=thread_pool_size)

    LOGGER.info(
        f"ManiSkill Server | max_sessions={max_sessions} threads={thread_pool_size}"
    )

    handler = ManiSkillHandler(
        session_timeout=session_timeout,
        max_sessions=max_sessions,
    )
    app = GymService(handler, max_inflight=max_inflight, api_key=api_key).build()

    @app.on_event("startup")
    async def _configure_executor():
        asyncio.get_running_loop().set_default_executor(executor)

    @app.on_event("shutdown")
    def _shutdown_executor():
        executor.shutdown(wait=True)

    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == "__main__":
    fire.Fire(main)
