"""
SVG environment server.

Usage:
    # Default: DINO + DreamSim on cuda:0
    python -m vagen.envs.svg.serve

    # Custom devices and model:
    python -m vagen.envs.svg.serve --dino_device=cuda:0 --dreamsim_device=cuda:1 --model_size=large --port=8002
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging

import fire
import uvicorn

from vagen.envs_remote import GymService
from .handler import SVGHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


def main(
    host: str = "0.0.0.0",
    port: int = 8002,
    # Scoring model config
    model_size: str = "small",
    dino_device: str = "cuda:0",
    dreamsim_device: str = "cuda:0",
    preload_models: bool = True,
    # Dataset
    dataset_name: str = "starvector/svg-icons-simple",
    data_dir: str = "data",
    split: str = "train",
    # Server limits
    max_sessions: int = 256,
    max_inflight: int = 128,
    thread_pool_size: int = 64,
    session_timeout: float = 3600.0,
    # Auth
    api_key: str = "",
    workers: int = 1,
):
    """Start the SVG environment server."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=thread_pool_size)

    LOGGER.info(
        f"SVG Server | model={model_size} | dino={dino_device} dreamsim={dreamsim_device} "
        f"| max_sessions={max_sessions} threads={thread_pool_size}"
    )

    handler = SVGHandler(
        session_timeout=session_timeout,
        max_sessions=max_sessions,
        model_size=model_size,
        dino_device=dino_device,
        dreamsim_device=dreamsim_device,
        preload_models=preload_models,
        dataset_name=dataset_name,
        data_dir=data_dir,
        split=split,
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
