"""
Primitive skill (ManiSkill) environment server.

Usage:
    # Auto-detect all GPUs, default settings:
    python -m vagen.envs.primitive_skill.serve

    # Specify GPUs and limits:
    python -m vagen.envs.primitive_skill.serve --devices='[0,1]' --max_envs=32 --port=8001
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
from typing import List, Optional

import fire
import uvicorn

from vagen.envs_remote import GymService
from vagen.envs.primitive_skill.handler import PrimitiveSkillHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _detect_gpus() -> List[int]:
    """Auto-detect NVIDIA GPUs via CUDA_VISIBLE_DEVICES or nvidia-smi."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return [int(d) for d in vis.split(",") if d.strip()]
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
        )
        return [int(line.strip()) for line in out.strip().split("\n") if line.strip()]
    except Exception:
        return [0]


def main(
    host: str = "0.0.0.0",
    port: int = 8000,
    devices: Optional[List[int]] = None,
    max_envs: int = 64,
    max_inflight: int = 64,
    thread_pool_size: int = 64,
    session_timeout: float = 3600.0,
    api_key: str = "",
    workers: int = 1,
):
    """Start the primitive_skill environment server."""
    if workers > 1:
        raise ValueError(
            f"workers={workers} is not supported. Sessions are stored in-memory "
            f"per process, so multiple workers would lose sessions across processes. "
            f"Use workers=1 (the default)."
        )

    if devices is None:
        devices = _detect_gpus()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=thread_pool_size)

    LOGGER.info(f"GPUs: {devices} | max_envs: {max_envs} | threads: {thread_pool_size}")

    handler = PrimitiveSkillHandler(
        devices=devices, session_timeout=session_timeout, max_envs=max_envs
    )
    service = GymService(handler, max_inflight=max_inflight, api_key=api_key)
    app = service.build(
        startup_callback=lambda: asyncio.get_running_loop().set_default_executor(executor),
        shutdown_callback=lambda: executor.shutdown(wait=True),
    )

    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == "__main__":
    fire.Fire(main)
