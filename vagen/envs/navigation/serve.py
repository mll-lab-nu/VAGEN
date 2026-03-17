"""
Navigation environment server.

Usage:
    # Auto-detect all GPUs, default settings:
    python -m vagen.envs.navigation.serve

    # Specify GPUs and limits:
    python -m vagen.envs.navigation.serve --devices='[0,1]' --max_envs=64 --port=8001
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
from vagen.envs.navigation.handler import NavigationHandler

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
    # GPU devices. None = auto-detect all visible GPUs.
    devices: Optional[List[int]] = None,
    # Max alive envs (active + cached). Bounds GPU memory.
    # Rule of thumb: ~2 envs/GB VRAM. For 2x RTX 5090 (32GB each) ≈ 128.
    max_envs: int = 128,
    # Max concurrent HTTP requests being processed. 0 = unlimited.
    # Set to ~max_envs to avoid queuing more requests than envs can serve.
    max_inflight: int = 128,
    # Thread pool for asyncio.to_thread(). Each AI2-THOR controller is a
    # separate Unity subprocess, so threads genuinely run in parallel (no GIL).
    # Should be >= max_envs so all envs can run blocking calls concurrently.
    thread_pool_size: int = 128,
    # Session idle timeout before auto-cleanup (seconds).
    session_timeout: float = 3600.0,
    # API key for authentication. Empty = no auth.
    api_key: str = "",
    # Uvicorn workers. Keep at 1 (handler state is in-process).
    workers: int = 1,
):
    """Start the Navigation environment server."""
    if devices is None:
        devices = _detect_gpus()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=thread_pool_size))

    LOGGER.info(f"GPUs: {devices} | max_envs: {max_envs} | threads: {thread_pool_size}")

    handler = NavigationHandler(devices=devices, session_timeout=session_timeout, max_envs=max_envs)
    app = GymService(handler, max_inflight=max_inflight, api_key=api_key).build()

    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == "__main__":
    fire.Fire(main)
