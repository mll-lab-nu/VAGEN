"""
Primitive skill (ManiSkill) environment server.

Uses one worker process per GPU for true multi-GPU parallelism.
ManiSkill/SAPIEN binds the render GPU at the process level, so
multi-process is the only way to use multiple GPUs.

Usage:
    # Auto-detect all GPUs, default settings:
    python -m vagen.envs.primitive_skill.serve

    # Specify GPUs and limits:
    python -m vagen.envs.primitive_skill.serve --devices='[0,1,2,3]' --max_envs_per_gpu=16 --port=8001
"""

from __future__ import annotations

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
    max_envs_per_gpu: int = 16,
    max_inflight: int = 0,
    session_timeout: float = 600.0,
    api_key: str = "",
    workers: int = 1,
):
    """
    Start the primitive_skill environment server.

    Args:
        host: Bind address.
        port: HTTP port.
        devices: GPU device IDs. Auto-detected if not specified.
        max_envs_per_gpu: Max alive envs per GPU worker (active + cached).
        max_inflight: Max concurrent HTTP requests (0 = unlimited).
        session_timeout: Idle timeout before session cleanup (seconds).
        api_key: Optional API key for authentication.
        workers: Must be 1 (sessions are in-memory).
    """
    if workers > 1:
        raise ValueError(
            f"workers={workers} is not supported. Sessions are stored in-memory "
            f"per process, so multiple workers would lose sessions across processes. "
            f"Use workers=1 (the default)."
        )

    if devices is None:
        devices = _detect_gpus()

    total_capacity = max_envs_per_gpu * len(devices)
    LOGGER.info(
        f"GPUs: {devices} | max_envs_per_gpu: {max_envs_per_gpu} | "
        f"total_capacity: {total_capacity}"
    )

    handler = PrimitiveSkillHandler(
        devices=devices,
        session_timeout=session_timeout,
        max_envs_per_gpu=max_envs_per_gpu,
    )
    service = GymService(handler, max_inflight=max_inflight, api_key=api_key)
    app = service.build()

    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == "__main__":
    fire.Fire(main)
