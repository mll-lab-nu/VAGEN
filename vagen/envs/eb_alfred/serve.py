"""
EB-ALFRED Remote Environment Server.

Starts a FastAPI service that exposes EB-ALFRED as a remote gym environment.
The service can run on a machine with GPU + X server (for AI2-THOR rendering),
while VAGEN RL training runs on a separate machine using GymImageEnvClient.

GPUs and X servers are auto-detected and started automatically.  You only
need to override them if the defaults don't work for your setup.

Usage:
    # Auto-detect all GPUs, start Xorg automatically:
    python -m vagen.envs.eb_alfred.serve

    # Override GPU list or other settings:
    python -m vagen.envs.eb_alfred.serve --devices='[0,1]' --capacity=64 --port=8001

    # Then on the training machine, configure env_config:
    #   base_urls: ["http://<server-ip>:8000"]
    #   eval_set: "base"
    #   resolution: 500
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import subprocess
import time
from typing import List, Optional

import fire
import uvicorn

from vagen.envs_remote import GymService
from .handler import EbAlfredHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

_XORG_CONF_TEMPLATE = """\
Section "Device"
    Identifier  "GPU{idx}"
    Driver      "nvidia"
    BusID       "{bus_id}"
    Option      "AllowEmptyInitialConfiguration" "True"
EndSection
Section "Monitor"
    Identifier  "Monitor{idx}"
    HorizSync   28.0-80.0
    VertRefresh 48.0-75.0
    Modeline    "1920x1080" 172.80 1920 2040 2248 2576 1080 1081 1084 1118
EndSection
Section "Screen"
    Identifier  "Screen{idx}"
    Device      "GPU{idx}"
    Monitor     "Monitor{idx}"
    DefaultDepth 24
    SubSection  "Display"
        Depth   24
        Modes   "1920x1080"
        Virtual 1920 1080
    EndSubSection
EndSection
Section "ServerLayout"
    Identifier "Layout{idx}"
    Screen 0   "Screen{idx}"
EndSection
"""


def _detect_gpus() -> List[int]:
    """Auto-detect NVIDIA GPU indices via CUDA_VISIBLE_DEVICES or nvidia-smi."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return [int(d) for d in vis.split(",") if d.strip()]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
        )
        return [int(line.strip()) for line in out.strip().split("\n") if line.strip()]
    except Exception:
        return [0]


def _get_pci_bus_id(gpu_index: int) -> str:
    """Return xorg-format PCI BusID for a GPU (e.g. 'PCI:65:0:0').

    nvidia-smi reports PCI IDs in hex (e.g. '0000:41:00.0').
    Xorg BusID uses decimal (e.g. 'PCI:65:0:0').
    """
    out = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
        text=True,
    ).strip()
    # Format: "0000:BUS:DEV.FUNC" (all hex)
    _, bus_hex, dev_func = out.split(":")
    dev_hex, func_hex = dev_func.split(".")
    return f"PCI:{int(bus_hex, 16)}:{int(dev_hex, 16)}:{int(func_hex, 16)}"


def _xorg_running(display: int) -> bool:
    """Check if an X server is already running on the given display."""
    return (os.path.exists(f"/tmp/.X{display}-lock")
            or os.path.exists(f"/tmp/.X11-unix/X{display}"))


def _start_xorg(gpu_index: int, display: int) -> None:
    """Generate xorg.conf and (re)start Xorg for a GPU/display pair.

    Skips silently if Xorg is already running on that display.
    """
    if _xorg_running(display):
        LOGGER.info(f"Xorg already running on :{display}, skipping")
        return

    pci_bus_id = _get_pci_bus_id(gpu_index)
    conf_path = f"/tmp/xorg{display}.conf"
    with open(conf_path, "w") as f:
        f.write(_XORG_CONF_TEMPLATE.format(idx=display, bus_id=pci_bus_id))

    LOGGER.info(f"Starting Xorg :{display} for GPU {gpu_index} (BusID={pci_bus_id})")
    subprocess.Popen(
        ["Xorg", "-noreset", "+extension", "GLX", "-config", conf_path, f":{display}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(15):
        time.sleep(1)
        if _xorg_running(display):
            LOGGER.info(f"Xorg :{display} ready")
            return

    raise RuntimeError(f"Xorg :{display} did not become ready within 15 seconds")


def main(
    host: str = "0.0.0.0",
    port: int = 8000,
    # GPU device indices. None = auto-detect via CUDA_VISIBLE_DEVICES or nvidia-smi.
    # Convention: GPU i is assigned to X display :i.
    devices: Optional[List[int]] = None,
    # Max concurrently running Unity environments (0 = unlimited).
    # Extra /connect requests are queued and served as slots free up.
    capacity: int = 16,
    # Max Unity processes starting up simultaneously (0 = unlimited).
    # Prevents CPU spikes when many capacity slots open at once.
    startup_concurrency: int = 8,
    # Max idle envs kept alive in pool for instant reuse.
    # -1 = same as capacity (default), 0 = disable pooling.
    pool_size: int = -1,
    # Thread pool for asyncio.to_thread(). Should be >= capacity.
    thread_pool_size: int = 128,
    # Session idle timeout before auto-cleanup (seconds).
    session_timeout: float = 3600.0,
    # Max total sessions (0 = unlimited).
    max_sessions: int = 0,
    # API key for authentication. Empty = no auth.
    api_key: str = "",
    # Uvicorn workers. Keep at 1 (handler state is in-process).
    workers: int = 1,
):
    """Start the EB-ALFRED environment server."""
    if devices is None:
        devices = _detect_gpus()

    # Start one Xorg server per GPU (display :i = GPU i)
    for gpu_idx in devices:
        _start_xorg(gpu_idx, display=gpu_idx)
    x_displays = [str(gpu_idx) for gpu_idx in devices]

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=thread_pool_size)

    LOGGER.info(
        f"GPUs: {devices} | displays: {x_displays} | "
        f"capacity: {capacity} | startup_concurrency: {startup_concurrency} | "
        f"pool_size: {pool_size} | threads: {thread_pool_size}"
    )

    handler = EbAlfredHandler(
        x_displays=x_displays,
        capacity=capacity,
        startup_concurrency=startup_concurrency,
        pool_size=pool_size,
        session_timeout=session_timeout,
        max_sessions=max_sessions,
    )
    app = GymService(handler, api_key=api_key).build()

    @app.on_event("startup")
    async def _configure_executor():
        asyncio.get_running_loop().set_default_executor(executor)

    @app.on_event("shutdown")
    def _shutdown_executor():
        executor.shutdown(wait=True)

    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == "__main__":
    fire.Fire(main)
