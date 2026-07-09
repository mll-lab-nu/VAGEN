"""WebArena environment server.

Usage:
    # 1) source setup_vars.sh to export DATASET + URL env vars
    source vagen/envs/webarena/setup_vars.sh

    # 2) run
    python -m vagen.envs.webarena.serve \\
        --task_config_file=vagen/envs/webarena/config_files/normalized_test.json \\
        --n_browsers=4 \\
        --max_contexts_per_browser=16 \\
        --port=8002
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import fire
import uvicorn

from vagen.envs_remote import GymService
from vagen.envs.webarena.handler import WebArenaHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _check_env_vars() -> None:
    missing = [v for v in ("DATASET", "REDDIT", "SHOPPING", "SHOPPING_ADMIN",
                            "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE")
               if not os.environ.get(v)]
    if missing:
        LOGGER.error("Missing env vars: %s", missing)
        LOGGER.error("Source setup_vars.sh before starting the server.")
        sys.exit(1)
    LOGGER.info("DATASET=%s", os.environ["DATASET"])
    for k in ("REDDIT", "SHOPPING", "SHOPPING_ADMIN", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE"):
        LOGGER.info("%s=%s", k, os.environ.get(k))


def main(
    host: str = "0.0.0.0",
    port: int = 8002,
    task_config_file: str = "vagen/envs/webarena/config_files/normalized_test.json",
    auth_cache_dir: str = "./.webarena_auth_cache",
    # Number of Chromium browsers in the pool. Each pins one OS thread.
    n_browsers: int = 4,
    # Max concurrent contexts per browser. Total capacity = n_browsers × this.
    max_contexts_per_browser: int = 16,
    # Session idle timeout before auto-cleanup (seconds).
    session_timeout: float = 3600.0,
    # Max concurrent HTTP requests being processed (0 = unlimited).
    max_inflight: int = 0,
    # API key for authentication. Empty = no auth.
    api_key: str = "",
    # Uvicorn workers. Must be 1 (handler state is in-process).
    workers: int = 1,
):
    if workers > 1:
        raise ValueError(f"workers={workers} not supported; use 1.")

    _check_env_vars()

    task_config_file = os.path.abspath(task_config_file)
    auth_cache_dir = os.path.abspath(auth_cache_dir)

    LOGGER.info(
        "Starting WebArena server: "
        f"pool={n_browsers}×{max_contexts_per_browser}="
        f"{n_browsers * max_contexts_per_browser} contexts, "
        f"tasks={task_config_file}, auth_cache={auth_cache_dir}"
    )

    handler = WebArenaHandler(
        task_config_file=task_config_file,
        auth_cache_dir=auth_cache_dir,
        n_browsers=n_browsers,
        max_contexts_per_browser=max_contexts_per_browser,
        session_timeout=session_timeout,
    )
    service = GymService(handler, max_inflight=max_inflight, api_key=api_key)
    # Pool starts lazily on first connect() (handler.start() is idempotent
    # and guarded by an asyncio lock). GymService.build() already calls
    # handler.aclose() on shutdown, so no shutdown_callback needed.
    app = service.build(
        startup_callback=lambda: asyncio.create_task(handler.start()),
    )

    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == "__main__":
    fire.Fire(main)
