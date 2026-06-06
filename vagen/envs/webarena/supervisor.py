"""WebArena supervisor: process-per-browser worker fleet.

Each worker is one `serve.py` subprocess pinned to `n_browsers=1,
max_contexts_per_browser=1` — i.e. one Chromium per process. The
supervisor health-checks workers and SIGKILL+respawns any that die
or stop responding (`max_consecutive_health_failures` strikes).

Why processes, not threads: Playwright sync calls can hang inside the
worker thread, and Python can't kill a thread. A process boundary lets
us `killpg` the whole subtree (Chromium child included) and start clean.
See memory note `[[webarena-browserpool-deadlock]]`.

Usage:
    source vagen/envs/webarena/setup_vars.sh
    PYTHONPATH=. python -m vagen.envs.webarena.supervisor \\
        --n_workers=8 --start_port=8002 \\
        --task_config_file=vagen/envs/webarena/config_files/normalized_test.json \\
        --auth_cache_dir=./vagen/envs/webarena/.wa_auth

Workers run on ports start_port .. start_port + n_workers - 1.
Clients pass that list to GymImageEnvClient as `base_urls=[...]`;
the client already does sticky-session routing with failover on 5xx.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import fire
import httpx


# Linux: PR_SET_PDEATHSIG. Tell the kernel to send signal N to this process
# when its parent dies. Used in the worker preexec_fn so an orphaned worker
# is killed if the supervisor crashes (otherwise start_new_session=True
# leaves them running indefinitely).
_PR_SET_PDEATHSIG = 1


def _set_parent_death_signal() -> None:
    """preexec_fn for worker subprocesses. SIGKILL on parent death."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # SIGKILL = 9 (can't import signal in preexec — exec hasn't happened)
        libc.prctl(_PR_SET_PDEATHSIG, 9, 0, 0, 0)
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("supervisor")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


@dataclass
class WorkerSpec:
    worker_id: int
    port: int
    log_path: Path
    task_config_file: str
    auth_cache_dir: str
    session_timeout: float


class Worker:
    """One `serve.py` subprocess owned by the supervisor."""

    def __init__(self, spec: WorkerSpec):
        self.spec = spec
        self.proc: Optional[asyncio.subprocess.Process] = None
        self._log_fp = None
        self._last_start_at: float = 0.0
        self._restart_count: int = 0
        self._failed_health: int = 0

    def base_url(self) -> str:
        return f"http://localhost:{self.spec.port}"

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self) -> None:
        if self.is_alive():
            return
        self.spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Append so successive restarts share a log
        self._log_fp = open(self.spec.log_path, "a", buffering=1)
        self._log_fp.write(
            f"\n=== [supervisor] worker {self.spec.worker_id} "
            f"start (restart #{self._restart_count}) at "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
        )

        argv = [
            sys.executable, "-u", "-m", "vagen.envs.webarena.serve",
            f"--port={self.spec.port}",
            f"--task_config_file={self.spec.task_config_file}",
            f"--auth_cache_dir={self.spec.auth_cache_dir}",
            "--n_browsers=1",
            "--max_contexts_per_browser=1",
            f"--session_timeout={self.spec.session_timeout}",
        ]

        # Cap numerical-library thread pools. Each worker imports
        # numpy/scipy/torch transitively (via Playwright→PIL→numpy,
        # evaluator→nltk etc.); on a 128-core node OpenBLAS spawns
        # ~64 threads per process at import time. With N workers
        # that's N×64 threads burned before serving a single request,
        # and can hit pthread_create EAGAIN under burst.
        worker_env = os.environ.copy()
        worker_env.setdefault("OPENBLAS_NUM_THREADS", "1")
        worker_env.setdefault("OMP_NUM_THREADS", "1")
        worker_env.setdefault("MKL_NUM_THREADS", "1")
        worker_env.setdefault("NUMEXPR_NUM_THREADS", "1")
        worker_env.setdefault("TOKENIZERS_PARALLELISM", "false")

        # start_new_session=True puts the child in its own process group,
        # so killpg() reaches Chromium and its renderer children too.
        # preexec_fn sets PR_SET_PDEATHSIG so an orphaned worker dies when
        # the supervisor crashes (instead of lingering as a runaway).
        self.proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=self._log_fp,
            stderr=asyncio.subprocess.STDOUT,
            env=worker_env,
            start_new_session=True,
            preexec_fn=_set_parent_death_signal,
        )
        self._last_start_at = time.time()
        self._failed_health = 0
        LOGGER.info(
            f"[worker {self.spec.worker_id}] started pid={self.proc.pid} "
            f"port={self.spec.port} log={self.spec.log_path}"
        )

    async def kill_and_wait(self, grace: float = 5.0) -> None:
        if not self.is_alive():
            self._close_log()
            return
        assert self.proc is not None
        try:
            pgid = os.getpgid(self.proc.pid)
        except ProcessLookupError:
            self._close_log()
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            self._close_log()
            return
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=grace)
        except asyncio.TimeoutError:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                LOGGER.warning(
                    f"[worker {self.spec.worker_id}] did not die after SIGKILL"
                )
        self._close_log()

    def _close_log(self) -> None:
        if self._log_fp is not None:
            try:
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None

    async def health(self, hc: httpx.AsyncClient, timeout: float) -> bool:
        try:
            r = await hc.get(f"{self.base_url()}/health", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


class Supervisor:
    def __init__(
        self,
        n_workers: int,
        start_port: int,
        task_config_file: str,
        auth_cache_dir: str,
        log_dir: str,
        session_timeout: float,
        health_interval: float,
        health_timeout: float,
        max_consecutive_health_failures: int,
        startup_grace: float,
        startup_stagger: float = 1.5,
    ):
        self.workers: List[Worker] = [
            Worker(WorkerSpec(
                worker_id=i,
                port=start_port + i,
                log_path=Path(log_dir) / f"worker_{i}_port_{start_port + i}.log",
                task_config_file=task_config_file,
                auth_cache_dir=auth_cache_dir,
                session_timeout=session_timeout,
            ))
            for i in range(n_workers)
        ]
        self.health_interval = health_interval
        self.health_timeout = health_timeout
        self.max_consecutive_health_failures = max_consecutive_health_failures
        self.startup_grace = startup_grace
        self.startup_stagger = startup_stagger
        self._stopping = False

    def base_urls(self) -> List[str]:
        return [w.base_url() for w in self.workers]

    async def start_all(self) -> None:
        # Stagger so we don't fork N Chromiums at the exact same instant.
        # Simultaneous Chromium launches with their renderer/zygote
        # children burst-create hundreds of threads/procs, which can hit
        # transient `pthread_create EAGAIN` on dense nodes.
        for i, w in enumerate(self.workers):
            await w.start()
            if i < len(self.workers) - 1 and self.startup_stagger > 0:
                await asyncio.sleep(self.startup_stagger)
        LOGGER.info(
            f"[supervisor] launched {len(self.workers)} workers; "
            f"giving them {self.startup_grace}s before health checks"
        )

    async def health_loop(self) -> None:
        async with httpx.AsyncClient() as hc:
            while not self._stopping:
                await asyncio.sleep(self.health_interval)
                if self._stopping:
                    return
                for w in self.workers:
                    if self._stopping:
                        return
                    await self._check_one(w, hc)

    async def _check_one(self, w: Worker, hc: httpx.AsyncClient) -> None:
        # Crashed?
        if not w.is_alive():
            rc = w.proc.returncode if w.proc is not None else "?"
            LOGGER.warning(
                f"[worker {w.spec.worker_id}] exited (rc={rc}); restarting"
            )
            w._restart_count += 1
            await w.start()
            return

        # Still in startup grace window?
        if time.time() - w._last_start_at < self.startup_grace:
            return

        ok = await w.health(hc, timeout=self.health_timeout)
        if ok:
            w._failed_health = 0
            return

        w._failed_health += 1
        LOGGER.warning(
            f"[worker {w.spec.worker_id}] health fail "
            f"{w._failed_health}/{self.max_consecutive_health_failures}"
        )
        if w._failed_health < self.max_consecutive_health_failures:
            return

        LOGGER.warning(
            f"[worker {w.spec.worker_id}] hung — SIGKILL + restart"
        )
        w._restart_count += 1
        await w.kill_and_wait()
        await w.start()

    async def shutdown(self) -> None:
        self._stopping = True
        LOGGER.info(f"[supervisor] shutting down {len(self.workers)} workers")
        await asyncio.gather(
            *(w.kill_and_wait() for w in self.workers),
            return_exceptions=True,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _check_env_vars() -> None:
    required = ("DATASET", "REDDIT", "SHOPPING", "SHOPPING_ADMIN",
                "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE")
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        LOGGER.error("Missing env vars: %s", missing)
        LOGGER.error("Source vagen/envs/webarena/setup_vars.sh before starting.")
        sys.exit(1)


async def _amain(**kwargs) -> None:
    sup = Supervisor(**kwargs)
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _on_signal(sig: signal.Signals) -> None:
        LOGGER.info(f"[supervisor] got {sig.name}; shutting down")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal, sig)

    health_task: Optional[asyncio.Task] = None
    try:
        await sup.start_all()
        LOGGER.info(f"[supervisor] base_urls: {sup.base_urls()}")
        health_task = asyncio.create_task(sup.health_loop())
        await stop_event.wait()
    finally:
        if health_task is not None:
            health_task.cancel()
        await sup.shutdown()


def main(
    n_workers: int = 8,
    start_port: int = 8002,
    task_config_file: str = "vagen/envs/webarena/config_files/normalized_test.json",
    auth_cache_dir: str = "./vagen/envs/webarena/.wa_auth",
    log_dir: str = "./log_files/webarena_workers",
    session_timeout: float = 3600.0,
    health_interval: float = 30.0,
    health_timeout: float = 10.0,
    max_consecutive_health_failures: int = 3,
    startup_grace: float = 30.0,
    startup_stagger: float = 1.5,
):
    """Launch N independent `serve.py` workers under one supervisor.

    Args:
        n_workers: How many worker processes to run. Each owns 1 Chromium.
        start_port: First port; workers use start_port .. start_port + n - 1.
        task_config_file: Aggregate task JSON (passed to every worker).
        auth_cache_dir: Shared cookie cache (workers read the same files).
        log_dir: Per-worker stdout/stderr logs land here.
        session_timeout: Forwarded to each worker's handler.
        health_interval: Seconds between health sweeps.
        health_timeout: Per-request timeout for /health probe.
        max_consecutive_health_failures: Strikes before SIGKILL+restart.
        startup_grace: Skip health checks for this long after each (re)start
            — auth-cache regen and Playwright launch take ~10-20s.
        startup_stagger: Seconds between worker launches. 0 = launch all at
            once; >0 avoids burst thread/proc creation when N is large.
    """
    _check_env_vars()
    asyncio.run(_amain(
        n_workers=n_workers,
        start_port=start_port,
        task_config_file=os.path.abspath(task_config_file),
        auth_cache_dir=os.path.abspath(auth_cache_dir),
        log_dir=os.path.abspath(log_dir),
        session_timeout=session_timeout,
        health_interval=health_interval,
        health_timeout=health_timeout,
        max_consecutive_health_failures=max_consecutive_health_failures,
        startup_grace=startup_grace,
        startup_stagger=startup_stagger,
    ))


if __name__ == "__main__":
    fire.Fire(main)
