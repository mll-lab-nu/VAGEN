"""
Handler for the primitive_skill (ManiSkill) environment.

Architecture:
    One worker **process** per GPU, each with its own env cache.
    The main (asyncio) process communicates with workers via mp.Queue pairs.
    This is necessary because ManiSkill/SAPIEN binds the render GPU at the
    process level (via CUDA_VISIBLE_DEVICES), so a single-process design
    cannot spread rendering across multiple GPUs.

Resource model:
    max_envs_per_gpu limits alive envs (active + cached) per worker.
    Total capacity = num_gpus * max_envs_per_gpu.

Caching:
    On client close, the env is NOT destroyed — it stays alive in the
    worker's cache. On next connect with the same env_id on the same
    worker, the cached env is reused (fast reset).
"""

from __future__ import annotations

import asyncio
import io
import logging
import multiprocessing as mp
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from vagen.envs_remote.handler import BaseGymHandler, HandlerResult, SessionContext

LOGGER = logging.getLogger(__name__)


# ======================================================================
# Worker process
# ======================================================================

def _worker_main(
    gpu_id: int,
    cuda_device: int,
    task_q: mp.Queue,
    result_q: mp.Queue,
    max_envs: int,
):
    """
    Worker process entry point.  Runs in a loop, processing commands from
    *task_q* and posting results to *result_q*.

    Each worker owns one GPU (set via CUDA_VISIBLE_DEVICES before any
    import of ManiSkill / SAPIEN / torch).
    """
    # ---- bind GPU BEFORE importing anything that touches CUDA ----------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    import numpy as np

    # Pre-load PartNet Mobility dataset (needed by PutAppleInDrawer)
    try:
        from mani_skill.utils.building.articulations.partnet_mobility import (
            _load_partnet_mobility_dataset,
            PARTNET_MOBILITY,
        )
        if PARTNET_MOBILITY is None or "model_urdf_paths" not in PARTNET_MOBILITY:
            _load_partnet_mobility_dataset()
            result_q.put((-1, "log", f"[Worker GPU {gpu_id}] PartNet Mobility loaded"))
    except Exception as e:
        result_q.put((-1, "log", f"[Worker GPU {gpu_id}] PartNet Mobility load failed: {e}"))

    from vagen.envs.primitive_skill.primitive_skill_env import PrimitiveSkillEnv

    # ---- per-worker state ------------------------------------------------
    # env_slot -> PrimitiveSkillEnv  (active sessions)
    active: Dict[str, PrimitiveSkillEnv] = {}
    # list of (env_id, PrimitiveSkillEnv)  (idle cached envs)
    cache: List[Tuple[str, PrimitiveSkillEnv]] = []

    def _total_alive():
        return len(active) + len(cache)

    def _pop_cached(env_id: str) -> Optional[PrimitiveSkillEnv]:
        """Try to pop a cached env matching env_id; else pop any; else None."""
        for i, (cid, env) in enumerate(cache):
            if cid == env_id:
                cache.pop(i)
                return env
        return None

    def _evict_one():
        """Destroy one cached env to free capacity."""
        if cache:
            _, env = cache.pop()
            try:
                env._sync_close()
            except Exception:
                pass

    def _acquire(env_config: dict) -> PrimitiveSkillEnv:
        env_id = env_config.get("env_id", "AlignTwoCube")
        # try cache hit
        env = _pop_cached(env_id)
        if env is not None:
            return env
        # need a new env — evict if at capacity
        while _total_alive() >= max_envs:
            if cache:
                _evict_one()
            else:
                raise RuntimeError(
                    f"Worker GPU {gpu_id}: at capacity ({_total_alive()}/{max_envs}), "
                    f"no cached envs to evict"
                )
        env = PrimitiveSkillEnv(env_config)
        return env

    def _img_to_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _encode_obs(obs: dict) -> dict:
        """Replace PIL images with PNG bytes for pickling across processes."""
        encoded = {}
        for k, v in obs.items():
            if k == "multi_modal_input" and isinstance(v, dict):
                encoded[k] = {
                    mk: [_img_to_bytes(img) for img in mv]
                    for mk, mv in v.items()
                }
            else:
                encoded[k] = v
        return encoded

    result_q.put((-1, "log", f"[Worker GPU {gpu_id}] Ready (cuda_device={cuda_device}, max_envs={max_envs})"))

    # ---- main loop -------------------------------------------------------
    while True:
        try:
            msg = task_q.get()
            if msg is None:
                break  # shutdown signal

            req_id, cmd, args = msg

            if cmd == "connect":
                env_config, seed = args
                try:
                    env = _acquire(env_config)
                    slot = uuid.uuid4().hex
                    active[slot] = env

                    result_data = {"slot": slot}
                    images_bytes = None

                    if seed is not None:
                        obs, info = env._sync_reset(seed)
                        result_data["obs"] = obs.get("obs_str", "")
                        result_data["info"] = info
                        encoded_obs = _encode_obs(obs)
                        images_bytes = encoded_obs.get("multi_modal_input")

                    result_q.put((req_id, "ok", (result_data, images_bytes)))
                except Exception as e:
                    result_q.put((req_id, "err", str(e)))

            elif cmd == "system_prompt":
                slot = args
                try:
                    env = active[slot]
                    # system_prompt is sync-safe (no GPU needed for text)
                    import asyncio as _aio
                    obs = _aio.run(env.system_prompt())
                    result_q.put((req_id, "ok", ({"obs": obs.get("obs_str", "")}, None)))
                except Exception as e:
                    result_q.put((req_id, "err", str(e)))

            elif cmd == "reset":
                slot, seed = args
                try:
                    env = active[slot]
                    obs, info = env._sync_reset(seed)
                    encoded_obs = _encode_obs(obs)
                    images_bytes = encoded_obs.get("multi_modal_input")
                    result_data = {"obs": obs.get("obs_str", ""), "info": info}
                    result_q.put((req_id, "ok", (result_data, images_bytes)))
                except Exception as e:
                    result_q.put((req_id, "err", str(e)))

            elif cmd == "step":
                slot, action_str = args
                try:
                    env = active[slot]
                    obs, reward, done, info = env._sync_step(action_str)
                    encoded_obs = _encode_obs(obs)
                    images_bytes = encoded_obs.get("multi_modal_input")
                    result_data = {
                        "obs": obs.get("obs_str", ""),
                        "reward": reward,
                        "done": done,
                        "info": info,
                    }
                    result_q.put((req_id, "ok", (result_data, images_bytes)))
                except Exception as e:
                    result_q.put((req_id, "err", str(e)))

            elif cmd == "close":
                slot = args
                try:
                    env = active.pop(slot, None)
                    if env is not None:
                        env_id = env.cfg.env_id
                        cache.append((env_id, env))
                    result_q.put((req_id, "ok", ({"closed": True}, None)))
                except Exception as e:
                    result_q.put((req_id, "err", str(e)))

            elif cmd == "destroy":
                # Force-destroy a slot (for timeout cleanup)
                slot = args
                try:
                    env = active.pop(slot, None)
                    if env is not None:
                        env._sync_close()
                    result_q.put((req_id, "ok", ({"destroyed": True}, None)))
                except Exception as e:
                    result_q.put((req_id, "err", str(e)))

            else:
                result_q.put((req_id, "err", f"Unknown command: {cmd}"))

        except Exception as e:
            try:
                result_q.put((-1, "err", f"Worker GPU {gpu_id} loop error: {e}"))
            except Exception:
                pass

    # ---- shutdown --------------------------------------------------------
    for _, env in active.items():
        try:
            env._sync_close()
        except Exception:
            pass
    for _, env in cache:
        try:
            env._sync_close()
        except Exception:
            pass
    result_q.put((-1, "log", f"[Worker GPU {gpu_id}] Shut down"))


# ======================================================================
# Handler (runs in main asyncio process)
# ======================================================================

@dataclass
class _SessionInfo:
    """Lightweight session metadata kept in the main process."""
    session_id: str
    gpu_id: int       # which worker owns this env
    slot: str         # slot id within the worker
    created_at: float
    last_access: float


class PrimitiveSkillHandler(BaseGymHandler):
    """
    Multi-process handler for ManiSkill environments.

    One worker process per GPU, communicating via mp.Queue.
    The main process (asyncio) dispatches requests to workers and
    collects results asynchronously.

    Usage:
        handler = PrimitiveSkillHandler(devices=[0,1,2,3], max_envs_per_gpu=16)
        app = GymService(handler).build()
    """

    def __init__(
        self,
        devices: Optional[List[int]] = None,
        session_timeout: float = 600.0,
        max_envs_per_gpu: int = 16,
        acquire_timeout: float = 300.0,
    ):
        super().__init__(session_timeout=session_timeout, max_sessions=0)
        self.devices = devices or [0]
        self.max_envs_per_gpu = max_envs_per_gpu
        self.acquire_timeout = acquire_timeout

        # Per-GPU worker state
        self._task_qs: Dict[int, mp.Queue] = {}
        self._result_qs: Dict[int, mp.Queue] = {}
        self._processes: Dict[int, mp.Process] = {}

        # Pending async futures: req_id -> asyncio.Future
        self._pending: Dict[int, asyncio.Future] = {}
        self._req_counter = 0

        # Session tracking (main process only, lightweight)
        self._session_info: Dict[str, _SessionInfo] = {}
        # Active env count per GPU (tracked in main process for load balancing)
        self._active_count: Dict[int, int] = {d: 0 for d in self.devices}

        # Background tasks
        self._reader_tasks: List[asyncio.Task] = []
        self._started = False

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _start_workers(self):
        if self._started:
            return
        ctx = mp.get_context("fork")
        for gpu_id in self.devices:
            tq = ctx.Queue()
            rq = ctx.Queue()
            p = ctx.Process(
                target=_worker_main,
                args=(gpu_id, gpu_id, tq, rq, self.max_envs_per_gpu),
                daemon=True,
            )
            p.start()
            self._task_qs[gpu_id] = tq
            self._result_qs[gpu_id] = rq
            self._processes[gpu_id] = p
            LOGGER.info(f"[PrimSkillHandler] Started worker process pid={p.pid} for GPU {gpu_id}")
        self._started = True

    def _ensure_readers(self):
        """Ensure background reader tasks are running (one per GPU)."""
        if self._reader_tasks:
            return
        for gpu_id in self.devices:
            task = asyncio.create_task(self._reader_loop(gpu_id))
            self._reader_tasks.append(task)

    async def _reader_loop(self, gpu_id: int):
        """Read results from a worker's result queue and resolve futures."""
        rq = self._result_qs[gpu_id]
        loop = asyncio.get_event_loop()
        while True:
            try:
                req_id, status, payload = await loop.run_in_executor(None, rq.get)
                if req_id == -1:
                    # log message from worker
                    if status == "log":
                        LOGGER.info(payload)
                    else:
                        LOGGER.error(payload)
                    continue

                fut = self._pending.pop(req_id, None)
                if fut is not None and not fut.done():
                    if status == "ok":
                        fut.set_result(payload)
                    else:
                        fut.set_exception(RuntimeError(payload))
            except asyncio.CancelledError:
                break
            except Exception as e:
                LOGGER.error(f"[PrimSkillHandler] Reader GPU {gpu_id} error: {e}")

    # ------------------------------------------------------------------
    # Send command to worker
    # ------------------------------------------------------------------

    async def _send(self, gpu_id: int, cmd: str, args: Any, timeout: Optional[float] = None) -> Any:
        """Send a command to a worker and await the result."""
        self._req_counter += 1
        req_id = self._req_counter

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut

        self._task_qs[gpu_id].put((req_id, cmd, args))

        try:
            result = await asyncio.wait_for(fut, timeout=timeout or self.acquire_timeout)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise RuntimeError(
                f"Timeout ({timeout or self.acquire_timeout}s) waiting for worker GPU {gpu_id} "
                f"cmd={cmd}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pick_device(self, env_id: str = "") -> int:
        """Pick GPU with fewest active envs."""
        return min(self._active_count, key=self._active_count.get)

    @staticmethod
    def _decode_images(images_bytes: Optional[dict]) -> Optional[List[Image.Image]]:
        """Convert PNG bytes back to PIL Images."""
        if images_bytes is None:
            return None
        for key, img_list in images_bytes.items():
            return [Image.open(io.BytesIO(b)) for b in img_list]
        return None

    # ------------------------------------------------------------------
    # BaseGymHandler abstract method (unused, we override connect/call)
    # ------------------------------------------------------------------

    async def create_env(self, env_config: Dict[str, Any]) -> Any:
        raise NotImplementedError("Envs are created inside worker processes")

    # ------------------------------------------------------------------
    # Connect
    # ------------------------------------------------------------------

    async def connect(
        self, env_config: Dict[str, Any], seed: Optional[int] = None
    ) -> HandlerResult:
        self._start_workers()
        self._ensure_readers()

        env_id = env_config.get("env_id", "AlignTwoCube")
        gpu_id = self._pick_device(env_id)
        self._active_count[gpu_id] += 1

        try:
            result_data, images_bytes = await self._send(
                gpu_id, "connect", (env_config, seed)
            )
        except BaseException:
            self._active_count[gpu_id] = max(0, self._active_count[gpu_id] - 1)
            raise

        slot = result_data.pop("slot")
        session_id = uuid.uuid4().hex

        self._session_info[session_id] = _SessionInfo(
            session_id=session_id,
            gpu_id=gpu_id,
            slot=slot,
            created_at=time.time(),
            last_access=time.time(),
        )

        # Also register in base class _sessions for compatibility
        self._sessions[session_id] = SessionContext(
            session_id=session_id,
            env=None,  # env lives in worker process
            created_at=time.time(),
            last_access=time.time(),
        )

        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        LOGGER.info(
            f"[PrimSkillHandler] Session {session_id[:8]} GPU {gpu_id} slot {slot[:8]} "
            f"({self._stats_str()})"
        )

        result_data["session_id"] = session_id
        images = self._decode_images(images_bytes)
        return HandlerResult(data=result_data, images=images)

    # ------------------------------------------------------------------
    # Call (dispatch to worker)
    # ------------------------------------------------------------------

    async def call(
        self,
        session_id: str,
        method: str,
        params: Dict[str, Any],
        images: List[Image.Image],
    ) -> HandlerResult:
        from vagen.envs_remote.handler import SessionNotFoundError

        info = self._session_info.get(session_id)
        if info is None:
            raise SessionNotFoundError(f"Session {session_id} not found")
        info.last_access = time.time()
        if session_id in self._sessions:
            self._sessions[session_id].last_access = time.time()

        gpu_id = info.gpu_id
        slot = info.slot

        if method == "system_prompt":
            result_data, images_bytes = await self._send(gpu_id, "system_prompt", slot)
            return HandlerResult(data=result_data, images=self._decode_images(images_bytes))

        elif method == "reset":
            seed = params.get("seed", 0)
            result_data, images_bytes = await self._send(gpu_id, "reset", (slot, seed))
            return HandlerResult(data=result_data, images=self._decode_images(images_bytes))

        elif method == "step":
            action_str = params.get("action_str", "")
            result_data, images_bytes = await self._send(gpu_id, "step", (slot, action_str))
            return HandlerResult(data=result_data, images=self._decode_images(images_bytes))

        elif method == "close":
            return await self._handle_close_session(session_id)

        else:
            raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    async def _handle_close(self, ctx: SessionContext) -> HandlerResult:
        """Called by base class — redirect to our close logic."""
        return await self._handle_close_session(ctx.session_id)

    async def _handle_close_session(self, session_id: str) -> HandlerResult:
        info = self._session_info.pop(session_id, None)
        self._sessions.pop(session_id, None)

        if info is None:
            return HandlerResult(data={"closed": True})

        self._active_count[info.gpu_id] = max(0, self._active_count[info.gpu_id] - 1)

        try:
            await self._send(info.gpu_id, "close", info.slot, timeout=30.0)
        except Exception as e:
            LOGGER.warning(f"[PrimSkillHandler] Error closing slot {info.slot[:8]}: {e}")

        LOGGER.info(
            f"[PrimSkillHandler] Session {session_id[:8]} closed GPU {info.gpu_id} "
            f"({self._stats_str()})"
        )
        return HandlerResult(data={"closed": True})

    # ------------------------------------------------------------------
    # Cleanup loop (override base class)
    # ------------------------------------------------------------------

    async def _cleanup_loop(self):
        while True:
            try:
                await asyncio.sleep(60)
                now = time.time()
                to_remove = []

                for session_id, info in self._session_info.items():
                    if now - info.last_access > self.session_timeout:
                        to_remove.append(session_id)
                        LOGGER.warning(
                            f"[PrimSkillHandler] Session {session_id[:8]} timed out "
                            f"after {now - info.last_access:.0f}s idle"
                        )

                for session_id in to_remove:
                    info = self._session_info.pop(session_id, None)
                    self._sessions.pop(session_id, None)
                    if info is None:
                        continue
                    self._active_count[info.gpu_id] = max(0, self._active_count[info.gpu_id] - 1)
                    try:
                        await self._send(info.gpu_id, "destroy", info.slot, timeout=30.0)
                    except Exception as e:
                        LOGGER.error(
                            f"[PrimSkillHandler] Cleanup destroy error "
                            f"{session_id[:8]}: {e}"
                        )
                    LOGGER.info(
                        f"[PrimSkillHandler] Cleaned up session {session_id[:8]} "
                        f"GPU {info.gpu_id} ({self._stats_str()})"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                LOGGER.error(f"[PrimSkillHandler] Cleanup loop error: {e}")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _stats_str(self) -> str:
        return f"active={dict(self._active_count)} total_sessions={len(self._session_info)}"

    def get_session_stats(self) -> Dict[str, Any]:
        stats = super().get_session_stats()
        stats["active_per_device"] = dict(self._active_count)
        stats["max_envs_per_gpu"] = self.max_envs_per_gpu
        stats["total_capacity"] = self.max_envs_per_gpu * len(self.devices)
        stats["worker_pids"] = {
            gpu_id: p.pid for gpu_id, p in self._processes.items()
        }
        return stats

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def aclose(self):
        # Cancel reader tasks
        for t in self._reader_tasks:
            t.cancel()
        for t in self._reader_tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._reader_tasks.clear()

        # Send shutdown signal to all workers
        for gpu_id, tq in self._task_qs.items():
            try:
                tq.put(None)  # shutdown signal
            except Exception:
                pass

        # Wait for workers to exit
        for gpu_id, p in self._processes.items():
            try:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()
                    LOGGER.warning(f"[PrimSkillHandler] Force-terminated worker GPU {gpu_id}")
            except Exception:
                pass

        self._session_info.clear()
        self._sessions.clear()
        self._pending.clear()
        LOGGER.info("[PrimSkillHandler] All workers shut down")
