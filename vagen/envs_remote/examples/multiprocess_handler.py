"""
Example: Multi-process handler with GPU allocation.

This example shows how to extend BaseGymHandler to support:
- Multiple worker processes
- GPU allocation
- Process-based environment isolation
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple
from multiprocessing import Process, Queue, Event
import queue

from PIL import Image
from vagen.envs.gym_image_env import GymImageEnv
from vagen.envs_remote import BaseGymHandler

LOGGER = logging.getLogger(__name__)


# ============================================================================
# Worker Process
# ============================================================================
def worker_process(worker_id: int, gpu_id: int, task_queue: Queue, result_queue: Queue, shutdown_event: Event):
    """
    Worker process that runs environments.

    Each worker can:
    - Run on a specific GPU
    - Handle multiple environments (one at a time)
    - Isolate environments in separate processes
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    LOGGER.info(f"[Worker-{worker_id}] Started on GPU {gpu_id}")

    # Each worker maintains a dict of active environments
    envs: Dict[str, Any] = {}

    try:
        while not shutdown_event.is_set():
            try:
                # Non-blocking get with timeout
                task = task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            task_type = task["type"]
            session_id = task["session_id"]

            try:
                if task_type == "create":
                    # Create environment in this process
                    env_config = task["env_config"]
                    env_class = task["env_class"]

                    # Import and create environment
                    module_path, class_name = env_class.rsplit(".", 1)
                    import importlib
                    module = importlib.import_module(module_path)
                    env_cls = getattr(module, class_name)

                    env = env_cls(env_config)
                    envs[session_id] = env

                    result_queue.put({"session_id": session_id, "status": "created"})
                    LOGGER.info(f"[Worker-{worker_id}] Created env for session {session_id}")

                elif task_type == "call":
                    # Call method on environment
                    method = task["method"]
                    kwargs = task["kwargs"]

                    env = envs.get(session_id)
                    if not env:
                        result_queue.put({
                            "session_id": session_id,
                            "status": "error",
                            "error": f"Session {session_id} not found"
                        })
                        continue

                    # Call method
                    result = asyncio.run(getattr(env, method)(**kwargs))

                    result_queue.put({
                        "session_id": session_id,
                        "status": "success",
                        "result": result
                    })

                elif task_type == "close":
                    # Close and remove environment
                    env = envs.pop(session_id, None)
                    if env:
                        asyncio.run(env.close())
                        LOGGER.info(f"[Worker-{worker_id}] Closed env for session {session_id}")

                    result_queue.put({"session_id": session_id, "status": "closed"})

            except Exception as e:
                LOGGER.error(f"[Worker-{worker_id}] Error processing task: {e}")
                result_queue.put({
                    "session_id": session_id,
                    "status": "error",
                    "error": str(e)
                })

    finally:
        # Cleanup: close all environments
        for session_id, env in envs.items():
            try:
                asyncio.run(env.close())
            except Exception as e:
                LOGGER.error(f"[Worker-{worker_id}] Error closing {session_id}: {e}")

        LOGGER.info(f"[Worker-{worker_id}] Shutdown")


# ============================================================================
# Proxy Environment (runs in main process, delegates to worker)
# ============================================================================
class ProcessEnvProxy(GymImageEnv):
    """
    Proxy environment that delegates all calls to a worker process.

    This object lives in the main process but all actual env operations
    happen in a worker process.
    """

    def __init__(
        self,
        session_id: str,
        worker_id: int,
        gpu_id: int,
        task_queue: Queue,
        result_queue: Queue,
        env_config: Dict[str, Any],
    ):
        super().__init__(env_config)
        self.session_id = session_id
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.task_queue = task_queue
        self.result_queue = result_queue

    async def _call_worker(self, method: str, **kwargs) -> Any:
        """Send task to worker and wait for result."""
        self.task_queue.put({
            "type": "call",
            "session_id": self.session_id,
            "method": method,
            "kwargs": kwargs,
        })

        # Wait for result (with timeout)
        timeout = 120.0
        start = asyncio.get_event_loop().time()

        while True:
            try:
                result = self.result_queue.get(timeout=0.1)
                if result["session_id"] == self.session_id:
                    if result["status"] == "error":
                        raise RuntimeError(result["error"])
                    return result["result"]
            except queue.Empty:
                elapsed = asyncio.get_event_loop().time() - start
                if elapsed > timeout:
                    raise TimeoutError(f"Worker call timed out after {timeout}s")
                await asyncio.sleep(0.01)

    async def system_prompt(self) -> Dict[str, Any]:
        return await self._call_worker("system_prompt")

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return await self._call_worker("reset", seed=seed)

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        return await self._call_worker("step", action_str=action_str)

    async def close(self) -> None:
        self.task_queue.put({
            "type": "close",
            "session_id": self.session_id,
        })
        # Wait for confirmation
        while True:
            try:
                result = self.result_queue.get(timeout=0.1)
                if result["session_id"] == self.session_id:
                    break
            except queue.Empty:
                await asyncio.sleep(0.01)


# ============================================================================
# Multi-Process Handler with GPU Allocation
# ============================================================================
class MultiProcessGPUHandler(BaseGymHandler):
    """
    Handler that manages multiple worker processes with GPU allocation.

    Features:
    - Each worker process runs on a specific GPU
    - Load balancing across workers
    - Process isolation for environments
    """

    def __init__(
        self,
        env_class: str,
        num_workers: int = 4,
        gpus: Optional[list] = None,
        session_timeout: float = 3600.0,
    ):
        """
        Args:
            env_class: Full path to env class (e.g., "vagen.envs.my_env.MyEnv")
            num_workers: Number of worker processes
            gpus: List of GPU IDs to use (e.g., [0, 1, 2, 3])
                  If None, uses GPUs 0..num_workers-1
            session_timeout: Session timeout in seconds
        """
        super().__init__(session_timeout)

        self.env_class = env_class
        self.num_workers = num_workers
        self.gpus = gpus or list(range(num_workers))

        if len(self.gpus) != num_workers:
            raise ValueError(f"Number of GPUs ({len(self.gpus)}) must match num_workers ({num_workers})")

        # Shared queues for communication
        self.task_queues = [Queue() for _ in range(num_workers)]
        self.result_queue = Queue()  # Single result queue

        # Worker processes
        self.shutdown_events = [Event() for _ in range(num_workers)]
        self.workers = []

        # Start workers
        for i in range(num_workers):
            worker = Process(
                target=worker_process,
                args=(i, self.gpus[i], self.task_queues[i], self.result_queue, self.shutdown_events[i]),
                daemon=True,
            )
            worker.start()
            self.workers.append(worker)

        # Load balancing: track number of envs per worker
        self.worker_loads = [0] * num_workers

        LOGGER.info(f"[MultiProcessGPUHandler] Started {num_workers} workers on GPUs {self.gpus}")

    def _select_worker(self) -> int:
        """Select worker with least load (round-robin tiebreaker)."""
        min_load = min(self.worker_loads)
        candidates = [i for i, load in enumerate(self.worker_loads) if load == min_load]
        return candidates[0]

    async def create_env(self, env_config: Dict[str, Any]) -> GymImageEnv:
        """
        Create environment in a worker process.

        This returns a proxy object that delegates to the worker.
        """
        # Select worker
        worker_id = self._select_worker()
        gpu_id = self.gpus[worker_id]

        # Generate session ID (will be set by BaseGymHandler.connect)
        # We'll use a temporary ID for now
        import uuid
        temp_session_id = uuid.uuid4().hex

        # Send create task to worker
        self.task_queues[worker_id].put({
            "type": "create",
            "session_id": temp_session_id,
            "env_config": env_config,
            "env_class": self.env_class,
        })

        # Wait for confirmation
        timeout = 30.0
        start = asyncio.get_event_loop().time()

        while True:
            try:
                result = self.result_queue.get(timeout=0.1)
                if result["session_id"] == temp_session_id:
                    if result["status"] != "created":
                        raise RuntimeError(f"Failed to create env: {result}")
                    break
            except queue.Empty:
                elapsed = asyncio.get_event_loop().time() - start
                if elapsed > timeout:
                    raise TimeoutError("Environment creation timed out")
                await asyncio.sleep(0.01)

        # Update worker load
        self.worker_loads[worker_id] += 1

        LOGGER.info(
            f"[MultiProcessGPUHandler] Created env on worker {worker_id} (GPU {gpu_id}), "
            f"load={self.worker_loads[worker_id]}"
        )

        # Return proxy
        return ProcessEnvProxy(
            session_id=temp_session_id,
            worker_id=worker_id,
            gpu_id=gpu_id,
            task_queue=self.task_queues[worker_id],
            result_queue=self.result_queue,
            env_config=env_config,
        )

    async def aclose(self):
        """Shutdown all workers."""
        LOGGER.info("[MultiProcessGPUHandler] Shutting down workers...")

        # Signal shutdown
        for event in self.shutdown_events:
            event.set()

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                LOGGER.warning(f"[MultiProcessGPUHandler] Worker {worker.pid} did not terminate, forcing...")
                worker.terminate()

        await super().aclose()
        LOGGER.info("[MultiProcessGPUHandler] All workers shutdown")


# ============================================================================
# Usage Example
# ============================================================================
if __name__ == "__main__":
    from vagen.envs_remote import build_gym_service
    import uvicorn

    # Create handler with 4 workers on GPUs 0-3
    handler = MultiProcessGPUHandler(
        env_class="vagen.envs.my_env.MyEnv",  # Replace with your env
        num_workers=4,
        gpus=[0, 1, 2, 3],
        session_timeout=3600.0,
    )

    # Build service
    app = build_gym_service(handler)

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
