"""
Simpler example: GPU round-robin allocation without multiprocessing.

This example shows the simplest way to support GPU allocation:
- Each environment is assigned to a GPU in round-robin fashion
- All environments run in the main process
- Suitable when you trust env isolation and don't need process boundaries
"""

import os
import logging
from typing import Any, Dict, List

from vagen.envs_remote import BaseGymHandler

LOGGER = logging.getLogger(__name__)


class GPURoundRobinHandler(BaseGymHandler):
    """
    Simple handler that assigns environments to GPUs in round-robin fashion.

    Example:
        GPUs = [0, 1, 2, 3]
        Session 1 -> GPU 0
        Session 2 -> GPU 1
        Session 3 -> GPU 2
        Session 4 -> GPU 3
        Session 5 -> GPU 0  (wrap around)
        ...
    """

    def __init__(
        self,
        env_class_factory,
        gpus: List[int],
        session_timeout: float = 3600.0,
    ):
        """
        Args:
            env_class_factory: Function that creates env given (env_config, gpu_id)
                              Example: lambda cfg, gpu: MyEnv({**cfg, "device": f"cuda:{gpu}"})
            gpus: List of GPU IDs to use
            session_timeout: Session timeout
        """
        super().__init__(session_timeout)
        self.env_class_factory = env_class_factory
        self.gpus = gpus
        self.next_gpu_index = 0

        LOGGER.info(f"[GPURoundRobinHandler] Initialized with GPUs: {gpus}")

    def _get_next_gpu(self) -> int:
        """Get next GPU ID in round-robin fashion."""
        gpu_id = self.gpus[self.next_gpu_index]
        self.next_gpu_index = (self.next_gpu_index + 1) % len(self.gpus)
        return gpu_id

    async def create_env(self, env_config: Dict[str, Any]):
        """Create environment on next available GPU."""
        gpu_id = self._get_next_gpu()

        LOGGER.info(f"[GPURoundRobinHandler] Creating env on GPU {gpu_id}")

        # Create environment with GPU assignment
        env = self.env_class_factory(env_config, gpu_id)

        return env


# ============================================================================
# Usage Examples
# ============================================================================

# Example 1: PyTorch environment
def create_pytorch_env(env_config, gpu_id):
    """Factory that creates env with PyTorch device."""
    import torch
    from my_envs import MyPyTorchEnv

    # Add device to config
    config = {**env_config, "device": torch.device(f"cuda:{gpu_id}")}
    return MyPyTorchEnv(config)


# Example 2: Environment with CUDA_VISIBLE_DEVICES
def create_cuda_env(env_config, gpu_id):
    """Factory that sets CUDA_VISIBLE_DEVICES before creating env."""
    # Note: This only works if env creates CUDA contexts lazily
    old_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    # Set visible GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        from my_envs import MyCUDAEnv
        env = MyCUDAEnv(env_config)
        return env
    finally:
        # Restore original setting
        if old_visible:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)


# Example 3: Simple pass-through
def create_simple_env(env_config, gpu_id):
    """Factory that just passes gpu_id in config."""
    from my_envs import MyEnv

    # Let env handle GPU assignment
    return MyEnv({**env_config, "gpu_id": gpu_id})


if __name__ == "__main__":
    from vagen.envs_remote import build_gym_service
    import uvicorn

    # Create handler
    handler = GPURoundRobinHandler(
        env_class_factory=create_pytorch_env,  # or create_cuda_env, etc.
        gpus=[0, 1, 2, 3],  # Use 4 GPUs
        session_timeout=3600.0,
    )

    # Build and run service
    app = build_gym_service(handler)
    uvicorn.run(app, host="0.0.0.0", port=8000)
