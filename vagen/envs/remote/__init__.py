"""Remote environment implementation."""

from vagen.envs.remote.remote_env import GymImageEnvClient

RemoteEnv = GymImageEnvClient

__all__ = ["GymImageEnvClient", "RemoteEnv"]
