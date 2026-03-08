"""
TOS (Theory of Space) environment package for VAGEN.

SpatialGym is registered in env_registry.yaml as:
  SpatialGym: vagen.envs.spatial_gym.spatial_gym_env.SpatialGym
"""

from vagen.envs.spatial_gym.spatial_gym_env import SpatialGym
from vagen.envs.spatial_gym.env_config import SpatialGymConfig

__all__ = ["SpatialGym", "SpatialGymConfig"]
