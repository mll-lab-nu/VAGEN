"""
TOS (Theory of Space) environment package for VAGEN.

SpatialGym is registered in env_registry.yaml as:
  SpatialGym: vagen.envs.tos.tos_env.SpatialGym
"""

from vagen.envs.tos.tos_env import SpatialGym
from vagen.envs.tos.env_config import SpatialGymConfig

__all__ = ["SpatialGym", "SpatialGymConfig"]
