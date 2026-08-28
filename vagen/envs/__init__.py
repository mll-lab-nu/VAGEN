"""Environment facade and registry.

Consumers should import contracts and builders from this module. Concrete environment
packages remain private implementation details except for their executable service
entry points.
"""

from vagen.envs._common import (
    BaseEnv,
    GymBaseEnv,
    GymEnvAdapter,
    GymImageEnv,
    HasStateReward,
    Obs,
    Reward,
    StateRewardSpec,
    StateRewardWrapper,
    TurnLimit,
    build_env,
    state_reward_names,
    state_reward_spec_of,
    supports_state_reward,
)
from vagen.envs.registry import get_env_cls, list_envs, register_env

__all__ = [
    "BaseEnv",
    "GymBaseEnv",
    "GymEnvAdapter",
    "GymImageEnv",
    "HasStateReward",
    "Obs",
    "Reward",
    "StateRewardSpec",
    "StateRewardWrapper",
    "TurnLimit",
    "build_env",
    "get_env_cls",
    "list_envs",
    "register_env",
    "state_reward_names",
    "state_reward_spec_of",
    "supports_state_reward",
]
