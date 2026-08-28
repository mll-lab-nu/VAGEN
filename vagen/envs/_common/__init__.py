"""Contracts and helpers shared by environment implementations."""

from vagen.envs._common.adapter import GymEnvAdapter
from vagen.envs._common.base import BaseEnv, Obs, Reward
from vagen.envs._common.gym_base import GymBaseEnv
from vagen.envs._common.gym_image import GymImageEnv
from vagen.envs._common.response_format import (
    ANSWER_FORMAT,
    FREE_THINK_FORMAT,
    WM_FORMAT,
    ResponseSections,
    parse_answer_sections,
    parse_free_think_sections,
    parse_wm_sections,
    split_actions,
)
from vagen.envs._common.rewards import (
    HasStateReward,
    StateRewardSpec,
    StateRewardWrapper,
    build_env,
    state_reward_names,
    state_reward_spec_of,
    supports_state_reward,
)
from vagen.envs._common.turn_limit import TurnLimit

__all__ = [
    "BaseEnv",
    "ANSWER_FORMAT",
    "FREE_THINK_FORMAT",
    "GymBaseEnv",
    "GymEnvAdapter",
    "GymImageEnv",
    "HasStateReward",
    "Obs",
    "Reward",
    "ResponseSections",
    "StateRewardSpec",
    "StateRewardWrapper",
    "TurnLimit",
    "WM_FORMAT",
    "build_env",
    "parse_answer_sections",
    "parse_free_think_sections",
    "parse_wm_sections",
    "state_reward_names",
    "state_reward_spec_of",
    "supports_state_reward",
    "split_actions",
]
