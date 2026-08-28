"""Reward contracts and helpers shared by environment implementations."""

from vagen.envs._common.rewards.judge import NullJudge, StructuredJudge, parse_items, shared_judge
from vagen.envs._common.rewards.factory import (
    HasStateReward,
    build_env,
    state_reward_names,
    state_reward_spec_of,
    supports_state_reward,
)
from vagen.envs._common.rewards.spans import spread, tagged_span, token_offsets, tokens_covering
from vagen.envs._common.rewards.spatial import exact_relation_match, f1, grouped_f1
from vagen.envs._common.rewards.state import (
    DEFAULT_SCORE_BASE,
    TAGS,
    StateRewardSpec,
    StateRewardWrapper,
)

__all__ = [
    "DEFAULT_SCORE_BASE",
    "HasStateReward",
    "NullJudge",
    "StateRewardSpec",
    "StateRewardWrapper",
    "StructuredJudge",
    "TAGS",
    "build_env",
    "exact_relation_match",
    "f1",
    "grouped_f1",
    "parse_items",
    "shared_judge",
    "spread",
    "tagged_span",
    "token_offsets",
    "tokens_covering",
    "state_reward_names",
    "state_reward_spec_of",
    "supports_state_reward",
]
