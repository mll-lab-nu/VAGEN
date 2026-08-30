"""Training-algorithm facade and registry."""

from vagen.algorithms._common import *  # noqa: F401,F403
from vagen.algorithms.default_gae import SPEC as DEFAULT_GAE
from vagen.algorithms.token_level_gae import SPEC as TOKEN_LEVEL_GAE
from vagen.algorithms.trajectory_grpo import SPEC as TRAJECTORY_GRPO
from vagen.algorithms.turn_level_gae import SPEC as TURN_LEVEL_GAE

__all__ = [
    "ALGORITHMS",
    "AlgorithmSpec",
    "DEFAULT_GAE",
    "TOKEN_LEVEL_GAE",
    "TRAJECTORY_GRPO",
    "TURN_LEVEL_GAE",
    "AdvantageInputs",
    "AdvantageOutputs",
    "CRITIC_ESTIMATORS",
    "PUBLISHES_TURN_ID",
    "SENTINEL_RETURN_ESTIMATORS",
    "TRAJECTORY_ESTIMATORS",
    "UNDISCOUNTED_ESTIMATORS",
    "advantage_estimator",
    "needs_critic",
    "needs_value_mask",
    "publishes_turn_id",
    "register_algorithm",
    "register_sentinel_adv_est",
    "register_trajectory_adv_est",
    "registered_algorithms",
    "requires_undiscounted",
    "resolve_algorithm",
    "spans_rows",
]
