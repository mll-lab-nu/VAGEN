"""Shared contracts and registry metadata for training algorithms."""

from vagen.algorithms._common.inputs import (
    AdvantageInputs,
    AdvantageOutputs,
    advantage_estimator,
)
from vagen.algorithms._common.registry import (
    CRITIC_ESTIMATORS,
    PUBLISHES_TURN_ID,
    SENTINEL_RETURN_ESTIMATORS,
    TRAJECTORY_ESTIMATORS,
    TURN_LUMPED_REWARD_ESTIMATORS,
    UNDISCOUNTED_ESTIMATORS,
    needs_critic,
    needs_value_mask,
    publishes_turn_id,
    register_sentinel_adv_est,
    register_trajectory_adv_est,
    requires_undiscounted,
    spans_rows,
    wants_turn_lumped_reward,
)
from vagen.algorithms._common.spec import (
    ALGORITHMS,
    AlgorithmSpec,
    register_algorithm,
    registered_algorithms,
    resolve_algorithm,
)

__all__ = [
    "AdvantageInputs",
    "AdvantageOutputs",
    "ALGORITHMS",
    "AlgorithmSpec",
    "CRITIC_ESTIMATORS",
    "PUBLISHES_TURN_ID",
    "SENTINEL_RETURN_ESTIMATORS",
    "TRAJECTORY_ESTIMATORS",
    "TURN_LUMPED_REWARD_ESTIMATORS",
    "UNDISCOUNTED_ESTIMATORS",
    "advantage_estimator",
    "needs_critic",
    "needs_value_mask",
    "publishes_turn_id",
    "register_algorithm",
    "register_sentinel_adv_est",
    "register_trajectory_adv_est",
    "requires_undiscounted",
    "registered_algorithms",
    "resolve_algorithm",
    "spans_rows",
    "wants_turn_lumped_reward",
]
