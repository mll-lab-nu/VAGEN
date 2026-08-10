"""VAGEN's advantage estimators, and the interface for adding one.

Write a new algorithm against :class:`AdvantageInputs` / :class:`AdvantageOutputs` and
register it with :func:`advantage_estimator`; see ``inputs.py`` for what arrives and
what verl does with what you return. Importing this package registers the built-ins.
"""

from vagen.custom_advantage.inputs import (  # noqa: F401
    AdvantageInputs,
    AdvantageOutputs,
    advantage_estimator,
)
from vagen.custom_advantage.registry import (  # noqa: F401
    CRITIC_ESTIMATORS,
    SENTINEL_RETURN_ESTIMATORS,
    TRAJECTORY_ESTIMATORS,
    TURN_LUMPED_REWARD_ESTIMATORS,
    UNDISCOUNTED_ESTIMATORS,
    needs_critic,
    needs_value_mask,
    requires_undiscounted,
    register_sentinel_adv_est,
    register_trajectory_adv_est,
    spans_rows,
    wants_turn_lumped_reward,
)
from vagen.custom_advantage import trajectory_algos  # noqa: F401,E402  (import for side effect)

__all__ = [
    "AdvantageInputs",
    "AdvantageOutputs",
    "advantage_estimator",
    "CRITIC_ESTIMATORS",
    "SENTINEL_RETURN_ESTIMATORS",
    "TRAJECTORY_ESTIMATORS",
    "TURN_LUMPED_REWARD_ESTIMATORS",
    "UNDISCOUNTED_ESTIMATORS",
    "needs_critic",
    "needs_value_mask",
    "requires_undiscounted",
    "register_sentinel_adv_est",
    "register_trajectory_adv_est",
    "spans_rows",
    "wants_turn_lumped_reward",
]
