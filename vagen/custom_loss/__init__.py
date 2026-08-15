"""Policy losses whose unit of action is not verl's row.

Imported for the side effect of registering them, the same way
``vagen.custom_advantage`` registers advantage estimators.
"""

from vagen.custom_loss import turn_gspo  # noqa: F401  (import for side effect)
from vagen.custom_loss.turn_gspo import (  # noqa: F401
    NO_TURN,
    aggregate_seq_mean_turn_sum,
    aggregate_seq_mean_turn_sum_token_mean,
    compute_policy_loss_turn_gspo,
    compute_policy_loss_turn_ppo,
    turn_geometric_mean_log_ratio,
    turn_log_ratio,
)

#: Losses whose unit of action is a turn. Both need a ``turn_id`` column and both must be
#: imported inside the actor worker, so the trainer's startup check treats them alike --
#: naming them here rather than in the check keeps that list next to the registrations.
TURN_LEVEL_LOSSES = ("turn_gspo", "turn_ppo")

__all__ = [
    "NO_TURN",
    "TURN_LEVEL_LOSSES",
    "aggregate_seq_mean_turn_sum",
    "compute_policy_loss_turn_ppo",
    "turn_log_ratio",
    "aggregate_seq_mean_turn_sum_token_mean",
    "compute_policy_loss_turn_gspo",
    "turn_geometric_mean_log_ratio",
]
