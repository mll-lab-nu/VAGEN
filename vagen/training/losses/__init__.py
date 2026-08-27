"""Policy-loss extension facade."""

from vagen.training.losses import turn_gspo as turn_gspo
from vagen.training.losses.turn_gspo import (
    NO_TURN,
    aggregate_seq_mean_turn_sum,
    aggregate_seq_mean_turn_sum_token_mean,
    compute_policy_loss_turn_gspo,
    compute_policy_loss_turn_ppo,
    turn_geometric_mean_log_ratio,
    turn_log_ratio,
)

TURN_LEVEL_LOSSES = ("turn_gspo", "turn_ppo")

__all__ = [
    "NO_TURN",
    "TURN_LEVEL_LOSSES",
    "aggregate_seq_mean_turn_sum",
    "aggregate_seq_mean_turn_sum_token_mean",
    "compute_policy_loss_turn_gspo",
    "compute_policy_loss_turn_ppo",
    "turn_geometric_mean_log_ratio",
    "turn_log_ratio",
]
