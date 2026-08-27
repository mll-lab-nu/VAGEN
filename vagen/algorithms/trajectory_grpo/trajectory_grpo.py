"""Trajectory-level GRPO implementation."""

from vagen.algorithms._common import AdvantageInputs, advantage_estimator, register_algorithm
from vagen.algorithms._common.trajectory_algos import _compute_trajectory_grpo


@advantage_estimator("trajectory_grpo", publishes_turn_id=False)
def compute_trajectory_grpo(inputs: AdvantageInputs):
    return _compute_trajectory_grpo(inputs)

SPEC = register_algorithm("trajectory_grpo", compute_trajectory_grpo)

__all__ = ["SPEC", "compute_trajectory_grpo"]
