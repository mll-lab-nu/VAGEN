"""Turn-level GAE implementation."""

from vagen.algorithms._common import AdvantageInputs, advantage_estimator, register_algorithm
from vagen.algorithms._common.trajectory_algos import _compute_turn_level_gae


@advantage_estimator("turn_level_gae", needs_critic=True, sentinel_returns=True)
def compute_turn_level_gae(inputs: AdvantageInputs):
    return _compute_turn_level_gae(inputs)

SPEC = register_algorithm("turn_level_gae", compute_turn_level_gae)

__all__ = ["SPEC", "compute_turn_level_gae"]
