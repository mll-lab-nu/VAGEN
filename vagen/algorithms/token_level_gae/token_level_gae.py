"""Token-level GAE implementation."""

from vagen.algorithms._common import AdvantageInputs, advantage_estimator, register_algorithm
from vagen.algorithms._common.trajectory_algos import _compute_token_level_gae


@advantage_estimator("token_level_gae", needs_critic=True)
def compute_token_level_gae(inputs: AdvantageInputs):
    return _compute_token_level_gae(inputs)

SPEC = register_algorithm("token_level_gae", compute_token_level_gae)

__all__ = ["SPEC", "compute_token_level_gae"]
