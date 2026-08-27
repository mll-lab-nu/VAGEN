"""Default episode-level GAE implementation."""

from vagen.algorithms._common import AdvantageInputs, advantage_estimator, register_algorithm
from vagen.algorithms._common.trajectory_algos import _compute_default_gae


@advantage_estimator("default_gae", needs_critic=True)
def compute_default_gae(inputs: AdvantageInputs):
    return _compute_default_gae(inputs)

SPEC = register_algorithm("default_gae", compute_default_gae)

__all__ = ["SPEC", "compute_default_gae"]
