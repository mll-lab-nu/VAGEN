"""Importing this package registers VAGEN's advantage estimators with verl."""

from vagen.custom_advantage import no_concat_gae  # noqa: F401  (import for side effect)
from vagen.custom_advantage import trajectory_algos  # noqa: F401  (import for side effect)
from vagen.custom_advantage.registry import (  # noqa: F401
    SENTINEL_RETURN_ESTIMATORS,
    needs_value_mask,
    register_sentinel_adv_est,
)
