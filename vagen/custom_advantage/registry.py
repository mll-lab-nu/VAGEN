"""Registration for advantage estimators that emit *sentinel* returns.

Turn-level estimators such as ``no_concat_gae`` write a real return at one anchor token
per turn and leave every other position at ``IGNORE_RETURN`` (-100.0). The critic must
therefore be told which positions carry supervision, via ``value_mask``; without it, it
is trained to regress towards the sentinel almost everywhere.

Deciding that from a hard-coded list of estimator names is what this module exists to
avoid: the list and the registered names drift apart silently, and the symptom -- a
critic fitting a constant -- shows up as a *falling* value loss and a healthy-looking
explained variance, so nothing obvious fails.

Instead an estimator declares that it emits sentinels at the point where it registers
itself, so the two cannot disagree. ``tests/test_advantage_registry.py`` additionally
asserts that every estimator whose implementation mentions ``IGNORE_RETURN`` has
actually declared it.
"""

from __future__ import annotations

from typing import Callable

from verl.trainer.ppo.core_algos import register_adv_est

# Estimator names whose `returns` contain IGNORE_RETURN at unsupervised positions.
SENTINEL_RETURN_ESTIMATORS: set[str] = set()


def register_sentinel_adv_est(name: str) -> Callable:
    """Register an advantage estimator that writes sentinel returns.

    Same contract as verl's ``register_adv_est``, and additionally records the name so
    the trainer can decide to compute ``value_mask`` without hard-coding anything.
    """

    def decorator(fn):
        SENTINEL_RETURN_ESTIMATORS.add(name)
        return register_adv_est(name)(fn)

    return decorator


def needs_value_mask(adv_estimator) -> bool:
    """Whether this estimator's returns require a ``value_mask``.

    Accepts a str or verl's ``AdvantageEstimator`` enum (whose members are str-valued).
    """
    return str(getattr(adv_estimator, "value", adv_estimator)) in SENTINEL_RETURN_ESTIMATORS
