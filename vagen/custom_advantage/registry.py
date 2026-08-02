# All comments are in English.
"""Registration for advantage estimators that emit *sentinel* returns.

Background -- a live bug this module exists to make impossible:

    ``no_concat_gae`` / ``no_concat_gae_last`` write a real return at one anchor token
    per turn and leave every other position at ``IGNORE_RETURN`` (-100.0). The critic
    must therefore be told which positions carry supervision, via ``value_mask``.

    ``ray_trainer.py`` decided that with a hard-coded list::

        if self.config.algorithm.adv_estimator in ["no_concat_gae_last", "no_concat_gae_first"]:
            batch.batch["value_mask"] = compute_value_mask(batch)

    but the registered names are ``no_concat_gae_last`` and ``no_concat_gae`` --
    ``no_concat_gae_first`` never existed. Every no-concat script uses
    ``adv_estimator=no_concat_gae``, so ``value_mask`` was never set and the critic was
    trained to regress towards -100 on almost every token. Observed on sokoban:
    ``critic/vf_loss`` 568 -> 482 -> ... -> 1.2e-4 (converging onto the constant
    sentinel), against ~0.5 for the concat control.

The fix is structural rather than a corrected string: an estimator declares that it
emits sentinels *at the point where it registers itself*, so the two can never drift.
``tests/test_advantage_registry.py`` additionally asserts that every estimator whose
implementation mentions ``IGNORE_RETURN`` has actually declared it.
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
