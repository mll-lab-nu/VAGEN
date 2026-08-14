"""An estimator that runs two clocks is only correct at ``gamma == 1``, and says so.

Crossing one turn costs a turn-level chain a single ``gamma``. It costs a token-level chain
``gamma ** (tokens in that turn)``. An estimator that applies both to the same span of
trajectory discounts it twice by two different amounts -- and the size of the disagreement
is set by how much the model wrote, which the policy changes as it trains.

Nothing about this fails. ``gamma`` has an ordinary default, every curve keeps its shape,
and the measured relative error against an exact policy gradient is 1.06% at 0.99 and 4.9%
at 0.95 -- large enough to matter and small enough to look like noise. So the estimator
declares ``undiscounted=True`` at registration and the trainer refuses the run at startup.

★ No estimator in the tree declares it today -- ``removed_estimator_gae_varlam``, which did, has been
removed, and ``removed_estimator_gae`` takes its two gammas separately so it is well-defined away
from 1.0. The flag stays because it is part of the registration contract a custom estimator
uses, and an extension point with no test is one that breaks the first time someone reaches
for it. Hence a throwaway estimator registered here rather than a real one borrowed.
"""

from __future__ import annotations

import pytest

import vagen.custom_advantage  # noqa: F401  -- registers the estimators
from vagen.custom_advantage import requires_undiscounted
from vagen.custom_advantage.registry import (
    CRITIC_ESTIMATORS,
    PUBLISHES_TURN_ID,
    TRAJECTORY_ESTIMATORS,
    UNDISCOUNTED_ESTIMATORS,
    register_trajectory_adv_est,
)
from vagen.trainer.mixin import VagenLogicMixin

TWO_CLOCK = "_test_two_clock_estimator"


@pytest.fixture(autouse=True)
def _register_a_two_clock_estimator():
    """Registered per test and removed after, so nothing else in the suite sees it.

    Safe against the parametrized contract tests either way: those build their id lists
    from TRAJECTORY_ESTIMATORS at collection time, before any fixture runs.
    """
    @register_trajectory_adv_est(TWO_CLOCK, needs_critic=True, undiscounted=True)
    def _estimator(inputs):  # pragma: no cover -- never invoked; only its flags are read
        raise AssertionError("registered for its declaration, not to be run")

    yield

    for registry in (TRAJECTORY_ESTIMATORS, PUBLISHES_TURN_ID, CRITIC_ESTIMATORS,
                     UNDISCOUNTED_ESTIMATORS):
        registry.discard(TWO_CLOCK)
    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
    ADV_ESTIMATOR_REGISTRY.pop(TWO_CLOCK, None)


class _Cfg(dict):
    """Attribute access over a dict, which is how the trainer reads its config."""

    __getattr__ = dict.__getitem__


class _Trainer(VagenLogicMixin):
    def __init__(self, estimator, gamma):
        self.config = _Cfg(algorithm=_Cfg(adv_estimator=estimator, gamma=gamma))


def test_the_flag_is_a_declaration_and_not_a_list_that_can_drift():
    assert requires_undiscounted(TWO_CLOCK)
    # token_level_gae has one clock; turn_level_gae is a self-consistent turn MDP where
    # mixes granularities on one gamma, so all are fine at gamma < 1.
    assert not requires_undiscounted("token_level_gae")
    assert not requires_undiscounted("turn_level_gae")
    assert not requires_undiscounted("removed_estimator_gae")
    assert not requires_undiscounted("trajectory_grpo")


def test_a_discounted_two_clock_run_is_refused_at_startup():
    with pytest.raises(ValueError, match=r"only defined at algorithm.gamma=1\.0"):
        _Trainer(TWO_CLOCK, 0.99)._vagen_check_estimator_is_undiscounted()


def test_the_refusal_says_how_wrong_it_would_have_been():
    """★ A message that only says "not allowed" gets the assertion deleted. The number is
    what makes the case: 0.99 ** 500 is 0.0066, so a long turn's bootstrap is over-weighted
    by more than a hundredfold against the turn-level chain's single 0.99."""
    with pytest.raises(ValueError) as exc:
        _Trainer(TWO_CLOCK, 0.99)._vagen_check_estimator_is_undiscounted()
    assert "0.99**500" in str(exc.value) or "0.006" in str(exc.value)
    assert "token_level_gae" in str(exc.value), "say what to use instead"


def test_gamma_one_passes():
    _Trainer(TWO_CLOCK, 1.0)._vagen_check_estimator_is_undiscounted()


def test_a_one_clock_estimator_may_discount():
    """The check must not become "gamma is always 1"; token_level_gae is a plain
    token MDP and discounting it is a legitimate thing to want."""
    _Trainer("token_level_gae", 0.99)._vagen_check_estimator_is_undiscounted()
    _Trainer("turn_level_gae", 0.95)._vagen_check_estimator_is_undiscounted()


def test_the_check_runs_at_startup_and_not_only_when_called():
    """It is only worth anything if `_vagen_init` invokes it -- an unreferenced private
    method passes its own unit test forever."""
    import inspect

    src = inspect.getsource(VagenLogicMixin._vagen_init)
    assert "_vagen_check_estimator_is_undiscounted()" in src
