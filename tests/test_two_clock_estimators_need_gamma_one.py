"""``bi_level_gae_varlam`` runs two clocks, and they only agree at ``gamma == 1``.

Crossing one turn costs the turn-level chain a single ``gamma``. It costs the token-level
chain ``gamma ** (tokens in that turn)``. Both factors are applied to the same span of
trajectory, so unless ``gamma`` is 1 the estimator discounts it twice by two different
amounts -- and the size of the disagreement is set by how much the model wrote, which the
policy changes as it trains.

Nothing about this fails. ``gamma`` has an ordinary default, every curve keeps its shape,
and the measured relative error against an exact policy gradient is 1.06% at 0.99 and
4.9% at 0.95 -- large enough to matter and small enough to look like noise.
"""

from __future__ import annotations

import pytest

import vagen.custom_advantage  # noqa: F401  -- registers the estimators
from vagen.custom_advantage import requires_undiscounted
from vagen.trainer.mixin import VagenLogicMixin


class _Cfg(dict):
    """Attribute access over a dict, which is how the trainer reads its config."""

    __getattr__ = dict.__getitem__


class _Trainer(VagenLogicMixin):
    def __init__(self, estimator, gamma):
        self.config = _Cfg(algorithm=_Cfg(adv_estimator=estimator, gamma=gamma))


def test_the_two_clock_estimator_is_declared_and_the_one_clock_ones_are_not():
    """A declaration, not a hard-coded list somewhere else that can drift from it."""
    assert requires_undiscounted("bi_level_gae_varlam")
    # token_level_gae has one clock; turn_level_gae is a self-consistent turn MDP where
    # gamma means "per turn". Neither mixes granularities, so both are fine at gamma < 1.
    assert not requires_undiscounted("token_level_gae")
    assert not requires_undiscounted("turn_level_gae")
    assert not requires_undiscounted("trajectory_grpo")


def test_a_discounted_bi_level_run_is_refused_at_startup():
    with pytest.raises(ValueError, match=r"only defined at algorithm.gamma=1\.0"):
        _Trainer("bi_level_gae_varlam", 0.99)._vagen_check_estimator_is_undiscounted()


def test_the_refusal_says_how_wrong_it_would_have_been():
    """★ A message that only says "not allowed" gets the assertion deleted. The number is
    what makes the case: 0.99 ** 500 is 0.0066, so a long turn's bootstrap is over-weighted
    by more than a hundredfold against the turn-level chain's single 0.99."""
    with pytest.raises(ValueError) as exc:
        _Trainer("bi_level_gae_varlam", 0.99)._vagen_check_estimator_is_undiscounted()
    assert "0.99**500" in str(exc.value) or "0.006" in str(exc.value)
    assert "token_level_gae" in str(exc.value), "say what to use instead"


def test_gamma_one_passes():
    _Trainer("bi_level_gae_varlam", 1.0)._vagen_check_estimator_is_undiscounted()


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
