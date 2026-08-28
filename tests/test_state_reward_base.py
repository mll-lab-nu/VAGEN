"""The state reward must pay for looking, not for speaking.

`grouped_f1` gives 0.5 for getting one of two axes right, and each axis is a 3-way
choice, so naming *any* relation already scores about a third. Measured over 300 real
Sokoban starts: uniform random 0.34, and the best constant answer -- "same, same", which
looks at nothing -- 0.40. The reward's usable range was 0.39 to 1.00, and the model's
observed 0.45-0.52 sat barely above the floor. Descriptions that were visibly wrong
scored well, which is exactly the complaint that started this.

`score_base` subtracts that floor and rescales, so the *marginal* reward for actually
perceiving something is most of the range instead of a tenth of it. `spatial.py` is left
alone -- it is a faithful port of the legacy bipartite F1 and stays that way.
"""

from __future__ import annotations

import pytest

from vagen.envs import StateRewardWrapper
from vagen.envs._common.rewards import DEFAULT_SCORE_BASE
from test_state_reward import BOX, CharTokenizer, Env, Judge, _spec


def _w(base=DEFAULT_SCORE_BASE, **kw):
    return StateRewardWrapper(env=None, spec=_spec(), enabled={"state_estimation": 1.0},
                              score_base=base, **kw)


@pytest.mark.parametrize(
    "f1,base,expected",
    [
        (1.0, 0.334, 1.0),        # a perfect description still earns the whole weight...
        (0.334, 0.334, 0.0),      # ...chance earns nothing...
        (0.0, 0.334, 0.0),        # ...and worse than chance is worth nothing, not a debt.
        (0.667, 0.334, 0.5),      # halfway between chance and perfect -> half the weight
        (0.5, 0.0, 0.5),          # base 0 is the legacy reward, untouched
        (1.0, 0.0, 1.0),
    ],
)
def test_the_base_is_subtracted_and_rescaled(f1, base, expected):
    assert _w(base)._above_base(f1) == pytest.approx(expected, abs=1e-6)


def test_a_perfect_description_still_earns_the_configured_reward():
    """Rescaled, not merely shifted, so the yaml's per-turn number stays truthful."""
    for base in (0.0, 0.334, 0.5, 0.9):
        assert _w(base)._above_base(1.0) == pytest.approx(1.0), base


def test_it_never_goes_negative():
    """A description worse than chance is worth nothing; it is not a debt against the
    task reward."""
    for f1 in (0.0, 0.1, 0.333):
        assert _w(0.334)._above_base(f1) >= 0.0


def test_base_zero_is_the_legacy_reward():
    """The escape hatch has to be exact, or 'reproduce the published runs' is not
    something anyone can do."""
    w = _w(0.0)
    for f1 in (0.0, 0.137, 0.5, 0.913, 1.0):
        assert w._above_base(f1) == f1


@pytest.mark.asyncio
async def test_it_reaches_the_paid_score_end_to_end():
    """The base has to be applied where the reward is computed, not just be a method
    nothing calls."""
    action = "<observation>A</observation>"
    ids = [ord(c) for c in action]

    full = StateRewardWrapper(env=Env(BOX, BOX, reward=0.0), spec=_spec(), judge=Judge(BOX),
                              enabled={"state_estimation": 1.0}, score_base=0.0)
    based = StateRewardWrapper(env=Env(BOX, BOX, reward=0.0), spec=_spec(), judge=Judge(BOX),
                               enabled={"state_estimation": 1.0}, score_base=0.5)
    # A *perfect* description is unchanged by the base -- that is the rescale working.
    _, _, _, i_full = await full.step(action, ids, CharTokenizer())
    _, _, _, i_based = await based.step(action, ids, CharTokenizer())
    assert i_full["state_reward/state_estimation"] == pytest.approx(1.0)
    assert i_based["state_reward/state_estimation"] == pytest.approx(1.0)

    # A half-right one is not.
    half = [{"object_id": "box", "vertical_relation": "below", "horizontal_relation": "WRONG"}]
    a = StateRewardWrapper(env=Env(BOX, BOX, reward=0.0), spec=_spec(), judge=Judge(half),
                           enabled={"state_estimation": 1.0}, score_base=0.0)
    b = StateRewardWrapper(env=Env(BOX, BOX, reward=0.0), spec=_spec(), judge=Judge(half),
                           enabled={"state_estimation": 1.0}, score_base=0.5)
    _, _, _, ia = await a.step(action, ids, CharTokenizer())
    _, _, _, ib = await b.step(action, ids, CharTokenizer())
    assert ia["state_reward/state_estimation"] == pytest.approx(0.5)
    assert ib["state_reward/state_estimation"] == pytest.approx(0.0), (
        "half-right is exactly chance, so with base=0.5 it must pay nothing"
    )


def test_the_default_is_the_measured_random_floor():
    """0.334 is not a round number chosen for looks -- it is the measured mean
    `grouped_f1` of a uniformly random describer on 300 real Sokoban starts. If someone
    changes it, they should have re-measured."""
    assert DEFAULT_SCORE_BASE == 0.334
