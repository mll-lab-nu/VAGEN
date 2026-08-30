"""Reward placement belongs to the producer.

The environment always pays a description on the span that earned it. The old
``placement`` switch is intentionally gone because it coupled rollout construction to
the advantage estimator and allowed two configs to disagree silently.
"""

from __future__ import annotations

import pytest

from vagen.envs import StateRewardWrapper
from test_state_reward import BOX, CharTokenizer, Env, Judge, _spec, _wm


def _wrapper():
    return StateRewardWrapper(
        env=Env(BOX, BOX, reward=3.0),
        spec=_spec(),
        judge=Judge(BOX),
        enabled={"state_estimation": 1.0},
    )


def test_the_old_placement_knob_is_hard_deleted():
    with pytest.raises(TypeError, match="placement"):
        StateRewardWrapper(
            env=Env(BOX, BOX), spec=_spec(), enabled={"state_estimation": 1.0},
            placement="turn_end",
        )


@pytest.mark.asyncio
async def test_description_credit_is_per_span_and_outcome_credit_is_per_turn():
    action = _wm()
    ids = [ord(c) for c in action]
    _, rewards, _, info = await _wrapper().step(action, ids, CharTokenizer())

    assert rewards[action.index("A")] == pytest.approx(1.0)
    assert rewards[-1] == pytest.approx(3.0)
    assert sum(rewards) == pytest.approx(4.0)
    assert info["state_reward/state_estimation"] == pytest.approx(1.0)
