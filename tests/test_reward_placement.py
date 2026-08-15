"""Where a turn's scores are paid, and why the advantage estimator decides it.

Placement and estimator are one choice made in two files, which is exactly the shape of
mistake that does not raise and does not show in a curve:

* an estimator with one reward slot per turn (``bi_level_gae`` reads a turn's reward
  only at its last token) credits a mid-turn score twice -- once through the inner token
  chain, once through the outer turn chain;
* the per-token estimators pay variance for a lumped score, because ``V`` then has to
  remember it for the rest of the turn.

So ``placement: auto`` resolves from ``algorithm.adv_estimator``, and these tests pin the
resolution rather than either mode's arithmetic alone.
"""

from __future__ import annotations

import pytest

import vagen.custom_advantage  # noqa: F401  registers the estimators the resolver reads
from vagen.agent_loop.gym_loop import resolve_reward_placement
from vagen.rewards.state_reward import StateRewardWrapper
from test_state_reward import BOX, CharTokenizer, Env, Judge, _spec


class _Cfg(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def _cfg(estimator):
    return _Cfg(algorithm=_Cfg(adv_estimator=estimator))


# ------------------------------------------------------------------- the resolution


@pytest.mark.parametrize(
    "estimator,expected",
    [
        # The one estimator whose outer chain has a single reward slot per turn.
        ("bi_level_gae", "turn_end"),
        # Everything else supervises per token and wants the score where it was earned.
        ("token_level_gae", "per_span"),
        ("turn_level_gae", "per_span"),
        ("default_gae", "per_span"),
        ("trajectory_grpo", "per_span"),
        ("gae", "per_span"),
    ],
)
def test_auto_resolves_from_the_estimator(estimator, expected):
    assert resolve_reward_placement(_cfg(estimator), "auto") == expected


def test_a_typo_in_the_estimator_name_does_not_silently_lump():
    """★ Near-miss names must fall to the majority default rather than being
    fuzzy-matched. Reading ``bi_level_gae_papper`` as the paper estimator would silently
    change where every reward in the run is paid."""
    assert resolve_reward_placement(_cfg("bi_level_gae_papper"), "auto") == "per_span"
    assert resolve_reward_placement(_cfg(""), "auto") == "per_span"


def test_an_explicit_setting_overrides_auto():
    assert resolve_reward_placement(_cfg("token_level_gae"), "turn_end") == "turn_end"
    assert resolve_reward_placement(_cfg("bi_level_gae"), "per_span") == "per_span"


def test_a_missing_algorithm_block_does_not_explode():
    """The reward wrapper is built inside the rollout worker, whose config need not carry
    the algorithm block at all."""
    assert resolve_reward_placement(_Cfg(), "auto") == "per_span"


def test_an_unknown_placement_is_refused_at_construction():
    with pytest.raises(ValueError, match="unknown placement"):
        StateRewardWrapper(env=None, spec=_spec(), enabled={"state_estimation": 1.0},
                           placement="wherever")


# ---------------------------------------------------------------- the two behaviours


def _run(placement, action="<observation>A</observation>zz"):
    w = StateRewardWrapper(
        env=Env(BOX, BOX, reward=3.0), spec=_spec(), judge=Judge(BOX),
        enabled={"state_estimation": 1.0}, format_reward=0.0, placement=placement,
    )
    return w


@pytest.mark.asyncio
async def test_per_span_pays_the_section_and_turn_end_pays_the_turn():
    """The two modes differ where it matters and nowhere else."""
    action = "<observation>A</observation>zz"
    ids = [ord(c) for c in action]

    _, span_v, _, _ = await _run("per_span").step(action, ids, CharTokenizer())
    _, turn_v, _, _ = await _run("turn_end").step(action, ids, CharTokenizer())

    # per_span: the description's 1.0 sits on the 'A', the outcome's 3.0 at the end.
    assert span_v[action.index("A")] == pytest.approx(1.0)
    assert span_v[-1] == pytest.approx(3.0)
    # turn_end: all 4.0 on the last token, nothing before it.
    assert turn_v[-1] == pytest.approx(4.0)
    assert sum(turn_v[:-1]) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_the_turn_total_is_the_same_either_way():
    """★ The invariant that makes the switch safe to flip. Placement moves reward, it
    never creates or destroys any -- so no length-hacking channel opens on one side, and a
    run's total auxiliary reward is comparable across the two."""
    action = "<observation>A</observation>zz"
    ids = [ord(c) for c in action]

    _, span_v, _, _ = await _run("per_span").step(action, ids, CharTokenizer())
    _, turn_v, _, _ = await _run("turn_end").step(action, ids, CharTokenizer())

    assert sum(span_v) == pytest.approx(sum(turn_v)) == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_the_breakdown_reaches_info_in_both_modes():
    """Which half of the reasoning was right must not depend on where it was paid."""
    action = "<observation>A</observation>zz"
    ids = [ord(c) for c in action]
    for placement in ("per_span", "turn_end"):
        _, _, _, info = await _run(placement).step(action, ids, CharTokenizer())
        assert info["state_reward/state_estimation"] == pytest.approx(1.0), placement
