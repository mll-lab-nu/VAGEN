"""Tests for the grounding and world-modeling rewards.

What matters is where the credit lands, not only how much of it there is. A scalar that
sums correctly but sits on the wrong tokens is exactly as invisible as a misplaced mask.

Each description score is paid on the final token of the section that earned it. An
advantage estimator that needs a coarser shape reduces it itself; the environment does
not know which estimator will consume the reward.
"""

import types

import pytest

from vagen.envs.sokoban.state_reward_spec import relations
from vagen.envs import StateRewardSpec, StateRewardWrapper
from vagen.envs._common.rewards import TAGS


class Judge:
    """Returns whatever it is told to, per call."""

    def __init__(self, *replies):
        self.replies, self.prompts, self.calls = list(replies), [], 0

    def __post_init__(self):
        pass

    async def parse_batch(self, prompts):
        self.prompts.extend(prompts)
        self.calls = getattr(self, "calls", 0) + 1
        return [self.replies.pop(0) if self.replies else None for _ in prompts]


class Env:
    def __init__(self, before, after, reward=1.0):
        self.state, self.after, self.reward = before, after, reward

    async def reset(self, seed=None):
        return {"obs_str": "start"}, {}

    async def system_prompt(self):
        return {"obs_str": "sys"}

    async def step(self, action):
        self.state = self.after
        return {"obs_str": "next"}, self.reward, False, {}

    async def close(self):
        pass


class CharTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


BOX = [{"object_id": "box", "vertical_relation": "below", "horizontal_relation": "same"}]
TGT = [{"object_id": "target", "vertical_relation": "above", "horizontal_relation": "same"}]


def _spec(**kw):
    return StateRewardSpec(
        relations=lambda env: env.state,
        judge_prompt="{content}",
        object_weights={"box": 1.0},
        examples={"state_estimation": "<observation>...</observation>",
                  "transition_prediction": "<prediction>...</prediction>"},
        axes="relations are relative to you",
        **kw,
    )


def _wrapper(env, judge, enabled=None, **kw):
    return StateRewardWrapper(
        env=env, spec=_spec(), judge=judge,
        enabled={"state_estimation": 1.0} if enabled is None else enabled, **kw,
    )


# ------------------------------------------------------------- the two switches


@pytest.mark.parametrize(
    "enabled,asked",
    [
        ({"state_estimation": 0.5}, ["observation"]),
        ({"transition_prediction": 0.5}, ["prediction"]),
        ({"state_estimation": 0.5, "transition_prediction": 0.5}, ["observation", "prediction"]),
        ({}, []),
    ],
)
def test_the_prompt_asks_for_exactly_what_is_scored(enabled, asked):
    """★ Derived, not configured separately. Asking for a section nothing scores trains
    the agent to write text for no reason; scoring one it was never asked for gives a
    silent zero every turn."""
    w = StateRewardWrapper(env=None, spec=_spec(), enabled=enabled)
    text = w.instructions()

    assert [tag for tag in TAGS.values() if f"<{tag}>" in text] == asked


def test_an_unknown_reward_name_is_rejected():
    with pytest.raises(ValueError, match="unknown state rewards"):
        StateRewardWrapper(env=None, spec=_spec(), enabled={"vibes": 1.0})


@pytest.mark.asyncio
async def test_only_the_enabled_reward_is_scored():
    """Turning one off must stop paying for it, not merely stop asking."""
    action = "<observation>A</observation><prediction>B</prediction>"
    w = _wrapper(Env(BOX, BOX, reward=0.0), Judge(BOX, BOX),
                 enabled={"state_estimation": 1.0})

    _, _, _, info = await w.step(action, [ord(c) for c in action], CharTokenizer())

    assert info["state_reward/state_estimation"] == pytest.approx(1.0)
    # Absent, not zero. A disabled reward reporting 0.0 reads as "scored nothing"
    # rather than "was not scored"; the adapter supplies the zero the metric needs.
    assert "state_reward/transition_prediction" not in info


# ------------------------------------------------------------------- placement


@pytest.mark.asyncio
async def test_each_description_is_paid_on_the_span_that_earned_it():
    """The environment preserves within-turn credit; estimators may reduce it later."""
    action = "<observation>A</observation>zz<prediction>B</prediction>"
    moved = [{"object_id": "box", "vertical_relation": "above", "horizontal_relation": "same"}]
    w = _wrapper(
        Env(before=BOX, after=moved), Judge(BOX, moved),
        enabled={"state_estimation": 1.0, "transition_prediction": 1.0},
    )

    _, vector, _, info = await w.step(action, [ord(c) for c in action], CharTokenizer())

    assert vector[action.index("A")] == pytest.approx(1.0)
    assert vector[action.index("B")] == pytest.approx(1.0)
    assert vector[-1] == pytest.approx(1.0), "the environment outcome belongs to the turn"
    assert sum(vector) == pytest.approx(3.0)
    assert info["state_reward/state_estimation"] == pytest.approx(1.0)
    assert info["state_reward/transition_prediction"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_prediction_is_scored_against_the_state_after_the_step():
    """★ Estimation describes what the agent acted from, prediction what it acted into.
    Scoring both against the same state would make one of them free."""
    action = "<prediction>A</prediction>"
    moved = [{"object_id": "box", "vertical_relation": "above", "horizontal_relation": "same"}]
    on = {"transition_prediction": 1.0}

    w = _wrapper(Env(before=BOX, after=moved), Judge(moved), enabled=on)
    _, _, _, after = await w.step(action, [ord(c) for c in action], CharTokenizer())

    w2 = _wrapper(Env(before=BOX, after=BOX), Judge(moved), enabled=on)
    _, _, _, before = await w2.step(action, [ord(c) for c in action], CharTokenizer())

    assert after["state_reward/transition_prediction"] == pytest.approx(1.0)
    assert before["state_reward/transition_prediction"] < after["state_reward/transition_prediction"]


@pytest.mark.asyncio
async def test_a_wrong_description_earns_nothing_but_does_not_go_negative():
    action = "<observation>A</observation>"
    wrong = [{"object_id": "box", "vertical_relation": "above", "horizontal_relation": "left"}]
    w = _wrapper(Env(BOX, BOX, reward=0.0), Judge(wrong))

    _, vector, _, _ = await w.step(action, [ord(c) for c in action], CharTokenizer())

    assert sum(vector) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_a_judge_outage_costs_the_process_reward_not_the_rollout():
    """★ The judge is a parser, not part of training. Losing it should cost one turn's
    shaping, not raise into the rollout."""
    action = "<observation>A</observation>"
    w = _wrapper(Env(BOX, BOX, reward=2.0), Judge(None))

    _, vector, _, _ = await w.step(action, [ord(c) for c in action], CharTokenizer())

    assert sum(vector) == pytest.approx(2.0), "the environment's own reward must survive"


@pytest.mark.asyncio
async def test_the_outcome_reward_stays_on_the_last_token():
    """The turn outcome stays at the end while description credit stays on its span."""
    action = "<observation>A</observation>zz"
    w = _wrapper(Env(BOX, BOX, reward=3.0), Judge(BOX))

    _, vector, _, _ = await w.step(action, [ord(c) for c in action], CharTokenizer())

    assert vector[action.index("A")] == pytest.approx(1.0)
    assert vector[-1] == pytest.approx(3.0)
    assert sum(vector) == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_every_enabled_section_is_required_before_any_description_pays():
    """A partial response must not farm the section it happened to include."""
    both_on = {"state_estimation": 1.0, "transition_prediction": 1.0}
    one = "<observation>A</observation>"
    both = "<observation>A</observation><prediction>B</prediction>"

    w1 = _wrapper(Env(BOX, BOX, reward=0.0), Judge(BOX), enabled=both_on)
    _, v1, _, _ = await w1.step(one, [ord(c) for c in one], CharTokenizer())

    w2 = _wrapper(Env(BOX, BOX, reward=0.0), Judge(BOX, BOX), enabled=both_on)
    _, v2, _, _ = await w2.step(both, [ord(c) for c in both], CharTokenizer())

    assert sum(v1) == pytest.approx(0.0)
    assert sum(v2) == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_both_descriptions_of_a_turn_go_out_in_one_batch():
    """★ Two round trips per turn would double the judge's latency on the critical path
    of every rollout."""
    action = "<observation>A</observation><prediction>B</prediction>"
    judge = Judge(BOX, BOX)
    w = _wrapper(Env(BOX, BOX), judge, enabled={"state_estimation": 1.0, "transition_prediction": 1.0})

    await w.step(action, [ord(c) for c in action], CharTokenizer())

    assert judge.calls == 1, f"the judge was called {judge.calls} times for one turn"


@pytest.mark.asyncio
async def test_without_tokens_the_wrapper_degrades_to_a_scalar():
    """An env used outside the token-aware loop should still work, just coarsely."""
    action = "<observation>A</observation>"
    w = _wrapper(Env(BOX, BOX, reward=1.0), Judge(BOX))

    _, reward, _, _ = await w.step(action)

    assert isinstance(reward, float) and reward == pytest.approx(2.0)


def test_one_judge_is_shared_per_endpoint():
    """★ A fresh judge per rollout makes its concurrency limit per rollout: with
    hundreds in flight the endpoint sees hundreds of times the intended load, and the
    timeouts read as the process reward quietly going to zero."""
    from vagen.envs._common.rewards import shared_judge

    a = shared_judge("http://x/v1", "m")
    b = shared_judge("http://x/v1", "m")
    c = shared_judge("http://y/v1", "m")

    assert a is b and a is not c


def test_sokoban_relations_are_relative_to_the_player():
    room = [[0, 0, 0], [0, 5, 0], [0, 4, 0]]      # player at (1,1), box below it
    fixed = [[0, 0, 0], [0, 0, 0], [0, 2, 0]]      # target at (2,1)

    import numpy as np

    env = types.SimpleNamespace(env=types.SimpleNamespace(room_state=np.array(room), room_fixed=np.array(fixed)))
    items = relations(env)

    assert {"object_id": "box", "vertical_relation": "below", "horizontal_relation": "same"} in items
    assert {"object_id": "target", "vertical_relation": "below", "horizontal_relation": "same"} in items


# ------------------------------------------------------- hallucination must cost


def _box(v="below", h="same"):
    return {"object_id": "box", "vertical_relation": v, "horizontal_relation": h}


def _target(v="above", h="same"):
    return {"object_id": "target", "vertical_relation": v, "horizontal_relation": h}


WEIGHTS = {"target": 0.5, "box": 0.5}


def test_describing_extra_items_of_a_real_type_costs_precision():
    """★ Inventing things has to be worse than not inventing them, or the cheapest way
    to raise recall is to list every relation the grid could contain."""
    from vagen.envs._common.rewards import grouped_f1

    gold = [_box(), _target()]
    exact = grouped_f1(gold, gold, WEIGHTS)
    one_extra = grouped_f1(gold + [_box("above", "left")], gold, WEIGHTS)
    two_extra = grouped_f1(gold + [_box("above", "left"), _box("same", "right")], gold, WEIGHTS)

    assert exact == pytest.approx(1.0)
    assert two_extra < one_extra < exact, f"{two_extra} < {one_extra} < {exact}"


def test_inventing_a_type_that_is_not_there_costs_more_than_silence():
    """★ The 'absent from both' rule only skips a type when neither side mentions it.
    Once the description mentions it, it counts -- with nothing to match against."""
    from vagen.envs._common.rewards import grouped_f1

    gold = [_box()]                                   # no targets in this scene

    silent_about_targets = grouped_f1([_box()], gold, WEIGHTS)
    invented_a_target = grouped_f1([_box(), _target()], gold, WEIGHTS)

    assert silent_about_targets == pytest.approx(1.0)
    assert invented_a_target == pytest.approx(0.5)
    assert invented_a_target < silent_about_targets
