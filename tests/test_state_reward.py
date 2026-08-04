"""Tests for the grounding and world-modeling rewards.

What matters is where the credit lands, not only how much of it there is. A scalar that
sums correctly but sits on the wrong tokens is exactly as invisible as a misplaced mask.
"""

import types

import pytest

from vagen.rewards.sokoban import relations
from vagen.rewards.state_reward import StateRewardSpec, StateRewardWrapper


class Judge:
    """Returns whatever it is told to, per call."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    async def parse_batch(self, prompts):
        self.prompts.extend(prompts)
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


def _spec():
    return StateRewardSpec(
        relations=lambda env: env.state,
        judge_prompt="{content}",
        object_weights={"box": 1.0},
    )


def _wrapper(env, judge, **kw):
    return StateRewardWrapper(env=env, spec=_spec(), judge=judge, **kw)


@pytest.mark.asyncio
async def test_grounding_credit_lands_on_the_observation_tokens():
    """★ The reason for doing this at all. A scalar on the turn tells credit assignment
    only that the turn went well; this says which part of the reasoning was right."""
    action = "<observation>A</observation><answer>x</answer>"
    w = _wrapper(Env(BOX, BOX, reward=0.0), Judge(BOX), grounding_weight=1.0, format_reward=0.0)

    _, vector, _, _, _ = await w.step(action, [ord(c) for c in action], CharTokenizer())

    inside = action.index("A")
    assert vector[inside] > 0, "the description's own tokens got nothing"
    assert sum(v for i, v in enumerate(vector) if i != inside) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_prediction_is_scored_against_the_state_after_the_step():
    """★ Grounding describes what the agent acted from, prediction what it acted into.
    Scoring both against the same state would make one of them free."""
    action = "<prediction>A</prediction>"
    moved = [{"object_id": "box", "vertical_relation": "above", "horizontal_relation": "same"}]
    w = _wrapper(Env(before=BOX, after=moved), Judge(moved), worldmodeling_weight=1.0, format_reward=0.0)

    _, _, _, _, info = await w.step(action, [ord(c) for c in action], CharTokenizer())

    assert info["state_reward/prediction"] == pytest.approx(1.0), (
        "the prediction matched the state after the step and must score full marks"
    )

    # The same description against the state *before* the step describes the wrong row,
    # so it must score strictly less. Not zero: a relation half right earns half credit
    # by design, and the column was unchanged.
    w2 = _wrapper(Env(before=BOX, after=BOX), Judge(moved), worldmodeling_weight=1.0, format_reward=0.0)
    _, _, _, _, info2 = await w2.step(action, [ord(c) for c in action], CharTokenizer())
    assert info2["state_reward/prediction"] < info["state_reward/prediction"]


@pytest.mark.asyncio
async def test_a_wrong_description_earns_nothing_but_does_not_go_negative():
    action = "<observation>A</observation>"
    wrong = [{"object_id": "box", "vertical_relation": "above", "horizontal_relation": "left"}]
    w = _wrapper(Env(BOX, BOX, reward=0.0), Judge(wrong), grounding_weight=1.0, format_reward=0.0)

    _, vector, _, _, _ = await w.step(action, [ord(c) for c in action], CharTokenizer())

    assert sum(vector) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_a_judge_outage_costs_the_process_reward_not_the_rollout():
    """★ The judge is a parser, not part of training. Losing it should cost one turn's
    shaping, not raise into the rollout."""
    action = "<observation>A</observation>"
    w = _wrapper(Env(BOX, BOX, reward=2.0), Judge(None), grounding_weight=1.0, format_reward=0.0)

    _, vector, _, _, _ = await w.step(action, [ord(c) for c in action], CharTokenizer())

    assert sum(vector) == pytest.approx(2.0), "the environment's own reward must survive"


@pytest.mark.asyncio
async def test_the_outcome_reward_stays_on_the_last_token():
    action = "<observation>A</observation>zz"
    w = _wrapper(Env(BOX, BOX, reward=3.0), Judge(BOX), grounding_weight=1.0, format_reward=0.0)

    _, vector, _, _, _ = await w.step(action, [ord(c) for c in action], CharTokenizer())

    assert vector[-1] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_the_format_bonus_needs_both_descriptions():
    """Paying it for one would make the cheaper half of the format optional."""
    one = "<observation>A</observation>"
    both = "<observation>A</observation><prediction>B</prediction>"

    w1 = _wrapper(Env(BOX, BOX, reward=0.0), Judge(BOX), grounding_weight=0.0, format_reward=0.5)
    _, v1, _, _, _ = await w1.step(one, [ord(c) for c in one], CharTokenizer())

    w2 = _wrapper(Env(BOX, BOX, reward=0.0), Judge(BOX, BOX), grounding_weight=0.0,
                  worldmodeling_weight=0.0, format_reward=0.5)
    _, v2, _, _, _ = await w2.step(both, [ord(c) for c in both], CharTokenizer())

    assert sum(v1) == pytest.approx(0.0)
    assert sum(v2) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_without_tokens_the_wrapper_degrades_to_a_scalar():
    """An env used outside the token-aware loop should still work, just coarsely."""
    action = "<observation>A</observation>"
    w = _wrapper(Env(BOX, BOX, reward=1.0), Judge(BOX), grounding_weight=1.0, format_reward=0.0)

    _, reward, _, _, _ = await w.step(action)

    assert isinstance(reward, float) and reward == pytest.approx(2.0)


def test_sokoban_relations_are_relative_to_the_player():
    room = [[0, 0, 0], [0, 5, 0], [0, 4, 0]]      # player at (1,1), box below it
    fixed = [[0, 0, 0], [0, 0, 0], [0, 2, 0]]      # target at (2,1)
    env = types.SimpleNamespace(env=types.SimpleNamespace(room_state=room, room_fixed=fixed))

    import numpy as np

    env.env.room_state = np.array(room)
    env.env.room_fixed = np.array(fixed)
    items = relations(env)

    assert {"object_id": "box", "vertical_relation": "below", "horizontal_relation": "same"} in items
    assert {"object_id": "target", "vertical_relation": "below", "horizontal_relation": "same"} in items


def test_silence_about_an_absent_object_is_not_rewarded():
    """★ Per-type scoring plus 'both empty is perfect' would pay for saying nothing
    about things that were never there -- free credit in any scene missing a type."""
    from vagen.rewards.spatial import grouped_f1

    gold = BOX                                   # boxes present, no targets
    weights = {"box": 0.5, "target": 0.5}

    perfect_box_only = grouped_f1(BOX, gold, weights)
    assert perfect_box_only == pytest.approx(1.0), "describing everything present is full marks"

    said_nothing = grouped_f1([], gold, weights)
    assert said_nothing == pytest.approx(0.0), "silence about the boxes that exist earns nothing"

    hallucinated = grouped_f1(BOX + TGT, gold, weights)
    assert hallucinated < 1.0, "inventing a target must cost"
