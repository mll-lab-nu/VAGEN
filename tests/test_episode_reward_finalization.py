"""Episode-aware rewards are finalized before token rows leave the rollout."""

from types import SimpleNamespace

import pytest

from vagen.rollout.runner import EpisodeResult
from vagen.rollout.scoring import ScoringSeam


class _Deferred(list):
    def finalize_episode(self, turns):
        return [value / turns for value in self]


class _Env:
    def __init__(self):
        self.turns = 0
        self.closed = False

    async def step(self, _response):
        self.turns += 1
        return {}, _Deferred([2.0]), self.turns == 2, False, {}

    async def close(self):
        self.closed = True


class _Client:
    def __init__(self):
        self.rewards = []

    def reward_call(self, call_id, reward):
        self.rewards.append((call_id, reward))


@pytest.mark.asyncio
async def test_deferred_rewards_use_realized_episode_length_and_keep_call_identity():
    env, client, result = _Env(), _Client(), EpisodeResult()
    seam = ScoringSeam(env, client, result)

    await seam.step(SimpleNamespace(call_id=7))
    await seam.step(SimpleNamespace(call_id=11))
    assert client.rewards == [], "length-dependent credit must not be committed early"

    await seam.close()

    assert client.rewards == [(7, [1.0]), (11, [1.0])]
    assert result.rewards == pytest.approx([1.0, 1.0])
    assert env.closed
