"""Integration tests for harness-owned episodes and the scoring seam."""

import inspect

import pytest

from vagen.harness import BaseHarness, ConcatHarness, NoConcatHarness
from vagen.rollout.client import BackendOutput, EpisodeBudgetExceeded, InferenceClient
from vagen.rollout.runner import run_episode


class Env:
    def __init__(self, reward=1.0, terminate_at=None):
        self.reward, self.terminate_at = reward, terminate_at
        self.responses, self.closed = [], False

    async def reset(self, seed=None):
        return {"obs_str": "start"}, {}

    async def system_prompt(self):
        return {"role": "system", "content": "sys"}

    async def step(self, response):
        self.responses.append(response)
        done = self.terminate_at is not None and len(self.responses) >= self.terminate_at
        return {"obs_str": f"obs{len(self.responses)}"}, self.reward, done, False, {}

    async def close(self):
        self.closed = True


class Client(InferenceClient):
    tokenizer = object()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = 0

    def encode(self, messages):
        return [ord(c) for message in messages for c in str(message.get("content", ""))]

    async def generate(self, prompt_ids, **kwargs):
        self.n += 1
        return BackendOutput(text=f"act{self.n}", token_ids=[500 + self.n])


@pytest.mark.asyncio
async def test_terminal_status_and_cleanup_are_owned_by_the_seam():
    env, client = Env(terminate_at=2), Client()
    result = await run_episode(env, ConcatHarness(), client, max_turns=10)
    assert result.turns == 2 and result.terminated and not result.truncated
    assert env.closed


@pytest.mark.asyncio
async def test_runner_backstop_reports_turn_exhaustion_as_truncation():
    result = await run_episode(Env(), ConcatHarness(), Client(), max_turns=3)
    assert result.turns == 3 and result.truncated and not result.terminated


@pytest.mark.asyncio
async def test_concat_produces_one_row_and_no_concat_one_per_turn():
    concat = Client()
    await run_episode(Env(terminate_at=3), ConcatHarness(), concat, max_turns=10)
    assert len(concat.rows()) == 1

    separate = Client()
    await run_episode(Env(terminate_at=3), NoConcatHarness(), separate, max_turns=10)
    assert len(separate.rows()) == 3


@pytest.mark.asyncio
async def test_scoring_seam_credits_the_exact_call():
    client = Client()
    result = await run_episode(Env(reward=1.0, terminate_at=3), ConcatHarness(), client)
    row = client.rows()[0]
    assert result.total_reward == 3.0
    assert sum(row.scores) == 3.0
    assert [i for i, score in enumerate(row.scores) if score] == [
        i for i, mask in enumerate(row.response_mask) if mask
    ]


@pytest.mark.asyncio
async def test_environment_receives_response_tokens_and_client_tokenizer():
    env, client = Env(terminate_at=1), Client()
    await run_episode(env, ConcatHarness(), client)
    response = env.responses[0]
    assert response.text == "act1"
    assert response.token_ids == [501]
    assert response.tokenizer is Client.tokenizer


@pytest.mark.asyncio
async def test_vector_reward_is_reported_as_scalar_and_kept_per_token():
    class VectorEnv(Env):
        async def step(self, response):
            obs, _, done, truncated, info = await super().step(response)
            return obs, [0.25] * len(response.token_ids), done, truncated, info

    client = Client()
    result = await run_episode(VectorEnv(terminate_at=2), ConcatHarness(), client)
    assert result.total_reward == pytest.approx(0.5)
    assert client.rows()[0].scores.count(0.25) == 2


@pytest.mark.asyncio
async def test_concat_full_messages_are_routed_to_one_incremental_conversation():
    client = Client()
    await run_episode(Env(terminate_at=4), ConcatHarness(), client)
    assert sum(client.rows()[0].response_mask) == 4
    assert len(client.conversations()) == 1


def test_environment_contract_is_response_shaped():
    from vagen.envs import BaseEnv

    assert "response" in inspect.signature(BaseEnv.step).parameters
    doc = inspect.getdoc(BaseEnv.step)
    assert "terminated" in doc and "truncated" in doc
    assert "re-encoding" in doc or "re-encoded" in doc


def test_client_bounds_a_harness_that_generates_without_stepping():
    import asyncio

    class NeverSteps(BaseHarness):
        async def run_episode(self, client, env):
            await env.reset()
            while True:
                await client.create([{"role": "user", "content": "x"}])

    client = Client(max_calls=4)
    with pytest.raises(EpisodeBudgetExceeded, match="4 model calls"):
        asyncio.run(run_episode(Env(), NeverSteps(), client))


def test_non_string_backend_text_is_normalised():
    import asyncio

    class NoneClient(Client):
        async def generate(self, prompt_ids, **kwargs):
            return BackendOutput(text=None, token_ids=[1])

    response = asyncio.run(NoneClient().create([{"role": "user", "content": "x"}]))
    assert response.text == ""
