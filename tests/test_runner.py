"""Tests for the episode loop.

The loop is short on purpose, so what is worth testing is what it refuses to do: act on
a response the harness kept, credit the wrong conversation, or report a turn limit as a
terminal state.
"""

import types

import pytest

from vagen.rollout.client import BackendOutput, InferenceClient
from vagen.harness import ConcatHarness, NoConcatHarness
from vagen.rollout.runner import run_episode


class Env:
    def __init__(self, steps=3, reward=1.0, terminate_at=None):
        self.steps, self.reward, self.terminate_at = steps, reward, terminate_at
        self.actions, self.closed = [], False

    async def reset(self, seed=None):
        return {"obs_str": "start"}, {}

    async def system_prompt(self):
        return {"role": "system", "content": "sys"}

    async def step(self, action, response_token_ids=None, tokenizer=None):
        self.actions.append(action)
        done = self.terminate_at is not None and len(self.actions) >= self.terminate_at
        return {"obs_str": f"obs{len(self.actions)}"}, self.reward, done, False, {}

    async def close(self):
        self.closed = True


class Client(InferenceClient):
    tokenizer = object()

    def __init__(self):
        super().__init__()
        self.n = 0

    def encode(self, messages):
        return [ord(c) for m in messages for c in str(m.get("content", ""))]

    async def generate(self, prompt_ids, **kwargs):
        self.n += 1
        return BackendOutput(text=f"act{self.n}", token_ids=[500 + self.n], prompt_token_ids=None)


@pytest.mark.asyncio
async def test_a_terminal_step_ends_the_episode():
    env, client = Env(terminate_at=2), Client()
    result = await run_episode(env, ConcatHarness(), client, max_turns=10)

    assert result.turns == 2 and result.terminated and not result.truncated
    assert env.closed


@pytest.mark.asyncio
async def test_running_out_of_turns_is_truncation_not_termination():
    """★ The value function must bootstrap past a turn limit and must not past a real
    ending; conflating them biases every episode that hits the cap."""
    env, client = Env(terminate_at=None), Client()
    result = await run_episode(env, ConcatHarness(), client, max_turns=3)

    assert result.turns == 3 and result.truncated and not result.terminated


@pytest.mark.asyncio
async def test_concat_produces_one_row_and_no_concat_one_per_turn():
    """★ The layout is entirely the harness's doing; the loop is identical."""
    env, client = Env(terminate_at=3), Client()
    await run_episode(env, ConcatHarness(), client, max_turns=10)
    assert len(client.rows()) == 1

    env2, client2 = Env(terminate_at=3), Client()
    await run_episode(env2, NoConcatHarness(), client2, max_turns=10)
    assert len(client2.rows()) == 3


@pytest.mark.asyncio
async def test_each_turns_reward_lands_on_that_turn():
    env, client = Env(terminate_at=3, reward=1.0), Client()
    await run_episode(env, ConcatHarness(), client, max_turns=10)

    row = client.rows()[0]
    assert sum(row.scores) == 3.0
    assert [i for i, s in enumerate(row.scores) if s] == [
        i for i, m in enumerate(row.response_mask) if m
    ], "one unit of credit per model token, since each turn is one token here"


@pytest.mark.asyncio
async def test_a_response_the_harness_keeps_is_not_given_to_the_environment():
    """★ Compaction's summary must not be stepped on. The loop asks again instead."""

    class KeepsFirst(ConcatHarness):
        def __init__(self):
            super().__init__()
            self.kept = 0

        def accept(self, response):
            if self.kept == 0:
                self.kept += 1
                return None
            return super().accept(response)

    env, client = Env(terminate_at=1), Client()
    await run_episode(env, KeepsFirst(), client, max_turns=5)

    assert env.actions == ["act2"], f"environment saw {env.actions}"
    assert client.n == 2, "the loop must ask again rather than skip the turn"


@pytest.mark.asyncio
async def test_the_environment_receives_the_token_ids_and_tokenizer():
    """An env that scores per token needs both; one that does not ignores them."""
    seen = {}

    class Recording(Env):
        async def step(self, action, response_token_ids=None, tokenizer=None):
            seen["ids"], seen["tok"] = response_token_ids, tokenizer
            return await super().step(action)

    client = Client()
    await run_episode(Recording(terminate_at=1), ConcatHarness(), client, max_turns=3)

    assert seen["ids"] == [501]
    assert seen["tok"] is Client.tokenizer


@pytest.mark.asyncio
async def test_a_vector_reward_is_summed_for_reporting_but_kept_per_token():
    class VectorEnv(Env):
        async def step(self, action, response_token_ids=None, tokenizer=None):
            obs, _, done, trunc, info = await super().step(action)
            return obs, [0.25] * len(response_token_ids), done, trunc, info

    client = Client()
    result = await run_episode(VectorEnv(terminate_at=2), ConcatHarness(), client, max_turns=5)

    assert result.total_reward == pytest.approx(0.5)
    assert client.rows()[0].scores.count(0.25) == 2


@pytest.mark.asyncio
async def test_a_concat_episode_accumulates_every_turn():
    """★ End to end across harness, client and record. Each layer's unit tests passed
    while the two disagreed about who removes already-sent messages, so the observations
    silently stopped entering the context. Only a test spanning them catches that."""
    env, client = Env(terminate_at=4), Client()
    await run_episode(env, ConcatHarness(), client, max_turns=10)

    import itertools

    row = client.rows()[0]
    runs = [(value, len(list(group))) for value, group in itertools.groupby(row.response_mask)]

    assert sum(row.response_mask) == 4, "one model token per turn"
    # 1, obs, 1, obs, 1, obs, 1 -- the last turn is terminal, so no observation follows.
    assert [value for value, _ in runs] == [1, 0, 1, 0, 1, 0, 1]
    assert all(length > 0 for value, length in runs if value == 0), "an observation is empty"


@pytest.mark.asyncio
async def test_a_budget_driven_harness_is_told_how_large_the_conversation_grew():
    """★ Without this the budget never moves and compaction never fires -- a harness
    that looks configured and silently behaves like plain concat."""
    from vagen.harness import CompactHarness

    seen = []

    class Watching(CompactHarness):
        def note_usage(self, used):
            seen.append(used)
            super().note_usage(used)

    env, client = Env(terminate_at=3), Client()
    await run_episode(env, Watching(budget=10**9), client, max_turns=5)

    assert seen and all(isinstance(u, int) for u in seen)
    assert seen == sorted(seen), "a conversation only grows, so the figures must not go backwards"


@pytest.mark.asyncio
async def test_compaction_fires_and_splits_the_episode_into_rows():
    """★ End to end: a budget small enough to trip every turn must start a new
    conversation each time, which is what makes the rows."""
    from vagen.harness import CompactHarness

    # A budget of one token trips on every turn, which is the shape the harness now
    # rejects: compaction that buys no turns is no_concat at twice the price. This fake
    # opens at 9 tokens, spends 5 a turn, and reopens at 30 after a summary, so 40 is
    # the smallest budget that leaves the *reseeded* conversation room for more than one
    # turn. Below it the harness refuses, correctly: at 30 the summary alone is three
    # quarters of the budget it has to fit inside.
    env, client = Env(terminate_at=12), Client()
    await run_episode(env, CompactHarness(budget=40), client, max_turns=12)

    assert len(client.rows()) > 1, "compaction never started a second conversation"
    assert all(any(r.response_mask) for r in client.rows()), "a row with no model output survived"


def test_the_environment_contract_is_written_down():
    """★ It used to exist only at this call site and in prose, so an environment could
    satisfy it by accident and fail in the one case nobody had written down."""
    import inspect

    from vagen.envs import BaseEnv

    step = inspect.signature(BaseEnv.step)
    assert {"action", "response_token_ids", "tokenizer"} <= set(step.parameters)

    doc = inspect.getdoc(BaseEnv.step)
    # The two distinctions an implementer gets wrong silently.
    assert "terminated" in doc and "truncated" in doc
    assert "re-encoding" in doc or "re-encoded" in doc


def test_a_turn_cannot_take_unboundedly_many_model_calls():
    """A backend whose response the harness always keeps spins the inner loop forever.

    Measured before this: 100,001 generations for one environment step, and under
    no_concat 100,001 conversations opened, for a client whose `text` came back None --
    which is what a closed API returns for a refusal. `Response.text` is typed `str` and
    nothing enforced it.
    """
    import asyncio

    import pytest

    from vagen.harness import BaseHarness, Call
    from vagen.rollout.runner import MAX_CALLS_PER_TURN, run_episode

    calls = []

    class _Keeps(BaseHarness):
        """Never yields an action -- the shape a misbehaving harness has."""

        def next_call(self):
            calls.append(1)
            return Call([{"role": "user", "content": "x"}], None)

        def accept(self, response):
            return None

    class _Client:
        tokenizer = None

        async def send(self, messages, conversation_id=None, **kw):
            class _R:
                text, conversation_id, token_ids = "act", "c1", [1]
            return _R()

        def usage(self, cid): return 1
        def reward(self, cid, v): pass
        def response_len(self, cid): return 0
        def measure(self, m): return 1

    class _Env:
        async def reset(self, seed): return {"obs_str": "o"}, {}
        async def system_prompt(self): return {"role": "system", "content": "s"}
        async def step(self, a, **kw): return {"obs_str": "o"}, 0.0, True, False, {}
        async def close(self): self.closed = True

    with pytest.raises(RuntimeError, match="without producing an action"):
        asyncio.run(run_episode(_Env(), _Keeps(), _Client(), max_turns=2))
    assert len(calls) <= MAX_CALLS_PER_TURN, f"{len(calls)} calls before giving up"


def test_a_non_string_response_is_treated_as_an_empty_generation():
    """`accept` forwards `response.text`; None is not None-the-sentinel, it is a value the
    loop would take for an action."""
    import asyncio

    from vagen.rollout.client import BackendOutput, InferenceClient

    class _C(InferenceClient):
        tokenizer = None

        def encode(self, messages): return [0]
        async def generate(self, prompt_ids, **kw):
            return BackendOutput(text=None, token_ids=[1])

    c = _C()
    r = asyncio.run(c.send([{"role": "user", "content": "x"}]))
    assert r.text == "", f"a None response reached the harness as {r.text!r}"
