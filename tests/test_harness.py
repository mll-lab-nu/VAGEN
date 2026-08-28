"""Tests for harness-owned episode policies."""

import inspect
import types

import pytest

from vagen.harness import CompactHarness, ConcatHarness, NoConcatHarness


class Client:
    def __init__(self):
        self.calls = []
        self.n = 0

    def size(self, messages):
        return sum(len(str(message.get("content", ""))) for message in messages)

    async def create(self, messages, **kwargs):
        self.calls.append((list(messages), kwargs))
        self.n += 1
        total = sum(self.size([message]) for message in messages) + self.n
        return types.SimpleNamespace(
            text=f"reply-{self.n}",
            token_ids=[self.n],
            conversation_id=f"c{1 + sum('Summary so far:' in str(call[0]) for call in self.calls)}",
            usage=types.SimpleNamespace(
                prompt_tokens=total - 1,
                completion_tokens=1,
                response_tokens=max(1, total - 5),
                total_tokens=total,
            ),
        )


class Env:
    def __init__(self, turns=3):
        self.turns = turns
        self.actions = []
        self.stop_reason = None

    async def reset(self):
        return {"role": "user", "content": "obs0"}, {}

    async def system_prompt(self):
        return {"role": "system", "content": "sys"}

    async def step(self, response):
        self.actions.append(response.text)
        done = len(self.actions) >= self.turns
        return {"role": "user", "content": f"obs{len(self.actions)}"}, 0.0, done, False, {}

    def truncate(self, reason):
        self.stop_reason = reason


@pytest.mark.asyncio
async def test_concat_grows_one_message_history():
    client = Client()
    await ConcatHarness().run_episode(client, Env(3))
    assert [[m["content"] for m in messages] for messages, _ in client.calls] == [
        ["sys", "obs0"],
        ["sys", "obs0", "reply-1", "obs1"],
        ["sys", "obs0", "reply-1", "obs1", "reply-2", "obs2"],
    ]


@pytest.mark.asyncio
async def test_no_concat_rebuilds_context_each_turn():
    client = Client()
    await NoConcatHarness().run_episode(client, Env(3))
    assert [[m["content"] for m in messages] for messages, _ in client.calls] == [
        ["sys", "obs0"], ["sys", "obs1"], ["sys", "obs2"]
    ]


@pytest.mark.asyncio
async def test_compact_keeps_summary_from_environment_and_reseeds_latest_observation():
    client, env = Client(), Env(4)
    harness = CompactHarness(budget=20, summary_budget=4)
    await harness.run_episode(client, env)

    summary_calls = [messages for messages, _ in client.calls
                     if "Summarise" in str(messages[-1].get("content"))]
    assert summary_calls, "the configured budget never caused compaction"
    assert not any(action.startswith("reply-3") for action in env.actions), (
        "the summary response was sent to the environment"
    )
    reseeds = [messages for messages, _ in client.calls
               if len(messages) == 2 and "Summary so far:" in str(messages[1]["content"])]
    assert reseeds and "obs2" in str(reseeds[0][1]["content"])
    assert harness.summarised_conversations


@pytest.mark.parametrize("cls", [ConcatHarness, NoConcatHarness, CompactHarness])
def test_every_harness_owns_an_episode_loop(cls):
    assert inspect.iscoroutinefunction(cls.run_episode)
    source = inspect.getsource(cls)
    assert "next_call" not in source and "accept(" not in source


def test_every_policy_is_reachable_by_name():
    from vagen.harness import HARNESSES, build_harness

    assert set(HARNESSES) == {"concat", "no_concat", "compact"}
    assert isinstance(build_harness("concat"), ConcatHarness)


def test_unknown_name_lists_available_harnesses():
    from vagen.harness import build_harness

    with pytest.raises(ValueError, match="choose from"):
        build_harness("concatenate")


def test_common_contract_contains_no_concrete_harness():
    from vagen.harness._common import base

    concrete = [
        obj for obj in vars(base).values()
        if inspect.isclass(obj) and issubclass(obj, base.BaseHarness) and obj is not base.BaseHarness
    ]
    assert concrete == []
