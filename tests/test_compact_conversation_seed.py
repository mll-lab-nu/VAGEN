"""Compaction reseeds from the latest world state."""

import types

import pytest

from vagen.harness import CompactHarness


class Client:
    def __init__(self):
        self.calls = []
        self.n = 0

    def size(self, messages):
        return 5

    async def create(self, messages, **kwargs):
        self.calls.append((list(messages), kwargs))
        self.n += 1
        summary = "Summarise" in str(messages[-1]["content"])
        text = "THE SUMMARY" if summary else f"act-{self.n}"
        # Two ordinary calls make the next turn cross budget=20.
        total = (8, 15, 18, 9, 16, 19)[min(self.n - 1, 5)]
        return types.SimpleNamespace(
            text=text, token_ids=[1], conversation_id="old" if self.n <= 3 else "new",
            usage=types.SimpleNamespace(total_tokens=total, response_tokens=total - 4),
        )


class Env:
    def __init__(self):
        self.n = 0
        self.actions = []

    async def reset(self):
        return {"role": "user", "content": "INITIAL", "images": ["frame0"]}, {}

    async def system_prompt(self):
        return {"role": "system", "content": "SYSTEM"}

    async def step(self, response):
        self.actions.append(response.text)
        self.n += 1
        obs = {"role": "user", "content": f"STEP-{self.n}", "images": [f"frame{self.n}"]}
        return obs, 0.0, self.n >= 4, False, {}

    def truncate(self, reason):
        raise AssertionError(reason)


@pytest.mark.asyncio
async def test_reseed_contains_summary_and_latest_observation_as_one_user_turn():
    client, env = Client(), Env()
    harness = CompactHarness(budget=20, summary_budget=4)
    await harness.run_episode(client, env)

    reseeds = [messages for messages, _ in client.calls
               if len(messages) == 2 and "Summary so far:" in str(messages[1]["content"])]
    assert reseeds
    seed = reseeds[0]
    assert [message["role"] for message in seed] == ["system", "user"]
    assert "THE SUMMARY" in seed[1]["content"]
    assert "STEP-2" in seed[1]["content"]
    assert seed[1]["images"] == ["frame2"]
    assert "THE SUMMARY" not in env.actions
