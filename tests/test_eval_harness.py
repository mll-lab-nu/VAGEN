"""Evaluation runs the same episode loop, and the harness comes from the config.

Before this, ``evaluate/`` carried a second implementation of the turn loop that predated
the harness abstraction. It hardcoded concat, approximated no_concat with a boolean, and
could not express compaction at all -- and because ``_parse_env_specs`` copied a fixed key
list and dropped the rest in silence, writing ``harness: compact`` in an eval config was
accepted, ignored, and ran concat. Nothing in the suite touched the eval path, so none of
that was visible.

These tests assert the thing that distinguishes the three policies: what history each
model call carries. They use a stub adapter, so what is under test is the wiring and the
harnesses, not any endpoint.
"""

from __future__ import annotations

import asyncio

import pytest

from vagen.evaluate.vision_workflow import GenericVisionInferenceWorkflow


class _Adapter:
    """Records the messages of every call and answers with a fixed action."""

    def __init__(self, reply="<think>go</think><answer>Right</answer>"):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def format_system(self, text, images):
        return {"role": "system", "content": [{"type": "text", "text": text}]}

    def format_user_turn(self, text, images):
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    def format_assistant_turn(self, text):
        return {"role": "assistant", "content": [{"type": "text", "text": text}]}

    async def acompletion(self, messages, **chat_config):
        self.calls.append([dict(m) for m in messages])
        self.chat_config_seen = dict(chat_config)
        return self.reply


#: Long enough that a compaction budget in the low hundreds is actually reached. The
#: client estimates 4 characters to the token, so this is ~100 tokens an observation.
_PAD = "." * 380


class _Env:
    """A text-only environment that never terminates on its own."""

    def __init__(self, env_config):
        self.config = env_config
        self.i = 0

    async def reset(self, seed=None):
        self.i = 0
        return {"obs_str": f"observation 0 {_PAD}"}, {}

    async def system_prompt(self):
        return {"obs_str": "you are a solver"}

    async def step(self, action, **kw):
        self.i += 1
        return {"obs_str": f"observation {self.i} {_PAD}"}, 1.0, False, {}

    async def close(self):
        self.closed = True


def _run(harness, turns=4, **kw):
    adapter = _Adapter()
    wf = GenericVisionInferenceWorkflow(adapter=adapter, dump_dir=None,
                                        harness=harness, **kw)
    result = asyncio.run(wf.arun_episode(_Env, {"name": "Stub"}, seed=0, max_turns=turns))
    return adapter, result


def _roles(call):
    return [m["role"] for m in call]


# --------------------------------------------------------------- the three policies
def test_concat_carries_the_whole_conversation_forward():
    adapter, _ = _run("concat")
    lengths = [len(c) for c in adapter.calls]
    assert lengths == sorted(lengths) and lengths[-1] > lengths[0], lengths
    # system once, at the front, and every earlier turn still present
    assert _roles(adapter.calls[-1])[0] == "system"
    assert _roles(adapter.calls[-1]).count("assistant") == len(adapter.calls) - 1


def test_no_concat_sends_only_the_system_prompt_and_the_latest_observation():
    adapter, _ = _run("no_concat")
    for call in adapter.calls:
        assert _roles(call) == ["system", "user"], _roles(call)
    # ...and it is the *latest* observation, not the first one repeated
    seen = [c[-1]["content"][0]["text"] for c in adapter.calls]
    assert len(set(seen)) == len(seen), seen


def test_compact_summarises_and_reopens_rather_than_growing_forever():
    """The policy the old eval loop could not express at all. A conversation is closed by
    asking the model to summarise, and the next one opens on that summary."""
    adapter, _ = _run("compact", turns=6, response_length_per_turn=64,
                      compact_budget=200, compact_summary_budget=40)
    lengths = [len(c) for c in adapter.calls]
    # it must come back down at least once -- that is the reopen
    assert any(b < a for a, b in zip(lengths, lengths[1:])), lengths


@pytest.mark.parametrize("harness", ["concat", "no_concat", "compact"])
def test_every_harness_produces_a_scored_episode(harness):
    _, result = _run(harness, turns=3, response_length_per_turn=256,
                     compact_budget=600, compact_summary_budget=120)
    assert result["num_turns"] >= 1
    assert result["cumulative_reward"] > 0


# --------------------------------------------------------------- config, not code
def test_an_unknown_harness_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown harness"):
        GenericVisionInferenceWorkflow(adapter=_Adapter(), harness="concat_multi_turn")


def test_the_harness_key_reaches_the_workflow_from_the_yaml(tmp_path):
    """★ The bug this whole change exists for: the key used to be dropped between the
    config and the workflow, silently, so every eval ran concat whatever it said."""
    from vagen.evaluate.run_eval import _parse_env_specs

    specs = _parse_env_specs(
        {"envs": [{"name": "Sokoban", "n_envs": 1, "tag_id": 0,
                   "harness": "compact", "compact_budget": 900}]}
    )
    assert specs[0].harness == "compact"
    assert specs[0].compact_budget == 900


def test_a_key_nothing_reads_is_an_error_rather_than_a_shrug():
    from vagen.evaluate.run_eval import _parse_env_specs

    with pytest.raises(ValueError, match="which nothing reads"):
        _parse_env_specs({"envs": [{"name": "Sokoban", "n_envs": 1,
                                    "harnes": "compact"}]})


def test_response_length_per_turn_becomes_the_api_call_max_tokens():
    """It was in an eval config already, and dropped. Now it bounds a turn the same way
    it does in training rather than leaving the endpoint to its own default."""
    adapter, _ = _run("concat", turns=2, response_length_per_turn=77)
    assert adapter.chat_config_seen.get("max_tokens") == 77
