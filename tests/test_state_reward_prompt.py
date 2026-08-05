"""What the agent is shown must be a description, not the schema.

The judge's whole job is turning a description into structured relations. Show the agent
the structure and it emits JSON, the judge becomes a re-parser of its own output format,
and the score measures format compliance rather than whether the agent can see where
things are -- while the F1 goes up, because none of the ambiguity the judge absorbs ever
arises. That shipped once and read as a healthy 0.96.
"""

import json
import re

import pytest

from vagen.agent_loop.gym_loop import STATE_REWARD_SPECS


@pytest.mark.parametrize("env_name,spec", sorted(STATE_REWARD_SPECS.items()))
def test_the_examples_are_prose_not_json(env_name, spec):
    for section, text in spec.examples.items():
        body = re.sub(r"</?[a-z_]+>", "", text)
        inner = re.search(r"[\[{].*[\]}]", body, re.S)
        if inner:
            try:
                json.loads(inner.group(0))
            except ValueError:
                pass
            else:
                pytest.fail(
                    f"{env_name}/{section} shows the agent parseable JSON; it will emit "
                    f"the schema and the judge will have nothing to do"
                )


@pytest.mark.parametrize("env_name,spec", sorted(STATE_REWARD_SPECS.items()))
def test_the_examples_still_read_like_sentences(env_name, spec):
    for section, text in spec.examples.items():
        body = re.sub(r"</?[a-z_]+>", "", text)
        assert '"object_id"' not in body, f"{env_name}/{section} names schema keys"
        assert len(body.split()) >= 8, f"{env_name}/{section} is too terse to be a description"


def test_the_judge_prompt_is_the_thing_that_knows_the_schema():
    """The schema belongs on the judge's side of the boundary, not the agent's."""
    for spec in STATE_REWARD_SPECS.values():
        assert "object_id" in spec.judge_prompt


# --------------------------------------- the wrapper must not re-teach what the env asks
import pytest as _pytest

from vagen.rewards.state_reward import StateRewardWrapper


class _Env:
    def __init__(self, prompt):
        self._prompt = prompt

    async def system_prompt(self):
        return {"obs_str": self._prompt}


def _wrapper(env, **kw):
    spec = STATE_REWARD_SPECS["Sokoban"]
    return StateRewardWrapper(
        env=env, spec=spec,
        enabled=kw.get("enabled", {"state_estimation": 0.1, "transition_prediction": 0.1}),
    )


@_pytest.mark.asyncio
async def test_nothing_is_appended_when_the_env_already_asks_for_the_tags():
    """Sokoban's wm format already requests <observation> and <prediction> with worked
    examples. A second block does not reinforce them, it competes: that is how six of
    eight episodes stopped emitting a usable action."""
    env_prompt = "... <observation>The box is above the player</observation> <prediction>x</prediction> ..."
    w = _wrapper(_Env(env_prompt))
    assert w.instructions(env_prompt) == ""
    assert (await w.system_prompt())["obs_str"] == env_prompt


@_pytest.mark.asyncio
async def test_instructions_are_added_when_the_env_says_nothing():
    """An environment with no world-model prompt still needs to be told what to write."""
    env_prompt = "Push the boxes onto the targets. Reply with <answer>Up</answer>."
    w = _wrapper(_Env(env_prompt))
    block = w.instructions(env_prompt)
    assert "<observation>" in block and "<prediction>" in block
    assert (await w.system_prompt())["obs_str"].startswith(env_prompt)


@_pytest.mark.asyncio
async def test_only_the_missing_section_is_added():
    env_prompt = "... <observation>already asked for</observation> ..."
    block = _wrapper(_Env(env_prompt)).instructions(env_prompt)
    assert "<prediction>" in block
    assert "Before acting" not in block, "re-taught a section the env already requests"
