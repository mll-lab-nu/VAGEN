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

from model_path import local_snapshot

from vagen.envs._common.rewards import state_reward_spec_of

def _state_reward_specs() -> dict:
    """Every environment that declares a state-reward spec, keyed by registry name.

    Derived from the registry rather than from a table, which is the point of the move:
    an environment gains the capability by declaring `STATE_REWARD_SPEC`, and nothing
    central has to be edited to agree with it.
    """
    import vagen.envs.registry as R
    from vagen.envs._common.rewards import state_reward_spec_of

    R._load_registry()
    out = {}
    for name, cls in R._ENV_REGISTRY.items():
        spec = state_reward_spec_of(cls)
        if spec is not None:
            out[name] = spec
    return out


STATE_REWARD_SPECS = _state_reward_specs()



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

from vagen.envs import StateRewardWrapper


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


# ------------------------------------------- offsets must match the text that was searched
def test_offsets_are_measured_the_way_the_action_text_was_decoded():
    """The client decodes the response with special tokens skipped, and the spans are
    found by matching tags in *that* string. Measuring offsets with specials rendered
    shifts every position after the first one, moving the reward off the description and
    onto the tag before it."""
    tok = _pytest.importorskip("transformers").AutoTokenizer.from_pretrained(
        local_snapshot() or "",
        trust_remote_code=True,
    ) if False else None

    class _Tok:
        """Two ordinary tokens and one special that prints when not skipped."""

        SPECIAL = 99

        def decode(self, ids, skip_special_tokens=False):
            out = []
            for i in ids:
                if i == self.SPECIAL:
                    if not skip_special_tokens:
                        out.append("<|special|>")
                else:
                    out.append(chr(ord("a") + int(i)))
            return "".join(out)

    from vagen.envs._common.rewards.spans import token_offsets, tokens_covering

    ids = [0, _Tok.SPECIAL, 1, 2]           # "a" <|special|> "b" "c"
    offsets = token_offsets(ids, _Tok())
    assert offsets == [1, 1, 2, 3], f"offsets counted the special token: {offsets}"

    # span "bc" in the skip-specials text "abc" is chars 1..3
    assert tokens_covering((1, 3), offsets) == [2, 3], (
        "the reward landed on the wrong tokens"
    )
