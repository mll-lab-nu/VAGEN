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
