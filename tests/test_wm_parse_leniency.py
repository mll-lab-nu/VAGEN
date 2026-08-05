"""A response that did the task must not be discarded over punctuation.

The wm parser was one ordered regex over four tags: any deviation returned no action, so
the environment stepped nowhere and the episode ended. Measured on the base model, half
of all episodes reached zero usable actions that way -- while writing correct spatial
reasoning under ``Observation:`` / ``Action:``, the ReAct labels it was pretrained on.

Extraction is forgiving now; ``format_correct`` is not. Keeping those apart is the point:
the data survives, the pressure to write the tags stays.
"""

from __future__ import annotations

import pytest

from vagen.envs.sokoban.utils.utils import parse_wm

CANONICAL = (
    "<observation>A box is below me</observation>"
    "<think>I should move down</think>"
    "<answer>Down</answer>"
    "<prediction>The box will move</prediction>"
)


def test_the_canonical_form_is_still_the_correct_one():
    p = parse_wm(CANONICAL)
    assert p["format_correct"] is True
    assert p["actions"] == ["down"]
    assert p["observation_content"] == "A box is below me"


def test_react_labels_still_yield_their_action():
    """The exact shape observed: plain labels, no tags."""
    p = parse_wm(
        "Observation: The box is below and left of the player.\n"
        "Thought: Pushing the box aligns it with the target.\n"
        "Action: Up, Left\n"
    )
    assert p["actions"] == ["up", "left"], "a usable action was thrown away"
    assert p["format_correct"] is False, "labels must not count as the requested format"
    assert p["observation_content"].startswith("The box is below")


def test_action_is_accepted_as_a_tag_alias():
    """Observed drift: the model reaches for <action> where the env wants <answer>."""
    p = parse_wm("<observation>x</observation><action>Right</action>")
    assert p["actions"] == ["right"]
    assert p["format_correct"] is False


def test_a_missing_section_no_longer_costs_the_whole_turn():
    """All four or nothing was the rule; three of four now still steps the env."""
    p = parse_wm("<observation>A box is below</observation><answer>Down</answer>")
    assert p["actions"] == ["down"]
    assert p["prediction_content"] == ""
    assert p["format_correct"] is False


def test_out_of_order_sections_are_still_read():
    p = parse_wm(
        "<answer>Up</answer><observation>the box</observation><prediction>it moves</prediction>"
    )
    assert p["actions"] == ["up"]
    assert p["observation_content"] == "the box"


def test_max_actions_holds_on_the_lenient_path_too():
    """The cap is what stops one turn from playing the whole episode."""
    p = parse_wm("Action: Up, Down, Left, Right, Up", max_actions=3)
    assert p["actions"] == ["up", "down", "left"]


def test_nothing_usable_stays_nothing():
    p = parse_wm("I am not sure what to do here.")
    assert p["actions"] == []
    assert p["format_correct"] is False


def test_the_descriptions_survive_for_the_judge_to_score():
    """The state reward reads these spans; losing them loses the auxiliary signal."""
    p = parse_wm(
        "Observation: A box sits directly above me.\n"
        "Action: Up\n"
        "Prediction: The box will be pushed one square up.\n"
    )
    assert "box sits directly above" in p["observation_content"]
    assert "pushed one square up" in p["prediction_content"]


@pytest.mark.parametrize("label", ["Action", "action", "ACTION", "Answer"])
def test_label_casing_and_synonyms(label):
    assert parse_wm(f"{label}: Left")["actions"] == ["left"]
