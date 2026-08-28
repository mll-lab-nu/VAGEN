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
    "<perception>A box is below me</perception>"
    "<reasoning>I should move down</reasoning>"
    "<prediction>The box will move</prediction>"
    "<answer>Down</answer>"
)


def test_the_canonical_form_is_still_the_correct_one():
    p = parse_wm(CANONICAL)
    assert p["format_correct"] is True
    assert p["actions"] == ["down"]
    assert p["perception_content"] == "A box is below me"


def test_react_labels_still_yield_their_action():
    """The exact shape observed: plain labels, no tags."""
    p = parse_wm(
        "Observation: The box is below and left of the player.\n"
        "Thought: Pushing the box aligns it with the target.\n"
        "Action: Up, Left\n"
    )
    assert p["actions"] == ["up", "left"], "a usable action was thrown away"
    assert p["format_correct"] is False, "labels must not count as the requested format"
    assert p["perception_content"].startswith("The box is below")


def test_action_is_accepted_as_a_tag_alias():
    """Observed drift: the model reaches for <action> where the env wants <answer>."""
    p = parse_wm("<perception>x</perception><action>Right</action>")
    assert p["actions"] == ["right"]
    assert p["format_correct"] is False


def test_a_missing_section_no_longer_costs_the_whole_turn():
    """All four or nothing was the rule; three of four now still steps the env."""
    p = parse_wm("<perception>A box is below</perception><answer>Down</answer>")
    assert p["actions"] == ["down"]
    assert p["prediction_content"] == ""
    assert p["format_correct"] is False


def test_out_of_order_sections_are_still_read():
    p = parse_wm(
        "<answer>Up</answer><perception>the box</perception><prediction>it moves</prediction>"
    )
    assert p["actions"] == ["up"]
    assert p["perception_content"] == "the box"


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
    assert "box sits directly above" in p["perception_content"]
    assert "pushed one square up" in p["prediction_content"]


@pytest.mark.parametrize("label", ["Action", "action", "ACTION", "Answer"])
def test_label_casing_and_synonyms(label):
    assert parse_wm(f"{label}: Left")["actions"] == ["left"]


# ----------------------------------------------- the `answer` format (native thinking)


def test_answer_format_reads_the_action_and_keeps_the_reasoning():
    """For models whose thinking is native, everything before <answer> is their own
    reasoning -- recorded, but not required to take any shape."""
    from vagen.envs.sokoban.utils.utils import parse_response

    p = parse_response(
        "The box sits to my right and the target beyond it, so I push right twice.\n"
        "<answer>Right,Right</answer>",
        prompt_format="answer",
    )
    assert p["format_correct"] is True
    assert p["actions"] == ["right", "right"]
    assert "push right twice" in p["reasoning_content"]


def test_answer_format_does_not_demand_the_tags_a_thinking_model_cannot_write():
    """`<think>` is a reserved control token on Qwen3.5, so a format that requires it as
    text is unsatisfiable there -- which is the whole reason this format exists."""
    from vagen.envs.sokoban.utils.utils import parse_response

    native = "reasoning with no tags at all whatsoever\n<answer>Up</answer>"
    assert parse_response(native, prompt_format="answer")["format_correct"] is True
    # the same response fails the tag-based formats
    assert parse_response(native, prompt_format="wm")["format_correct"] is False
    assert parse_response(native, prompt_format="wm_think")["format_correct"] is False


def test_wm_think_accepts_native_reasoning_before_the_canonical_suffix():
    from vagen.envs.sokoban.utils.utils import parse_response

    native = "native reasoning\n</think>\n" + CANONICAL
    parsed = parse_response(native, prompt_format="wm_think")
    assert parsed["format_correct"] is True
    assert parsed["actions"] == ["down"]


def test_wm_think_uses_the_final_native_close_before_the_structured_suffix():
    from vagen.envs.sokoban.utils.utils import parse_response

    native = "forced mid-thought</think>)\n</think>\n" + CANONICAL
    assert parse_response(native, prompt_format="wm_think")["format_correct"] is True


def test_legacy_free_wm_name_is_only_a_compatibility_alias():
    from vagen.envs.sokoban.utils.utils import PROMPT_FORMATS, parse_response

    assert "free_wm" not in PROMPT_FORMATS
    assert parse_response(CANONICAL, prompt_format="free_wm")["format_correct"] is True


def test_answer_format_still_salvages_an_action_when_the_tag_is_missing():
    """Same leniency the other formats have: format_correct goes False, the episode
    continues rather than dying on syntax."""
    from vagen.envs.sokoban.utils.utils import parse_response

    p = parse_response("I will go up.\nAction: Up", prompt_format="answer")
    assert p["format_correct"] is False
    assert p["actions"] == ["up"]


def test_answer_format_caps_the_action_count():
    from vagen.envs.sokoban.utils.utils import parse_response

    p = parse_response("<answer>Up,Down,Left,Right</answer>", prompt_format="answer", max_actions=3)
    assert p["actions"] == ["up", "down", "left"]


def test_the_answer_prompt_can_drop_the_worked_example():
    """Native thinking makes the example counterproductive: it teaches imitation of a
    shape the model no longer has to produce."""
    from vagen.envs.sokoban.utils.prompt import format_prompt

    with_ex = format_prompt(3, ",", add_example=True, prompt_format="answer")
    without = format_prompt(3, ",", add_example=False, prompt_format="answer")
    assert "Example:" in with_ex and "Example:" not in without
    assert "<answer>" in without


# ------------------------------------------- the `free_think` format (think + answer)
#
# The point of this format on a native-thinking model is the *closing* tag: it is the only
# clause in any sokoban format that says "stop reasoning". These tests pin the two halves
# of that -- a response that closes and then answers passes, one that never closes fails,
# however much it rambled and whatever it drafted along the way.


def test_free_think_accepts_the_classic_both_tags_form():
    from vagen.envs.sokoban.utils.utils import parse_response

    p = parse_response(
        "<think>The box is below me.</think><answer>Down</answer>",
        prompt_format="free_think",
    )
    assert p["format_correct"] is True
    assert p["actions"] == ["down"]
    assert p["reasoning_content"] == "The box is below me."


def test_free_think_accepts_a_think_block_the_chat_template_opened():
    """Qwen3.5's generation prompt ends `assistant\\n<think>\\n`, so the response starts
    inside the block and its first tag is the closing one. Requiring the opening tag makes
    the format unsatisfiable on exactly the family it is meant for."""
    from vagen.envs.sokoban.utils.utils import parse_response

    p = parse_response(
        "The box is below me, so I push down.\n</think>\n\n<answer>Down</answer>",
        prompt_format="free_think",
    )
    assert p["format_correct"] is True
    assert p["actions"] == ["down"]
    assert "push down" in p["reasoning_content"]


def test_free_think_rejects_visible_prose_between_the_close_and_the_answer():
    from vagen.envs.sokoban.utils.utils import parse_response

    p = parse_response(
        "<think>reasoning</think>\nHere is my plan, at some length.\n<answer>Up,Left</answer>",
        prompt_format="free_think",
    )
    assert p["format_correct"] is False
    assert p["actions"] == ["up", "left"]


def test_free_think_salvages_glm_native_box_without_format_credit():
    from vagen.envs.sokoban.utils.utils import parse_response

    raw = (
        "<think>The box is below me, so I should move down.</think>\n"
        "<|begin_of_box|>Down<|end_of_box|>"
    )
    p = parse_response(raw, prompt_format="free_think")
    assert p["format_correct"] is False
    assert p["actions"] == ["down"]


def test_free_think_salvages_answer_nested_inside_glm_native_box():
    from vagen.envs.sokoban.utils.utils import parse_response

    p = parse_response(
        "<think>reasoning</think>"
        "<|begin_of_box|><answer>Left</answer><|end_of_box|>",
        prompt_format="free_think",
    )
    assert p["format_correct"] is False
    assert p["actions"] == ["left"]


def test_free_think_rejects_a_trace_that_never_closed_its_thinking():
    """★ The failure this format exists to catch. Measured on Qwen3.5-4B under the
    `answer` format, half of all rollouts ran to the 16384-token cap without emitting
    `</think>` -- and those that had drafted an `<answer>` mid-ramble were scored
    format_correct and paid the format reward, rewarding non-termination."""
    from vagen.envs.sokoban.utils.utils import parse_response

    ramble = "wait, let me reconsider. " * 50 + "<answer>Up</answer>" + " hmm, but actually "
    p = parse_response(ramble, prompt_format="free_think")
    assert p["format_correct"] is False
    assert parse_response(ramble, prompt_format="answer")["format_correct"] is False


def test_free_think_takes_the_answer_after_the_close_not_a_draft_inside_it():
    from vagen.envs.sokoban.utils.utils import parse_response

    p = parse_response(
        "maybe <answer>Up</answer> no wait </think> On reflection: <answer>Down</answer>",
        prompt_format="free_think",
    )
    assert p["format_correct"] is False
    assert p["actions"] == ["down"]


def test_free_think_still_salvages_an_action_when_the_format_fails():
    """Same two-tier contract as parse_wm: strict for format_correct, lenient for
    extraction, so a malformed turn keeps its data instead of dying on syntax."""
    from vagen.envs.sokoban.utils.utils import parse_response

    p = parse_response("I never closed my thinking.\nAction: Up", prompt_format="free_think")
    assert p["format_correct"] is False
    assert p["actions"] == ["up"]


def test_free_think_caps_the_action_count():
    from vagen.envs.sokoban.utils.utils import parse_response

    p = parse_response(
        "<think>t</think><answer>Up,Down,Left,Right</answer>",
        prompt_format="free_think",
        max_actions=3,
    )
    assert p["actions"] == ["up", "down", "left"]


def test_the_free_think_prompt_names_the_closing_tag_not_the_opening_one():
    """`<think>` is a reserved control token on the native-thinking families, so an
    instruction to emit it is unsatisfiable; the instruction that carries the format is to
    close the block and answer after it."""
    from vagen.envs.sokoban.utils.prompt import format_prompt

    p = format_prompt(3, ",", add_example=False, prompt_format="free_think")
    assert "</think>" in p and "must come after" in p
