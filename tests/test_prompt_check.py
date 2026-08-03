"""Tests for the prompt/engine length agreement check.

The failure it guards against is invisible by construction: both sequences are
well-formed, neither side sees both, and the loss stays finite while the score simply
fails to improve. So the check itself has to be hard to get wrong.
"""

import types
import warnings

import pytest

from vagen.agent_loop.prompt_check import PromptLengthMismatch, check_prompt_matches_engine


def _output(count):
    return types.SimpleNamespace(extra_fields={} if count is None else {"prompt_token_count": count})


def test_matching_lengths_pass_quietly():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_prompt_matches_engine([1, 2, 3], _output(3))


def test_mismatch_raises_by_default():
    """★ The whole point: sampling from one sequence and training on another is an
    off-policy corruption that no metric reports."""
    with pytest.raises(PromptLengthMismatch, match="disagrees with the inference engine"):
        check_prompt_matches_engine([1, 2, 3], _output(729))


def test_message_names_the_direction_and_the_knob():
    """A bare 'mismatch' sends the reader hunting; the delta and the config key are
    what actually shorten the search."""
    with pytest.raises(PromptLengthMismatch) as excinfo:
        check_prompt_matches_engine([1] * 300, _output(256), env_name="sokoban")

    message = str(excinfo.value)
    assert "sokoban" in message
    assert "300" in message and "256" in message and "-44" in message
    assert "mm_processor_kwargs" in message


def test_non_strict_warns_instead():
    with pytest.warns(UserWarning, match="disagrees with the inference engine"):
        check_prompt_matches_engine([1, 2], _output(5), strict=False)


def test_engine_that_reports_nothing_is_not_an_error():
    """★ Absence of the field is not evidence of a problem -- older servers and
    non-vLLM backends simply do not carry it, and failing on that would make the check
    impossible to roll out."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_prompt_matches_engine([1, 2, 3], _output(None))
        check_prompt_matches_engine([1, 2, 3], types.SimpleNamespace())
        check_prompt_matches_engine([1, 2, 3], types.SimpleNamespace(extra_fields=None))


def test_both_loops_run_the_check():
    """★ A loop that skips it is exactly where the corruption would hide."""
    import ast
    import inspect

    from vagen.agent_loop import gym_agent_loop, gym_agent_loop_no_concat

    for module in (gym_agent_loop, gym_agent_loop_no_concat):
        called = {
            node.func.id
            for node in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "check_prompt_matches_engine" in called, f"{module.__name__} does not check"


# ------------------------------------------------------- adopting the engine's prompt


def _output_with_ids(ids):
    return types.SimpleNamespace(
        extra_fields={"prompt_token_ids": list(ids), "prompt_token_count": len(ids)}
    )


def test_engine_prompt_ids_are_returned_as_a_plain_list():
    from vagen.agent_loop.prompt_check import engine_prompt_ids

    out = engine_prompt_ids(_output_with_ids((1, 2, 3)))

    assert out == [1, 2, 3] and isinstance(out, list)


def test_absent_engine_prompt_is_none_not_empty():
    """None means 'nothing reported', which the caller answers by falling back to the
    length check; an empty list would read as a legitimately empty prompt."""
    from vagen.agent_loop.prompt_check import engine_prompt_ids

    assert engine_prompt_ids(types.SimpleNamespace(extra_fields={})) is None
    assert engine_prompt_ids(types.SimpleNamespace()) is None
    assert engine_prompt_ids(types.SimpleNamespace(extra_fields={"prompt_token_ids": []})) is None


def test_only_the_per_turn_loop_adopts_the_engine_prompt():
    """★ The concat loop accumulates one prompt across turns and tracks its response
    mask by appending counts, so adopting a re-expanded prompt would move tokens the
    mask has already been measured against. It must keep checking instead."""
    import ast
    import inspect

    from vagen.agent_loop import gym_agent_loop, gym_agent_loop_no_concat

    def calls(module):
        return {
            node.func.id
            for node in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

    assert "engine_prompt_ids" in calls(gym_agent_loop_no_concat)
    assert "engine_prompt_ids" not in calls(gym_agent_loop), (
        "the accumulating loop must not adopt a re-expanded prompt"
    )
    assert "check_prompt_matches_engine" in calls(gym_agent_loop)
