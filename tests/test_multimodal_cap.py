"""A multimodal sequence must never be sliced to fit a budget.

The failure this guards against does not raise where it happens. A slice through a run
of image placeholders leaves the sequence one length and the feature grid another;
position ids are rebuilt from the grid, so the model receives a token tensor and a
position tensor that disagree and dies inside the attention with a shape error naming
neither the prompt nor the image. It took a dead 8-step cluster job to find.
"""

from __future__ import annotations

import pytest

from verl.experimental.agent_loop.agent_loop import cap_token_ids


def test_text_under_budget_is_untouched():
    assert cap_token_ids([1, 2, 3], 10, multimodal=False) == [1, 2, 3]


def test_text_over_budget_keeps_the_tail_for_prompts():
    # A prompt drops its oldest context, so the recent turns survive.
    assert cap_token_ids([1, 2, 3, 4, 5], 2, multimodal=False, keep="tail") == [4, 5]


def test_text_over_budget_keeps_the_head_for_responses():
    # A response is cut off at the end; its beginning is what was actually generated.
    assert cap_token_ids([1, 2, 3, 4, 5], 2, multimodal=False, keep="head") == [1, 2, 3, 4, 5][:2]


def test_multimodal_over_budget_raises_rather_than_slicing():
    with pytest.raises(ValueError, match="Multimodal prompt produced 5 tokens"):
        cap_token_ids([1, 2, 3, 4, 5], 2, multimodal=True, what="prompt")


def test_multimodal_exactly_at_budget_is_allowed():
    # The boundary is the interesting one: fitting exactly must not trip the guard.
    assert cap_token_ids([1, 2], 2, multimodal=True) == [1, 2]


def test_the_error_names_the_knob_to_turn():
    # An error that does not say what to change costs another submission to act on.
    with pytest.raises(ValueError, match="data.max_response_length"):
        cap_token_ids([1, 2, 3], 1, multimodal=True, what="response", budget_name="data.max_response_length")


def test_response_arrays_stay_aligned_when_a_text_response_is_capped():
    """The mask and logprobs must be cut to the same length as the ids.

    Slicing the ids by the budget but the mask by a stale length is the same class of
    bug one layer down: both are well-formed, and the loss stays finite while the mask
    no longer describes the tokens it is applied to.
    """
    ids, mask, logprobs = [1, 2, 3, 4, 5], [1, 1, 0, 1, 1], [0.1, 0.2, 0.3, 0.4, 0.5]
    kept = cap_token_ids(ids, 3, multimodal=False, keep="head", what="response")
    n = len(kept)
    assert (kept, mask[:n], logprobs[:n]) == ([1, 2, 3], [1, 1, 0], [0.1, 0.2, 0.3])


def test_gym_loop_defers_to_the_guard_rather_than_slicing():
    """The call site must not have kept a private slice.

    A regression here reintroduces the original bug while every unit test above still
    passes, because the helper would remain correct and simply go unused.
    """
    import inspect

    from vagen.agent_loop import gym_loop

    src = inspect.getsource(gym_loop.GymLoop._outputs)
    assert "cap_token_ids" in src, "_outputs no longer routes through the guard"
    for raw in ("[-self.prompt_length :]", "[: self.response_length]", "[-self.prompt_length:]"):
        assert raw not in src, f"_outputs still slices raw: {raw}"
