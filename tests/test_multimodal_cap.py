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


def test_the_loop_cuts_the_parallel_arrays_to_the_same_length():
    """Production code, not arithmetic performed by the test. Slicing mask and logprobs
    inside the test and then asserting they match proves only that Python slices."""
    from vagen.agent_loop.gym_loop import GymLoop

    class _Row:
        conversation_id = "c"
        prompt_ids = [1, 2]
        response_ids = [10, 11, 12, 13, 14]
        response_mask = [1, 1, 0, 1, 1]
        logprobs = [0.1, 0.2, 0.3, 0.4, 0.5]
        scores = [0.0, 1.0, 0.0, 2.0, 0.0]
        response_spans = [(0, 2), (3, 5)]

    class _Client:
        def rows(self): return [_Row()]
        def images(self, cid): return []

    class _Env:
        success = False
        state_scores = {}

    class _Result:
        turns = 2

    loop = GymLoop.__new__(GymLoop)
    loop.prompt_length, loop.response_length = 100, 3     # cap below the response
    out = GymLoop._outputs(loop, _Client(), _Env(), _Result(),
                           {"group_idx": "g", "traj_idx": 0}, "ep")[0]
    n = len(out.response_ids)
    assert n == 3
    assert len(out.response_mask) == n, "mask outlived the ids it indexes"
    assert len(out.response_logprobs) == n, "logprobs outlived the ids"
    assert len(out.extra_fields["per_token_reward"]) == n, "reward vector outlived the ids"
    # The spans index into response_ids, so they must be cut with it. Left whole, the
    # second span (3, 5) points past a 3-token response, and the turn it describes is
    # silently dropped by a range check further downstream.
    assert all(e <= n for _, e in out.extra_fields["response_spans"]), (
        f"spans point past the response: {out.extra_fields['response_spans']}"
    )


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
