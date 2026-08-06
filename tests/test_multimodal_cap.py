"""A multimodal sequence must never be sliced to fit a budget.

The failure this guards against does not raise where it happens. A slice through a run
of image placeholders leaves the sequence one length and the feature grid another;
position ids are rebuilt from the grid, so the model receives a token tensor and a
position tensor that disagree and dies inside the attention with a shape error naming
neither the prompt nor the image. It took a dead 8-step cluster job to find.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import cap_token_ids


def test_text_under_budget_is_untouched():
    assert cap_token_ids([1, 2, 3], 10, multimodal=False) == [1, 2, 3]


def test_text_over_budget_keeps_the_tail_for_prompts():
    # verl's default, kept: trimming a dataset prompt to fit is ordinary, and the sample
    # is still the sample afterwards.
    assert cap_token_ids([1, 2, 3, 4, 5], 2, multimodal=False, keep="tail") == [4, 5]


def test_text_over_budget_keeps_the_head_for_responses():
    # A response is cut off at the end; its beginning is what was actually generated.
    assert cap_token_ids([1, 2, 3, 4, 5], 2, multimodal=False, keep="head") == [1, 2, 3, 4, 5][:2]


def test_text_over_budget_can_be_made_to_raise():
    with pytest.raises(ValueError, match="system prompt"):
        cap_token_ids([1, 2, 3], 2, multimodal=False, keep="tail", on_overflow="raise")
    with pytest.raises(ValueError, match="closing turns and their rewards"):
        cap_token_ids([1, 2, 3], 2, multimodal=False, keep="head", on_overflow="raise")


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


def test_an_episode_over_the_budget_is_refused_rather_than_trained_on():
    """Text overflow raises in the agent loop, unlike verl's default.

    Truncating here does not produce a shorter version of the episode, it produces a
    different one: the closing turns go, and the rewards earned in them go with the
    spans that index them -- the arrays stay mutually consistent and the loss stays
    finite, so nothing downstream can tell. The message has to name the mode, because
    the budget is the same knob in all three and the reason it was hit never is.
    """
    from vagen.agent_loop.gym_loop import GymLoop

    class _Row:
        # Numbered where the conversation was opened, not by position among the rows
        # that survive -- see Conversation.ordinal.
        ordinal = 0
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
    loop.config = OmegaConf.create({"trainer": {"harness": "concat", "compact_budget": 400}})

    with pytest.raises(ValueError) as e:
        GymLoop._outputs(loop, _Client(), _Env(), _Result(),
                         {"group_idx": "g", "traj_idx": 0}, "ep")
    assert "data.max_response_length" in str(e.value), "the message does not name the knob"
    assert "harness=concat" in str(e.value), "the message does not name the mode that overflowed"
    assert "compact" in str(e.value), "concat's overflow does not point at the mode that fixes it"


def test_the_overflow_hint_differs_by_mode():
    """One message for all three modes would send you to raise a number every time."""
    from vagen.agent_loop.gym_loop import GymLoop

    def hint(trainer):
        loop = GymLoop.__new__(GymLoop)
        loop.config = OmegaConf.create({"trainer": trainer})
        return GymLoop._overflow_hint(loop)

    assert "switch trainer.harness to compact" in hint({"harness": "concat"})
    assert "compact_budget=400" in hint({"harness": "compact", "compact_budget": 400})
    assert "single" in hint({"harness": "no_concat"})
    # Unset harness follows the legacy flag rather than reporting no mode at all.
    assert "harness=concat" in hint({"harness": None, "concat_multi_turn": True})
    assert "harness=no_concat" in hint({"harness": None, "concat_multi_turn": False})


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
