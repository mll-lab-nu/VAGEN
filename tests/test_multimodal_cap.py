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

class _NoCompaction:
    """A harness that never summarised. `_outputs` asks it which conversations ended at a
    compaction seam rather than because the environment stepped; only CompactHarness ever
    answers non-empty."""

    summarised_conversations: set = set()




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


def test_an_over_budget_response_is_truncated_and_an_over_budget_prompt_is_not():
    """The two regions are not symmetric, and the asymmetry is the point.

    The response region holds observations, whose size the environment decides. Refusing
    there makes a long-tail rollout impossible to debug, and what gets cut is context:
    the model's own tokens are bounded by max_new_tokens, so only observations can
    overflow, and observations carry mask 0 and no reward.

    The prompt region is the *opening call* -- system prompt plus the first observation.
    Nothing in it is old, so a left cut takes the instructions. It still raises.
    """
    from omegaconf import OmegaConf

    from vagen.training.agent_loop.gym_loop import GymLoop

    class _Row:
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

    def _loop(prompt_len, response_len):
        loop = GymLoop.__new__(GymLoop)
        loop.prompt_length, loop.response_length = prompt_len, response_len
        loop.processor, loop.tokenizer, loop._ph_cache = None, None, None
        loop.config = OmegaConf.create({"trainer": {"harness": "concat", "compact_budget": 400}})
        return loop

    out = GymLoop._outputs(_loop(100, 3), _Client(), _Env(), _Result(),
                           {"group_idx": "g", "traj_idx": 0}, "ep", _NoCompaction())[0]
    n = len(out.response_ids)
    assert n == 3, f"the response was not cut to the budget: {n}"
    assert len(out.response_mask) == n, "mask outlived the ids it indexes"
    assert len(out.response_logprobs) == n, "logprobs outlived the ids"
    assert len(out.extra_fields["per_token_reward"]) == n, "reward vector outlived the ids"
    assert all(e <= n for _, e in out.extra_fields["response_spans"]), (
        f"spans point past the response: {out.extra_fields['response_spans']}"
    )

    class _BigPrompt(_Row):
        prompt_ids = list(range(50))

    class _BigClient(_Client):
        def rows(self): return [_BigPrompt()]

    with pytest.raises(ValueError, match="data.max_prompt_length"):
        GymLoop._outputs(_loop(3, 100), _BigClient(), _Env(), _Result(),
                         {"group_idx": "g", "traj_idx": 0}, "ep", _NoCompaction())



def test_the_overflow_hint_differs_by_mode():
    """One message for all three modes would send you to raise a number every time."""
    from vagen.training.agent_loop.gym_loop import GymLoop

    def hint(trainer):
        loop = GymLoop.__new__(GymLoop)
        loop.config = OmegaConf.create({"trainer": trainer})
        return GymLoop._overflow_hint(loop)

    assert "switch trainer.harness to compact" in hint({"harness": "concat"})
    assert "compact_budget=400" in hint({"harness": "compact", "compact_budget": 400})
    assert "single" in hint({"harness": "no_concat"})
    # Unset harness still names a mode rather than reporting none at all. It resolves to
    # concat, which is also what the shipped config now says outright.
    assert "harness=concat" in hint({"harness": None})


def test_gym_loop_defers_to_the_guard_rather_than_slicing():
    """The call site must not have kept a private slice.

    A regression here reintroduces the original bug while every unit test above still
    passes, because the helper would remain correct and simply go unused.
    """
    import inspect

    from vagen.training.agent_loop import gym_loop

    src = inspect.getsource(gym_loop.GymLoop._outputs)
    assert "cap_token_ids" in src, "_outputs no longer routes through the guard"
    for raw in ("[-self.prompt_length :]", "[: self.response_length]", "[-self.prompt_length:]"):
        assert raw not in src, f"_outputs still slices raw: {raw}"


def test_a_turn_straddling_the_cut_is_dropped_whole_not_clipped():
    """Clipping looks gentler and is worse.

    The surviving fragment keeps mask 1 -- trained on as if it were an action -- while
    the reward, which add_reward writes at scores[end-1], is past the cut and dropped.
    The model is optimised on half a move at reward zero. And it is always the *last*
    turn, because the response region ends on a model span: measured on a solve-at-5
    episode with a four-token overflow, 10.4 earned and 0.4 trained.
    """
    from omegaconf import OmegaConf

    from vagen.training.agent_loop.gym_loop import GymLoop

    class _Row:
        ordinal = 0
        conversation_id = "c"
        prompt_ids = [1]
        #        turn 0        obs          turn 1 (straddles a cut at 14)
        response_ids = [10, 11, 12, 13] + [20, 21, 22, 23] + [30, 31, 32, 33]
        response_mask = [1, 1, 1, 1] + [0, 0, 0, 0] + [1, 1, 1, 1]
        logprobs = [0.0] * 12
        scores = [0, 0, 0, 1.0] + [0] * 4 + [0, 0, 0, 9.0]     # the big one is last
        response_spans = [(0, 4), (8, 12)]

    class _Client:
        def rows(self): return [_Row()]
        def images(self, cid): return []

    class _Env:
        success = True
        state_scores = {}

    class _Result:
        turns = 2

    loop = GymLoop.__new__(GymLoop)
    loop.prompt_length, loop.response_length = 100, 10        # cuts inside span (8,12)
    loop.processor = loop.tokenizer = None
    loop._ph_cache = (set(), set())
    loop.config = OmegaConf.create({"trainer": {"harness": "concat", "compact_budget": 400}})

    out = GymLoop._outputs(loop, _Client(), _Env(), _Result(),
                           {"group_idx": "g", "traj_idx": 0}, "ep", _NoCompaction())[0]
    spans = out.extra_fields["response_spans"]
    assert spans == [(0, 4)], f"the straddling turn survived as a fragment: {spans}"
    assert all(e <= len(out.response_ids) for _, e in spans)
    # nothing past the last surviving span may still look like a decision
    assert not any(out.response_mask[4:]), (
        f"a half-turn is still masked as an action: {out.response_mask}"
    )
    assert out.reward_score == pytest.approx(sum(out.extra_fields["per_token_reward"]))


def test_absent_logprobs_are_published_as_absent():
    """The tape fills unsupplied positions with 0.0, and `[0.0, ...] or None` is the list
    -- so verl got a real rollout_log_probs tensor of zeros and read it as the rollout's
    actual belief."""
    from omegaconf import OmegaConf

    from vagen.training.agent_loop.gym_loop import GymLoop

    class _Row:
        ordinal = 0
        conversation_id = "c"
        prompt_ids = [1]
        response_ids = [10, 11]
        response_mask = [1, 1]
        logprobs = [0.0, 0.0]              # what the tape writes when none came back
        scores = [0.0, 1.0]
        response_spans = [(0, 2)]

    class _Client:
        def rows(self): return [_Row()]
        def images(self, cid): return []

    class _Env:
        success = True
        state_scores = {}

    class _Result:
        turns = 1

    loop = GymLoop.__new__(GymLoop)
    loop.prompt_length, loop.response_length = 100, 100
    loop.processor = loop.tokenizer = None
    loop._ph_cache = (set(), set())
    loop.config = OmegaConf.create({"trainer": {"harness": "concat", "compact_budget": 400}})

    out = GymLoop._outputs(loop, _Client(), _Env(), _Result(),
                           {"group_idx": "g", "traj_idx": 0}, "ep", _NoCompaction())[0]
    assert out.response_logprobs is None, (
        f"a fabricated all-zero logprob vector was published: {out.response_logprobs}"
    )
