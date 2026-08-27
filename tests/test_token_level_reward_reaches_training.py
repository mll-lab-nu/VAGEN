"""A per-token reward must survive the trip into the training batch.

The state reward exists to pay <observation> and <prediction> on the tokens that carry
them. verl's AgentLoopOutput has one `reward_score: float`, written at the final token --
so a vector handed to it is summed away and every span-level score becomes a single
number at the end of the response. Nothing fails; turn- and token-level credit
assignment just quietly has nothing to assign.
"""

from __future__ import annotations

import pytest

class _NoCompaction:
    """A harness that never summarised. `_outputs` asks it which conversations ended at a
    compaction seam rather than because the environment stepped; only CompactHarness ever
    answers non-empty."""

    summarised_conversations: set = set()



torch = pytest.importorskip("torch")


class _Row:
    # Numbered where the conversation was opened, not by position among the rows
    # that survive -- see Conversation.ordinal.
    ordinal = 0
    conversation_id = "c"
    prompt_ids = [1, 2]
    response_ids = [10, 11, 12, 13]
    response_mask = [1, 1, 1, 1]
    logprobs = [0.0] * 4
    response_spans = [(0, 4)]
    #: 1.0 earned by token 1, 2.0 by token 3 -- different spans, different amounts
    scores = [0.0, 1.0, 0.0, 2.0]


class _Client:
    def rows(self):
        return [_Row()]

    def images(self, conversation_id):
        return []


class _Env:
    success = True
    state_scores = {"state_estimation": 1.0, "format": 0.0}


class _Result:
    turns = 1


def _output():
    from vagen.training.agent_loop.gym_loop import GymLoop

    loop = GymLoop.__new__(GymLoop)
    loop.prompt_length = 100
    loop.response_length = 100
    return GymLoop._outputs(loop, _Client(), _Env(), _Result(),
                            {"group_idx": "g", "traj_idx": 0}, "ep", _NoCompaction())[0]


def test_the_vector_is_published_not_only_its_sum():
    out = _output()
    assert out.extra_fields.get("per_token_reward") == [0.0, 1.0, 0.0, 2.0], (
        "only the sum survives; span-level credit is erased"
    )
    assert out.reward_score == 3.0, "the scalar is still needed for verl's own metrics"


def test_the_scalar_is_the_sum_of_the_vector_that_trains():
    """verl's metrics read the scalar and the loss reads the vector, so a run whose
    reported reward and actual reward disagree looks healthy from the outside."""
    from vagen.training.agent_loop.gym_loop import GymLoop

    loop = GymLoop.__new__(GymLoop)
    loop.prompt_length, loop.response_length = 100, 100     # both fit
    out = GymLoop._outputs(loop, _Client(), _Env(), _Result(),
                           {"group_idx": "g", "traj_idx": 0}, "ep", _NoCompaction())[0]
    assert len(out.extra_fields["per_token_reward"]) == len(out.response_ids)
    assert out.reward_score == pytest.approx(sum(out.extra_fields["per_token_reward"]))


def test_upstream_writes_the_vector_rather_than_one_scalar():
    """The publisher is useless if verl still collapses it."""
    import inspect

    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    src = inspect.getsource(AgentLoopWorker._postprocess)
    assert "per_token_reward" in src, "verl no longer reads the per-token vector"
    assert "rm_scores[b, :width]" in src, "verl no longer writes it across positions"


def test_the_key_does_not_collide_with_a_tensor_verl_already_has():
    """extra_fields become non-tensor columns. verl has a *tensor* called
    token_level_scores, and DataProto.to_tensordict asserts the two namespaces are
    disjoint -- so naming the vector that killed every run at the critic update, three
    modes in a row, with an assertion that names only the key."""
    from verl.protocol import DataProto

    import inspect

    from vagen.training.agent_loop import gym_loop

    src = inspect.getsource(gym_loop.GymLoop._outputs)
    tensor_names = {"token_level_scores", "token_level_rewards", "responses",
                    "response_mask", "advantages", "returns", "values", "old_log_probs"}
    for name in tensor_names:
        assert f'"{name}":' not in src, (
            f"the loop publishes an extra_field named {name!r}, which is also a tensor "
            f"key; to_tensordict will refuse the batch"
        )
    assert '"per_token_reward":' in src


# ------------------------------------- reported reward must equal trained reward


class _TruncatedRow:
    """A two-turn conversation whose second turn is cut by the response budget.

    Turn 0 covers tokens 0-3, turn 1 covers 4-7. The budget keeps 6 tokens, so turn 1
    straddles the cut and is dropped whole -- its mask goes to 0. A *vector* reward pays
    near the start of a turn, so turn 1's 0.3 sits at token 4, inside what survives the
    `[:keep]` slice but underneath a mask that has been zeroed.
    """

    ordinal = 0
    conversation_id = "c"
    prompt_ids = [1, 2]
    response_ids = [10, 11, 12, 13, 14, 15, 16, 17]
    response_mask = [1, 1, 1, 1, 1, 1, 1, 1]
    logprobs = [0.0] * 8
    response_spans = [(0, 4), (4, 8)]
    scores = [0.0, 0.0, 0.0, 0.5, 0.3, 0.0, 0.0, 0.0]


class _TruncatedClient:
    def rows(self):
        return [_TruncatedRow()]

    def images(self, conversation_id):
        return []


def _truncated_output(response_length):
    from omegaconf import OmegaConf

    from vagen.training.agent_loop.gym_loop import GymLoop

    loop = GymLoop.__new__(GymLoop)
    loop.prompt_length = 100
    loop.response_length = response_length
    # The truncation path reports which policy it is running under.
    loop.config = OmegaConf.create({"trainer": {"harness": "concat", "compact_budget": 400}})
    return GymLoop._outputs(loop, _TruncatedClient(), _Env(), _Result(),
                            {"group_idx": "g", "traj_idx": 0}, "ep", _NoCompaction())[0]


def test_a_dropped_turns_reward_leaves_with_its_mask():
    """★ Half of this was already fixed and half was not. A *scalar* reward sits at a
    turn's last token and is clipped away with the span. A *vector* reward pays near the
    turn's start, so it survived the slice while the mask above it was zeroed: the
    estimators gather only mask-1 positions and drop it, but `token_level_scores` still
    carried it into critic/score/mean, the custom metrics, and the STARPO-S filter's
    per-sample reward -- which decides which groups survive.
    """
    out = _truncated_output(6)
    mask = list(out.response_mask)
    rewards = out.extra_fields["per_token_reward"]

    assert mask[4:] == [0, 0], "the dropped turn should be unmasked"
    orphaned = sum(r for r, m in zip(rewards, mask) if not m)
    assert orphaned == 0.0, (
        f"{orphaned} of reward sits on tokens the loss will never see; reported reward "
        "and trained reward disagree"
    )


def test_the_scalar_matches_what_is_actually_trained():
    out = _truncated_output(6)
    trained = sum(r for r, m in zip(out.extra_fields["per_token_reward"], out.response_mask) if m)
    assert out.reward_score == pytest.approx(trained), (
        "verl's metrics read the scalar and the loss reads the vector; they must agree"
    )
    assert out.reward_score == pytest.approx(0.5), "only turn 0's reward survives"


def test_an_untruncated_row_keeps_every_reward():
    """The fix must not eat rewards when nothing was cut."""
    out = _truncated_output(100)
    assert out.extra_fields["per_token_reward"] == [0.0, 0.0, 0.0, 0.5, 0.3, 0.0, 0.0, 0.0]
    assert out.reward_score == pytest.approx(0.8)
