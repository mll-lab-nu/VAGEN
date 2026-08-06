"""A per-token reward must survive the trip into the training batch.

The state reward exists to pay <observation> and <prediction> on the tokens that carry
them. verl's AgentLoopOutput has one `reward_score: float`, written at the final token --
so a vector handed to it is summed away and every span-level score becomes a single
number at the end of the response. Nothing fails; turn- and token-level credit
assignment just quietly has nothing to assign.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class _Row:
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
    from vagen.agent_loop.gym_loop import GymLoop

    loop = GymLoop.__new__(GymLoop)
    loop.prompt_length = 100
    loop.response_length = 100
    return GymLoop._outputs(loop, _Client(), _Env(), _Result(),
                            {"group_idx": "g", "traj_idx": 0}, "ep")[0]


def test_the_vector_is_published_not_only_its_sum():
    out = _output()
    assert out.extra_fields.get("per_token_reward") == [0.0, 1.0, 0.0, 2.0], (
        "only the sum survives; span-level credit is erased"
    )
    assert out.reward_score == 3.0, "the scalar is still needed for verl's own metrics"


def test_the_scalar_is_the_sum_of_the_vector_that_trains():
    """verl's metrics read the scalar and the loss reads the vector, so a run whose
    reported reward and actual reward disagree looks healthy from the outside."""
    from vagen.agent_loop.gym_loop import GymLoop

    loop = GymLoop.__new__(GymLoop)
    loop.prompt_length, loop.response_length = 100, 100     # both fit
    out = GymLoop._outputs(loop, _Client(), _Env(), _Result(),
                           {"group_idx": "g", "traj_idx": 0}, "ep")[0]
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

    from vagen.agent_loop import gym_loop

    src = inspect.getsource(gym_loop.GymLoop._outputs)
    tensor_names = {"token_level_scores", "token_level_rewards", "responses",
                    "response_mask", "advantages", "returns", "values", "old_log_probs"}
    for name in tensor_names:
        assert f'"{name}":' not in src, (
            f"the loop publishes an extra_field named {name!r}, which is also a tensor "
            f"key; to_tensordict will refuse the batch"
        )
    assert '"per_token_reward":' in src
