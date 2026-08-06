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
    assert out.extra_fields.get("token_level_scores") == [0.0, 1.0, 0.0, 2.0], (
        "only the sum survives; span-level credit is erased"
    )
    assert out.reward_score == 3.0, "the scalar is still needed for verl's own metrics"


def test_the_vector_is_capped_with_the_response_it_indexes():
    """A score at position 9 of a response truncated to 4 indexes nothing."""
    from vagen.agent_loop.gym_loop import GymLoop

    loop = GymLoop.__new__(GymLoop)
    loop.prompt_length = 100
    loop.response_length = 2          # cap below the response length
    out = GymLoop._outputs(loop, _Client(), _Env(), _Result(),
                           {"group_idx": "g", "traj_idx": 0}, "ep")[0]
    assert len(out.extra_fields["token_level_scores"]) == len(out.response_ids)
    assert out.reward_score == pytest.approx(sum(out.extra_fields["token_level_scores"]))


def test_upstream_writes_the_vector_rather_than_one_scalar():
    """The publisher is useless if verl still collapses it."""
    import inspect

    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    src = inspect.getsource(AgentLoopWorker._postprocess)
    assert "token_level_scores" in src, "verl no longer reads the per-token vector"
    assert "rm_scores[b, :width]" in src, "verl no longer writes it across positions"
