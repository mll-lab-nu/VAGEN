"""Tests for the variable-clock bi-level estimator."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from verl.trainer.ppo.core_algos import get_adv_estimator_fn

import vagen.algorithms  # noqa: F401 -- register estimators
from vagen.algorithms import needs_critic, requires_undiscounted, spans_rows


class _Cfg(dict):
    gamma = 1.0
    lam = 1.0

    def __init__(self, **values):
        super().__init__(values)
        for key, value in values.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)


def _call(scores, masks, values, turns, **algorithm):
    scores_t = torch.tensor(scores, dtype=torch.float64)
    masks_t = torch.tensor(masks, dtype=torch.long)
    batch = TensorDict(
        {
            "token_level_scores": scores_t,
            "token_level_rewards": scores_t.clone(),
            "response_mask": masks_t,
            "values": torch.tensor(values, dtype=torch.float64),
        },
        batch_size=[len(scores)],
    )
    nt = {
        "group_idx": np.array(["g"] * len(scores), dtype=object),
        "traj_idx": np.zeros(len(scores), dtype=int),
        "turn_idx": np.array(turns),
    }
    params = {
        "gamma_turn": 0.95,
        "lambda_turn": 0.9,
        "lambda_token": 1.0,
        **algorithm,
    }
    return get_adv_estimator_fn("bi_level_gae")(
        batch=batch,
        non_tensor_batch=nt,
        config=_Cfg(**params),
    )


def _valid(tensor, masks):
    mask = torch.tensor(masks, dtype=torch.bool)
    return [float(x) for row, keep in zip(tensor, mask) for x in row[keep]]


def test_concat_and_split_layouts_agree():
    concat = _call(
        [[0.0, 0.5, 0.0, 0.0, 1.0]],
        [[1, 1, 0, 1, 1]],
        [[0.1, 0.2, 9.0, 0.3, 0.4]],
        [0],
    )
    split = _call(
        [[0.0, 0.5], [0.0, 1.0]],
        [[1, 1], [1, 1]],
        [[0.1, 0.2], [0.3, 0.4]],
        [0, 1],
    )
    assert _valid(concat[0], [[1, 1, 0, 1, 1]]) == pytest.approx(
        _valid(split[0], [[1, 1], [1, 1]]), rel=1e-6, abs=1e-6
    )
    assert _valid(concat[1], [[1, 1, 0, 1, 1]]) == pytest.approx(
        _valid(split[1], [[1, 1], [1, 1]]), rel=1e-6, abs=1e-6
    )

def test_turn_end_reward_reaches_answer_tokens_that_section_reward_cannot_train():
    """Autoregressive credit only flows backward from a reward's token position."""
    masks = [[1, 1, 1, 1]]
    values = [[0.0, 0.0, 0.0, 0.0]]
    section_adv, section_returns = _call(
        [[0.0, 1.0, 0.0, 0.0]], masks, values, [0]
    )
    _, turn_returns = _call(
        [[0.0, 0.0, 0.0, 1.0]], masks, values, [0]
    )

    assert section_returns[0].tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert turn_returns[0].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert section_adv[0, :2].min() > 0
    assert section_adv[0, 2:].max() < 0, (
        "after whitening, answer tokens after an earlier section reward can be actively "
        "penalized even though the description was correct"
    )


def test_variable_lambda_mix_has_exact_default_and_bilevel_endpoints():
    args = (
        [[0.0, 0.5], [0.0, 1.0]],
        [[1, 1], [1, 1]],
        [[0.1, 0.2], [0.3, 0.4]],
        [0, 1],
    )
    default_adv, default_returns = _call(
        *args, gamma_turn=1.0, lambda_turn=1.0
    )
    bilevel_adv, bilevel_returns = _call(*args, bi_level_mix=1.0)

    assert torch.equal(
        _call(*args, bi_level_mix=0.0)[0], default_adv
    )
    assert torch.equal(
        _call(*args, bi_level_mix=1.0)[0], bilevel_adv
    )
    # Returns expose the pre-whitening mixture exactly; advantages are whitened after
    # mixing, so whitening each endpoint separately and averaging is not equivalent.
    assert _call(*args, bi_level_mix=0.5)[1] == pytest.approx(
        default_returns + 0.5 * (bilevel_returns - default_returns)
    )
    assert _call(*args, bi_level_mix=0.75)[1] == pytest.approx(
        default_returns + 0.75 * (bilevel_returns - default_returns)
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("bi_level_mix", 1.1),
        ("gamma_turn", -0.1),
        ("lambda_turn", float("inf")),
        ("lambda_token", float("nan")),
    ],
)
def test_bi_level_parameters_must_be_finite_probabilities(name, value):
    with pytest.raises(ValueError, match=rf"{name} must be finite and in"):
        _call(
            [[0.0, 1.0]],
            [[1, 1]],
            [[0.0, 0.0]],
            [0],
            **{name: value},
        )


def test_release_defaults_match_the_validated_setting():
    scores = [[0.0, 0.5], [0.0, 1.0]]
    masks = [[1, 1], [1, 1]]
    values = [[0.1, 0.2], [0.3, 0.4]]
    turns = [0, 1]

    scores_t = torch.tensor(scores, dtype=torch.float64)
    batch = TensorDict(
        {
            "token_level_scores": scores_t,
            "token_level_rewards": scores_t.clone(),
            "response_mask": torch.tensor(masks, dtype=torch.long),
            "values": torch.tensor(values, dtype=torch.float64),
        },
        batch_size=[len(scores)],
    )
    non_tensor_batch = {
        "group_idx": np.array(["g"] * len(scores), dtype=object),
        "traj_idx": np.zeros(len(scores), dtype=int),
        "turn_idx": np.array(turns),
    }
    default = get_adv_estimator_fn("bi_level_gae")(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        config=_Cfg(),
    )

    explicit = _call(
        scores,
        masks,
        values,
        turns,
        gamma_turn=0.95,
        lambda_turn=0.95,
        lambda_token=1.0,
        bi_level_mix=0.75,
    )
    assert torch.equal(default[0], explicit[0])
    assert torch.equal(default[1], explicit[1])


def test_registry_contracts_are_explicit():
    assert needs_critic("bi_level_gae")
    assert spans_rows("bi_level_gae")
    assert requires_undiscounted("bi_level_gae")
