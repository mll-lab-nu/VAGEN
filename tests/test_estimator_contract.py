"""The battery every registered estimator must pass, parametrised over the registry.

Adding an algorithm should not mean remembering what the multi-turn layout makes easy to
break. Register it and these run against it: the properties below are the ones that have
actually gone wrong here, and every one of them fails *silently* in training -- no
exception, no obviously wrong curve, just credit assigned to the wrong tokens.

A new estimator with a required hyperparameter needs an entry in ``PARAMS``. Forgetting
is not silent: ``test_runs_with_the_standard_fixture`` fails with instructions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from verl.trainer.ppo.core_algos import get_adv_estimator_fn

from vagen.algorithms import TRAJECTORY_ESTIMATORS

ESTIMATORS = sorted(TRAJECTORY_ESTIMATORS)

#: Hyperparameters an estimator requires before it will run at all. Empty today -- every
#: registered estimator runs on the shared defaults -- but the contract below is written
#: against this rather than against "no estimator needs anything", so adding one that does
#: means adding a line here rather than special-casing it.
PARAMS: dict[str, dict] = {}


class _Cfg(dict):
    """config.algorithm, which is both attribute- and get-accessed."""

    def __init__(self, gamma=1.0, lam=0.9, **extra):
        super().__init__()
        self.gamma, self.lam = gamma, lam
        for key, value in extra.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)


def _cfg_for(name, **overrides):
    return _Cfg(**{**PARAMS.get(name, {}), **overrides})


# --------------------------------------------------------------------------- fixtures
#
# One episode, two turns of two model-output tokens each, reward 1.0 on the final token.
# Laid out two ways. The values are attached to *tokens*, so they are the same numbers in
# both layouts -- which is what makes the two comparable at all.

VALUES = [0.10, 0.20, 0.30, 0.40]        # per model-output token, in emission order
REWARD_AT_END = 1.0


def _concat_rows(n_traj=2):
    """One row per episode: [a0 a1 <obs> a2 a3], the observation masked out."""
    scores, masks, values, group, traj, turn = [], [], [], [], [], []
    for i in range(n_traj):
        s = [0.0, 0.0, 0.0, 0.0, REWARD_AT_END * (i + 1)]
        scores.append(s)
        masks.append([1, 1, 0, 1, 1])
        values.append([VALUES[0], VALUES[1], 9.9, VALUES[2], VALUES[3]])
        group.append("g")
        traj.append(i)
        turn.append(0)
    return scores, masks, values, group, traj, turn


def _split_rows(n_traj=2):
    """One row per turn: [a0 a1] then [a2 a3]. Same tokens, same values, same rewards."""
    scores, masks, values, group, traj, turn = [], [], [], [], [], []
    for i in range(n_traj):
        scores += [[0.0, 0.0], [0.0, REWARD_AT_END * (i + 1)]]
        masks += [[1, 1], [1, 1]]
        values += [[VALUES[0], VALUES[1]], [VALUES[2], VALUES[3]]]
        group += ["g", "g"]
        traj += [i, i]
        turn += [0, 1]
    return scores, masks, values, group, traj, turn


def _call(name, layout, cfg=None, duplicate=None):
    scores, masks, values, group, traj, turn = layout
    if duplicate is not None:
        # Exactly what pad_dataproto_to_divisor appends: a byte-identical copy.
        for seq in (scores, masks, values, group, traj, turn):
            seq.append(seq[duplicate])
    batch = TensorDict(
        {
            "token_level_scores": torch.tensor(scores, dtype=torch.float32),
            "response_mask": torch.tensor(masks, dtype=torch.long),
            "values": torch.tensor(values, dtype=torch.float32),
        },
        batch_size=[len(scores)],
    )
    non_tensor = {
        "group_idx": np.array(group, dtype=object),
        "traj_idx": np.array(traj),
        "turn_idx": np.array(turn),
    }
    fn = get_adv_estimator_fn(name)
    adv, ret = fn(batch=batch, non_tensor_batch=non_tensor, config=cfg or _cfg_for(name))
    return adv, ret, torch.tensor(masks, dtype=torch.bool), batch


def _at_model_tokens(tensor, mask):
    """Values at model-output positions in order -- the one view that is layout-free."""
    return [float(v) for row, m in zip(tensor, mask) for v in row[m]]


# ----------------------------------------------------------------------------- battery


@pytest.mark.parametrize("name", ESTIMATORS)
def test_runs_with_the_standard_fixture(name):
    """Smoke test, and the place a missing PARAMS entry announces itself."""
    try:
        adv, ret, mask, _ = _call(name, _concat_rows())
    except ValueError as exc:
        if "algorithm." in str(exc):
            pytest.fail(
                f"{name} requires a hyperparameter this battery does not supply. Add it "
                f"to PARAMS in this file so the contract tests can run it.\n  {exc}"
            )
        raise
    assert adv.shape == ret.shape == mask.shape
    assert torch.isfinite(adv).all(), f"{name} produced non-finite advantages"


@pytest.mark.parametrize("name", ESTIMATORS)
def test_same_numbers_under_concat_and_no_concat(name):
    """★ The property the whole layer exists for. concat and no-concat are two layouts
    of one trajectory, not two algorithms, so an estimator must not be able to tell them
    apart. Failing this is how an estimator turns out to be a no-concat-only algorithm
    wearing an algorithm's name."""
    a_concat, _, m_concat, _ = _call(name, _concat_rows())
    a_split, _, m_split, _ = _call(name, _split_rows())

    assert _at_model_tokens(a_concat, m_concat) == pytest.approx(
        _at_model_tokens(a_split, m_split), rel=1e-5, abs=1e-6
    ), f"{name} scores the two layouts differently"


@pytest.mark.parametrize("name", ESTIMATORS)
def test_observation_tokens_get_no_advantage(name):
    """The model did not emit them, so they are state. A non-zero advantage there is a
    gradient on a token the policy never chose."""
    adv, _, mask, _ = _call(name, _concat_rows())
    assert torch.all(adv[~mask] == 0), f"{name} put advantage on non-model tokens"


@pytest.mark.parametrize("name", ESTIMATORS)
def test_padding_duplicates_do_not_change_the_result(name):
    """★ ``pad_dataproto_to_divisor`` repeats real rows to reach a multiple of the DP
    world size. Scoring a duplicate as a separate episode double-counts it in every
    group statistic and every backward recursion -- and the batch size that triggers it
    depends on the cluster, so it reproduces on 8 GPUs and not on 4."""
    plain, _, mask, _ = _call(name, _concat_rows())
    padded, _, mask_padded, _ = _call(name, _concat_rows(), duplicate=0)

    n = plain.shape[0]
    assert _at_model_tokens(padded[:n], mask_padded[:n]) == pytest.approx(
        _at_model_tokens(plain, mask), rel=1e-5, abs=1e-6
    ), f"{name} changed its answer when a padding copy was appended"
    # And the copy is given the same numbers as the row it copies.
    assert padded[-1].tolist() == pytest.approx(padded[0].tolist())


@pytest.mark.parametrize("name", ESTIMATORS)
def test_tolerates_the_extra_kwargs_verl_passes(name):
    """verl adds `index` and `reward_baselines` for estimators that want them, and hands
    them to everyone that declares **kwargs. A stale signature only fails after a
    cluster is up and a rollout has run."""
    scores, masks, values, group, traj, turn = _concat_rows()
    batch = TensorDict(
        {
            "token_level_scores": torch.tensor(scores, dtype=torch.float32),
            "response_mask": torch.tensor(masks, dtype=torch.long),
            "values": torch.tensor(values, dtype=torch.float32),
        },
        batch_size=[len(scores)],
    )
    get_adv_estimator_fn(name)(
        batch=batch,
        non_tensor_batch={
            "group_idx": np.array(group, dtype=object),
            "traj_idx": np.array(traj),
            "turn_idx": np.array(turn),
        },
        config=_cfg_for(name),
        index=np.array(group, dtype=object),
        reward_baselines=torch.zeros(len(scores)),
    )


@pytest.mark.parametrize("name", ESTIMATORS)
def test_verl_will_actually_hand_over_the_containers(name):
    """★ verl decides whether to pass `batch`/`non_tensor_batch` by inspecting the
    registered function's signature. `functools.wraps` sets `__wrapped__`, which
    `inspect.signature` follows to the inner function -- so a decorator written the
    obvious way makes verl pass neither, and the estimator sees an empty batch."""
    import inspect

    params = inspect.signature(get_adv_estimator_fn(name)).parameters
    assert "batch" in params and "non_tensor_batch" in params, (
        f"verl will not pass the containers to {name}; is the adapter using "
        "functools.wraps?"
    )


@pytest.mark.parametrize("name", ESTIMATORS)
def test_returns_are_supervised_or_declared_unsupervised(name):
    """Every position of `returns` either carries a real target or is excluded by
    `value_mask`. The failure this catches is the critic regressing towards the -100
    sentinel, which makes its loss *fall* -- so nothing looks wrong."""
    from vagen.algorithms import needs_value_mask
    from vagen.training.trainer.logic import IGNORE_RETURN

    _, ret, mask, batch = _call(name, _concat_rows())
    sentinel_at = (ret == IGNORE_RETURN) & mask
    if not sentinel_at.any():
        return

    assert needs_value_mask(name), (
        f"{name} leaves positions at the sentinel but is not registered with "
        "sentinel_returns=True, so nothing will build a value_mask for it"
    )
    assert "value_mask" in batch.keys(), (
        f"{name} declares sentinel returns but did not publish a value_mask"
    )
    excluded = ~batch["value_mask"].to(torch.bool)
    assert torch.all(excluded[sentinel_at]), (
        f"{name}'s value_mask still supervises sentinel positions"
    )
