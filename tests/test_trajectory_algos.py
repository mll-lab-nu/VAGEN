"""Tests for the layout-independent advantage estimators.

The claim these exist to support is that concat and no-concat are two layouts of the
same trajectory, not two algorithms. So the central test scores one trajectory both ways
and requires the same numbers out; the rest cover what the layouts make easy to get
wrong -- skipping observation tokens, carrying the recursion across a row boundary, and
padding duplicates being counted twice.
"""

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from verl.trainer.ppo.core_algos import get_adv_estimator_fn

from vagen.custom_advantage.trajectory import TrajectoryView

TOKEN_GAE = get_adv_estimator_fn("traj_token_gae")
TRAJ_GRPO = get_adv_estimator_fn("traj_grpo")


class _Cfg(dict):
    """Stands in for config.algorithm, which is attribute- and get-accessed."""

    gamma = 1.0
    lam = 1.0

    def get(self, key, default=None):
        return getattr(self, key, default)


def _batch(scores, masks, values=None):
    scores = torch.tensor(scores, dtype=torch.float32)
    masks = torch.tensor(masks, dtype=torch.long)
    values = torch.zeros_like(scores) if values is None else torch.tensor(values, dtype=torch.float32)
    return TensorDict(
        {"token_level_scores": scores, "response_mask": masks, "values": values},
        batch_size=[scores.shape[0]],
    )


def _nt(group, traj, turn):
    return {
        "group_idx": np.array(group, dtype=object),
        "traj_idx": np.array(traj),
        "turn_idx": np.array(turn),
    }


def _tokens(adv, mask):
    """Advantages at model-output positions, in order -- position-independent."""
    return [float(v) for row, m in zip(adv, torch.tensor(mask, dtype=torch.bool)) for v in row[m]]


# ------------------------------------------------------------------ layout equivalence


def test_token_gae_is_the_same_under_both_layouts():
    """★ The whole point. One trajectory, two turns of two tokens, reward 1.0 on the
    last token. Laid out as a single concat row (with an observation token between the
    turns) and as two no-concat rows, token-level GAE must produce the same numbers."""
    cfg = _Cfg()

    concat_mask = [[1, 1, 0, 1, 1]]  # turn 1, observation, turn 2
    concat = TOKEN_GAE(
        batch=_batch([[0.0, 0.0, 5.0, 0.0, 1.0]], concat_mask, [[0.1, 0.2, 9.0, 0.3, 0.4]]),
        non_tensor_batch=_nt(["g"], [0], [0]),
        config=cfg,
    )

    noconcat_mask = [[1, 1], [1, 1]]
    noconcat = TOKEN_GAE(
        batch=_batch([[0.0, 0.0], [0.0, 1.0]], noconcat_mask, [[0.1, 0.2], [0.3, 0.4]]),
        non_tensor_batch=_nt(["g", "g"], [0, 0], [0, 1]),
        config=cfg,
    )

    # approx, not exact: the two layouts reach the same values by different reduction
    # orders, so they agree only to float32 precision.
    assert _tokens(concat[0], concat_mask) == pytest.approx(_tokens(noconcat[0], noconcat_mask), rel=1e-5)
    assert _tokens(concat[1], concat_mask) == pytest.approx(_tokens(noconcat[1], noconcat_mask), rel=1e-5)


def test_grpo_is_the_same_under_both_layouts():
    """Two trajectories in one group, scoring 1 and 0."""
    cfg = _Cfg()

    concat_mask = [[1, 1, 0, 1], [1, 1, 0, 1]]
    concat = TRAJ_GRPO(
        batch=_batch([[0.0, 0.0, 7.0, 1.0], [0.0, 0.0, 7.0, 0.0]], concat_mask),
        non_tensor_batch=_nt(["g", "g"], [0, 1], [0, 0]),
        config=cfg,
    )

    noconcat_mask = [[1, 1], [1, 0], [1, 1], [1, 0]]
    noconcat = TRAJ_GRPO(
        batch=_batch([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], noconcat_mask),
        non_tensor_batch=_nt(["g", "g", "g", "g"], [0, 0, 1, 1], [0, 1, 0, 1]),
        config=cfg,
    )

    assert set(_tokens(concat[0], concat_mask)) == set(_tokens(noconcat[0], noconcat_mask))


# ------------------------------------------------------------------- token-level GAE


def test_observation_tokens_are_skipped_by_the_recursion():
    """★ A reward parked on a non-output token must not enter the return, and the
    discount must not count that position as a step."""
    cfg = _Cfg()
    mask = [[1, 0, 1]]
    adv, ret = TOKEN_GAE(
        batch=_batch([[0.0, 100.0, 1.0]], mask), non_tensor_batch=_nt(["g"], [0], [0]), config=cfg
    )


    # gamma=lam=1, zero values: both output tokens see a return of 1.0, not 101.0.
    assert _tokens(ret, mask) == pytest.approx([1.0, 1.0])
    # Nothing is written where the model did not produce a token.
    assert float(adv[0, 1]) == 0.0 and float(ret[0, 1]) == 0.0


def test_recursion_carries_across_a_row_boundary():
    """A reward in the last turn has to reach the first turn's tokens, which is the
    part a per-row estimator cannot do."""
    cfg = _Cfg()
    mask = [[1, 1], [1, 1]]
    _, ret = TOKEN_GAE(
        batch=_batch([[0.0, 0.0], [0.0, 1.0]], mask),
        non_tensor_batch=_nt(["g", "g"], [0, 0], [0, 1]),
        config=cfg,
    )

    assert float(ret[0, 0]) == pytest.approx(1.0), "reward did not propagate to the earlier turn"


def test_discounting_applies_per_token_not_per_turn():
    cfg = _Cfg()
    cfg.gamma, cfg.lam = 0.5, 1.0
    mask = [[1, 1]]
    _, ret = TOKEN_GAE(batch=_batch([[0.0, 1.0]], mask), non_tensor_batch=_nt(["g"], [0], [0]), config=cfg)

    assert _tokens(ret, mask) == pytest.approx([0.5, 1.0])


def test_separate_trajectories_do_not_leak_into_each_other():
    cfg = _Cfg()
    mask = [[1, 1], [1, 1]]
    _, ret = TOKEN_GAE(
        batch=_batch([[0.0, 0.0], [0.0, 1.0]], mask),
        non_tensor_batch=_nt(["g", "g"], [0, 1], [0, 0]),  # different traj_idx
        config=cfg,
    )

    assert float(ret[0, 0]) == 0.0, "reward leaked across trajectories"


# --------------------------------------------------------------------------- GRPO


def test_grpo_gives_every_token_of_a_trajectory_the_same_advantage():
    cfg = _Cfg()
    mask = [[1, 1], [1, 1]]
    adv, _ = TRAJ_GRPO(
        batch=_batch([[0.0, 0.0], [0.0, 1.0]], mask),
        non_tensor_batch=_nt(["g", "g", "g", "g"][:2], [0, 0], [0, 1]),
        config=cfg,
    )
    values = _tokens(adv, mask)

    assert len(set(values)) == 1, f"advantage varies within one trajectory: {values}"


def test_grpo_normalises_within_the_prompt_group():
    cfg = _Cfg()
    mask = [[1], [1]]
    adv, _ = TRAJ_GRPO(
        batch=_batch([[1.0], [0.0]], mask), non_tensor_batch=_nt(["g", "g"], [0, 1], [0, 0]), config=cfg
    )

    assert float(adv[0, 0]) > 0 > float(adv[1, 0])
    assert float(adv[0, 0]) == pytest.approx(-float(adv[1, 0]))


def test_grpo_group_with_no_spread_yields_zeros_not_nan():
    """★ Dividing by a zero std would poison the whole update with NaN."""
    cfg = _Cfg()
    adv, _ = TRAJ_GRPO(
        batch=_batch([[1.0], [1.0]], [[1], [1]]), non_tensor_batch=_nt(["g", "g"], [0, 1], [0, 0]), config=cfg
    )

    assert torch.isfinite(adv).all()
    assert float(adv.abs().sum()) == 0.0


def test_grpo_groups_are_independent():
    cfg = _Cfg()
    adv, _ = TRAJ_GRPO(
        batch=_batch([[1.0], [0.0], [5.0], [5.0]], [[1], [1], [1], [1]]),
        non_tensor_batch=_nt(["a", "a", "b", "b"], [0, 1, 0, 1], [0, 0, 0, 0]),
        config=cfg,
    )

    assert float(adv[2, 0]) == 0.0 and float(adv[3, 0]) == 0.0, "group b has no spread"
    assert float(adv[0, 0]) != 0.0


# ------------------------------------------------------------------- padding duplicates


def test_padded_duplicate_rows_are_not_counted_twice():
    """★ pad_dataproto_to_divisor repeats rows to reach a multiple of the world size.
    Scoring a duplicate again would double-count it in the backward recursion."""
    cfg = _Cfg()
    # Two output tokens: masked_whiten rejects a batch with a single unmasked element.
    real = TOKEN_GAE(
        batch=_batch([[0.0, 1.0]], [[1, 1]]), non_tensor_batch=_nt(["g"], [0], [0]), config=cfg
    )
    padded = TOKEN_GAE(
        batch=_batch([[0.0, 1.0], [0.0, 1.0]], [[1, 1], [1, 1]]),
        non_tensor_batch=_nt(["g", "g"], [0, 0], [0, 0]),  # identical triple
        config=cfg,
    )

    assert padded[1][0].tolist() == pytest.approx(real[1][0].tolist())
    assert padded[1][1].tolist() == pytest.approx(real[1][0].tolist()), "duplicate must mirror the original"


# ----------------------------------------------------------------------- the view


def test_view_orders_rows_by_turn_regardless_of_row_order():
    """★ Batch order is not turn order -- balancing and chunking reshuffle rows."""
    mask = torch.ones(3, 1, dtype=torch.long)
    view = TrajectoryView.build(mask, _nt(["g", "g", "g"], [0, 0, 0], [2, 0, 1]))

    assert len(view.trajectories) == 1
    turns = [int(view.rows[i]) for i in view.trajectories[0]]
    assert turns == [1, 2, 0], f"expected rows sorted by turn_idx, got {turns}"


def test_view_separates_trajectories_and_groups():
    mask = torch.ones(4, 1, dtype=torch.long)
    view = TrajectoryView.build(mask, _nt(["a", "a", "b", "b"], [0, 1, 0, 0], [0, 0, 0, 1]))

    assert sorted(len(t) for t in view.trajectories) == [1, 1, 2]


def test_view_reports_an_empty_row_as_having_no_last_position():
    mask = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
    view = TrajectoryView.build(mask, _nt(["g", "g"], [0, 1], [0, 0]))

    assert view.last_pos.tolist() == [-1, 1]
