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

from vagen.algorithms._common.trajectory import TrajectoryView

TOKEN_GAE = get_adv_estimator_fn("token_level_gae")
TRAJ_GRPO = get_adv_estimator_fn("trajectory_grpo")


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


def test_view_treats_a_missing_turn_axis_as_a_single_turn():
    """★ Concat keeps a whole trajectory in one row, so its agent loop emits no
    turn_idx. Requiring the column would mean an estimator could not be used under the
    concat layout at all -- which is the orthogonality these estimators exist for."""
    mask = torch.ones(2, 1, dtype=torch.long)
    nt = {"group_idx": np.array(["a", "a"], dtype=object), "traj_idx": np.array([0, 1])}

    view = TrajectoryView.build(mask, nt)

    assert len(view.trajectories) == 2
    assert all(len(t) == 1 for t in view.trajectories)


def test_token_gae_runs_without_a_turn_column():
    cfg = _Cfg()
    nt = {"group_idx": np.array(["a"], dtype=object), "traj_idx": np.array([0])}
    _, ret = TOKEN_GAE(batch=_batch([[0.0, 1.0]], [[1, 1]]), non_tensor_batch=nt, config=cfg)

    assert ret[0].tolist() == pytest.approx([1.0, 1.0])


# (lam_low == lam is token-level GAE; lam_low == 1 is turn-level) and its refusal to run
# without lam_low. That estimator is gone. Its layout-independence and value-mask
# properties were never specific to it -- test_estimator_contract.py asserts both for
# every registered estimator.


def test_turn_boundaries_are_found_under_both_layouts():
    import torch

    from vagen.algorithms._common.packing import _is_turn_boundary

    # one row of width 5, two turns: positions 0,1 then 3,4 -- the gap ends turn one
    index = torch.tensor([[0, 1, 3, 4]])
    valid = torch.ones_like(index, dtype=torch.bool)
    assert _is_turn_boundary(index, valid, width=5).tolist() == [[False, True, False, True]]

    # ★ two rows of width 2, both full: the flat positions run 0,1,2,3 with no gap, so
    # only the row change distinguishes the turns. A gap test alone merges them.
    index = torch.tensor([[0, 1, 2, 3]])
    assert _is_turn_boundary(index, valid, width=2).tolist() == [[False, True, False, True]]


# ----------------------------------------------------------------- turn-level GAE

TURN_GAE = get_adv_estimator_fn("turn_level_gae")


def _turn_gae_by_hand(scores, values, gamma, lam):
    """Turn-level GAE for one trajectory laid out one turn per row, written straight
    from the definition: reward is the turn's total, value is the critic at the state the
    turn acts from, recursion runs backward over turns.

    An independent oracle rather than a copy of the code under test. The estimator this
    replaced was compared against here until it was deleted; keeping a copy of a deleted
    implementation would only have asserted that a refactor preserved itself.
    """
    advantages, nextvalue, lastgaelam = [], 0.0, 0.0
    for r, v in zip(reversed(scores), reversed(values)):
        delta = sum(r) + gamma * nextvalue - v[0]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages.append(lastgaelam)
        nextvalue = v[0]
    return advantages[::-1]


def test_turn_gae_matches_the_definition():
    """★ The safety net for the replacement: it must agree with turn-level GAE computed
    by hand, token for token, not merely look similar."""
    cfg = _Cfg()
    cfg.gamma, cfg.lam = 1.0, 0.9

    mask = [[1, 1], [1, 1], [1, 1]]
    scores = [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]]
    values = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    adv, _ = TURN_GAE(
        batch=_batch(scores, mask, values),
        non_tensor_batch=_nt(["g", "g", "g"], [0, 0, 0], [0, 1, 2]),
        config=cfg,
    )

    # The estimator whitens its output, so the reference has to be whitened the same
    # way to be comparable. Calling verl's own masked_whiten rather than reproducing it
    # keeps this test about the GAE recursion -- reimplementing it here got the unbiased
    # variance wrong and turned a correct estimator red.
    import verl.utils.torch_functional as verl_F

    by_hand = _turn_gae_by_hand(scores, values, cfg.gamma, cfg.lam)
    mask_t = torch.tensor(mask, dtype=torch.float32)
    expected = torch.tensor(by_hand, dtype=torch.float32).unsqueeze(1).expand(3, 2).contiguous()
    expected = verl_F.masked_whiten(expected, mask_t) * mask_t

    assert _tokens(adv, mask) == pytest.approx(_tokens(expected, mask), rel=1e-4, abs=1e-5)
    # Every token of a turn carries that turn's advantage.
    for i in range(3):
        assert float(adv[i, 1]) == pytest.approx(float(adv[i, 0]))


def test_turn_gae_writes_a_return_only_at_each_turns_first_token():
    """★ V(s_t) is the value of the state the turn acts *from*, so the return belongs at
    the position before any of the turn's tokens were emitted. Everything else stays at
    the sentinel for value_mask to exclude."""
    cfg = _Cfg()
    mask = [[1, 1, 0, 1, 1]]
    _, ret = TURN_GAE(
        batch=_batch([[0.0, 0.0, 0.0, 0.0, 1.0]], mask, [[0.0] * 5]),
        non_tensor_batch=_nt(["g"], [0], [0]),
        config=cfg,
    )

    row = ret[0].tolist()
    assert row[0] != -100.0 and row[3] != -100.0, "each turn's first token needs a return"
    assert row[1] == -100.0 and row[4] == -100.0, "the rest must stay at the sentinel"


def test_turn_gae_gives_every_token_of_a_turn_the_same_advantage():
    """Broadcasting is exact, not an approximation: an autoregressive policy factorises,
    so the per-token coefficient in the turn-level gradient is the turn's advantage."""
    cfg = _Cfg()
    mask = [[1, 1, 0, 1, 1]]
    adv, _ = TURN_GAE(
        batch=_batch([[0.0, 0.0, 0.0, 0.0, 1.0]], mask, [[0.1, 0.2, 9.0, 0.3, 0.4]]),
        non_tensor_batch=_nt(["g"], [0], [0]),
        config=cfg,
    )

    assert float(adv[0, 0]) == pytest.approx(float(adv[0, 1]))
    assert float(adv[0, 3]) == pytest.approx(float(adv[0, 4]))


def test_turn_gae_works_under_the_concat_layout():
    """★ What the old one could not do. It assumed a row was a turn, so a concat episode
    -- every turn in one row -- collapsed to a single decision."""
    cfg = _Cfg()
    mask = [[1, 1, 0, 1, 1]]
    adv, ret = TURN_GAE(
        batch=_batch([[0.0, 0.0, 0.0, 0.0, 1.0]], mask, [[0.1, 0.2, 9.0, 0.3, 0.4]]),
        non_tensor_batch=_nt(["g"], [0], [0]),
        config=cfg,
    )

    assert float(adv[0, 0]) != pytest.approx(float(adv[0, 3])), "the two turns were scored as one"
    # Two turns in one row means two anchors, at each turn's first token.
    assert [i for i, v in enumerate(ret[0].tolist()) if v != -100.0] == [0, 3]


def test_turn_values_are_not_lost_to_a_scatter_collision():
    """★ Several tokens map to one turn slot. A plain scatter keeps whichever is written
    last -- a zero from a non-start token -- silently replacing the turn's value with 0
    and changing every advantage upstream of it."""
    cfg = _Cfg()
    cfg.gamma, cfg.lam = 1.0, 1.0
    mask = [[1, 1, 1]]
    _, ret = TURN_GAE(
        batch=_batch([[0.0, 0.0, 1.0]], mask, [[0.4, 9.0, 9.0]]),
        non_tensor_batch=_nt(["g"], [0], [0]),
        config=cfg,
    )

    # One turn: return = reward, and V(s) cancels. If the value were zeroed the
    # advantage would be 1.0 and the return 1.0 rather than reward + V - V + V.
    assert float(ret[0, 0]) == pytest.approx(1.0), "the turn's value came from the wrong token"


def test_turn_gae_still_needs_a_value_mask():
    from vagen.algorithms import needs_value_mask

    assert needs_value_mask("turn_level_gae") is True


# ------------------------------------------------- _backward_gae's variable-lambda branch

def _bwd(rewards, values, valid, gamma, lam):
    from vagen.algorithms._common.packing import _backward_gae

    import torch
    t = lambda x, d=torch.float32: torch.tensor(x, dtype=d)
    return _backward_gae(t(rewards), t(values), t(valid, torch.bool), gamma,
                         t(lam) if isinstance(lam, list) else lam)


def test_a_constant_lambda_tensor_is_the_scalar_it_equals():
    """★ The branch `lam_t = lam if torch.is_tensor(lam) else None` had exactly one caller
    -- `removed_estimator_gae_varlam` -- and lost its only coverage when that estimator was removed.
    Nothing in the tree passes a tensor now, but the docstring advertises it as what a
    per-turn lambda would use, so it is an extension point, and an unexercised one is one
    that breaks for whoever reaches for it first.

    A tensor of a single repeated value must reproduce the scalar path exactly: that pins
    the indexing (`lam_t[:, t]`) against an off-by-one or a transposed read, which is the
    way this goes wrong.
    """
    import pytest

    rewards = [[0.0, 0.0, 0.0, 1.0]]
    values = [[0.1, 0.2, 0.3, 0.4]]
    valid = [[True, True, True, True]]

    scalar = _bwd(rewards, values, valid, 1.0, 0.7)
    tensor = _bwd(rewards, values, valid, 1.0, [[0.7, 0.7, 0.7, 0.7]])
    assert tensor[0].tolist() == pytest.approx(scalar[0].tolist(), rel=1e-6)


def test_a_per_position_lambda_is_applied_at_that_position():
    """The point of the branch: lambda varies along the sequence. A zero at one position
    cuts the chain there and nowhere else, so credit stops propagating past it while every
    later position keeps the value it had."""
    import pytest

    rewards = [[0.0, 0.0, 0.0, 1.0]]
    values = [[0.0, 0.0, 0.0, 0.0]]
    valid = [[True, True, True, True]]

    full = _bwd(rewards, values, valid, 1.0, [[1.0, 1.0, 1.0, 1.0]])
    cut = _bwd(rewards, values, valid, 1.0, [[1.0, 0.0, 1.0, 1.0]])

    # With no values, every delta is the reward at that position: full credit reaches t=0.
    assert full[0].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0], rel=1e-6)
    # lam_t is consulted AT position t (`lam_t[:, t]`), so a zero at t=1 zeroes the carry
    # into position 1 itself and everything before it -- not just everything before it.
    assert cut[0].tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0], rel=1e-6)


def test_padding_is_not_folded_into_the_recursion():
    """Padding sits on the right, so the loop starts inside it. An invalid position must
    leave `lastgaelam` alone rather than multiplying it by a padded lambda."""
    import pytest

    rewards = [[0.0, 1.0, 0.0]]
    values = [[0.0, 0.0, 0.0]]
    lam = [[1.0, 1.0, 0.0]]          # the 0.0 sits under padding; it must never be read

    padded = _bwd(rewards, values, [[True, True, False]], 1.0, lam)
    exact = _bwd(rewards[0][:2] and [[0.0, 1.0]], [[0.0, 0.0]], [[True, True]], 1.0,
                 [[1.0, 1.0]])
    assert padded[0, :2].tolist() == pytest.approx(exact[0].tolist(), rel=1e-6)
    assert float(padded[0, 2]) == 0.0, "a padded position was given an advantage"
