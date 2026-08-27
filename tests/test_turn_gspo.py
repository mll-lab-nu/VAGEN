"""``turn_gspo`` must be GSPO with the turn as the sequence -- provably, at both limits.

The whole file rests on two claims that can be checked exactly rather than argued:

* **one turn per row** ==> elementwise identical to verl's own ``gspo``. If it is not,
  either the turn grouping or the aggregation is wrong, and both would otherwise produce
  numbers that look entirely reasonable.
* **every token its own turn** ==> the ratio is each token's own ratio and the clipping is
  vanilla PPO's. A geometric mean over one element is that element, so anything else means
  the grouping is leaking across turns.

Everything else this loss does lives between those two points, so if both hold the
machinery is doing what it says. A test that only asserted "the loss is finite and the
gradient is not zero" would pass with the turn boundaries read off by mistake from the
row, which is precisely the bug ``turn_gspo`` exists to fix.
"""

from __future__ import annotations

import pytest
import torch

import vagen.algorithms  # noqa: F401
import vagen.training.losses  # noqa: F401
from vagen.training.losses.turn_gspo import (
    aggregate_seq_mean_turn_sum_token_mean,
    compute_policy_loss_turn_gspo,
    turn_geometric_mean_log_ratio,
)


def _Cfg(global_batch_size=None, dp_size=1):
    """A real ActorConfig, bypassing __init__.

    Not a duck type: both this loss and verl's ``gspo`` assert ``isinstance(config,
    ActorConfig)``, and the whole point of the comparison test is to call them with the
    same object.
    """
    from verl.workers.config import ActorConfig

    cfg = object.__new__(ActorConfig)
    cfg.clip_ratio = 0.2
    cfg.clip_ratio_low = None
    cfg.clip_ratio_high = None
    cfg.clip_ratio_c = 3.0
    cfg.loss_agg_mode = "seq-mean-token-mean"
    cfg.loss_scale_factor = None
    cfg.global_batch_info = {
        "dp_size": dp_size,
        "batch_num_tokens": None,
        "global_batch_size": global_batch_size,
        "loss_scale_factor": None,
    }
    return cfg


def _batch(rows=3, width=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    old = torch.randn((rows, width), generator=g) * 0.1
    new = old + torch.randn((rows, width), generator=g) * 0.05
    adv = torch.randn((rows, width), generator=g)
    mask = torch.ones((rows, width), dtype=torch.bool)
    mask[:, -2:] = False              # trailing padding, as a real batch has
    return old, new, adv, mask


# ------------------------------------------------------------------- the grouping

def test_the_geometric_mean_is_taken_within_a_turn_and_not_across_them():
    log_ratio = torch.tensor([[1.0, 3.0, 10.0, 20.0]])
    turn_id = torch.tensor([[0, 0, 1, 1]])
    mask = torch.ones((1, 4), dtype=torch.bool)

    means, lengths = turn_geometric_mean_log_ratio(log_ratio, turn_id, mask)
    assert means[0].tolist() == [2.0, 2.0, 15.0, 15.0]
    assert lengths[0].tolist() == [2.0, 2.0, 2.0, 2.0]


def test_masked_out_tokens_do_not_enter_the_mean_or_the_length():
    """★ Observation tokens sit inside a turn's span under concat. Counting them would
    shrink ``1 / L_t`` by however verbose the *environment* was, so a turn's step size
    would depend on the length of the text it was replying to."""
    log_ratio = torch.tensor([[1.0, 999.0, 3.0]])
    # ★ Labelled as part of turn 0 *and* masked out, which is the only arrangement that
    # tests anything: the estimators write -1 at masked positions, so a masked token with
    # turn_id -1 lands in its own bucket and the mask never has to do any work. The mask
    # is what must win when the two disagree -- the turn_id column comes from whichever
    # estimator is configured, and the mask is the ground truth about what the model said.
    turn_id = torch.tensor([[0, 0, 0]])
    mask = torch.tensor([[True, False, True]])

    means, lengths = turn_geometric_mean_log_ratio(log_ratio, turn_id, mask)
    assert means[0][0].item() == pytest.approx(2.0), "the masked token entered the mean"
    assert lengths[0][0].item() == pytest.approx(2.0), "the masked token was counted"


def test_turns_are_grouped_per_row_so_two_rows_may_reuse_an_id():
    log_ratio = torch.tensor([[1.0, 3.0], [100.0, 300.0]])
    turn_id = torch.tensor([[0, 0], [0, 0]])
    mask = torch.ones((2, 2), dtype=torch.bool)

    means, _ = turn_geometric_mean_log_ratio(log_ratio, turn_id, mask)
    assert means[0][0].item() == pytest.approx(2.0)
    assert means[1][0].item() == pytest.approx(200.0), "rows were pooled"


# --------------------------------------------------------------- limit 1: verl's gspo

def test_one_turn_per_row_is_elementwise_verls_gspo():
    """★ The first exact limit. verl's gspo takes the geometric mean over a whole row; if
    a row holds exactly one turn those are the same set of tokens, so every intermediate
    tensor must match, not merely the scalar."""
    from verl.trainer.ppo.core_algos import compute_policy_loss_gspo

    old, new, adv, mask = _batch()
    turn_id = torch.where(mask, torch.zeros_like(adv, dtype=torch.long), -1)
    cfg = _Cfg(global_batch_size=mask.any(dim=-1).sum().item())

    ours, our_metrics = compute_policy_loss_turn_gspo(
        old_log_prob=old, log_prob=new, advantages=adv, response_mask=mask,
        config=cfg, turn_id=turn_id,
    )
    theirs, their_metrics = compute_policy_loss_gspo(
        old_log_prob=old, log_prob=new, advantages=adv, response_mask=mask, config=cfg,
    )

    assert ours.item() == pytest.approx(theirs.item(), rel=1e-6), (
        f"turn_gspo {ours.item()} != verl gspo {theirs.item()} at one turn per row"
    )
    assert our_metrics["actor/pg_clipfrac"] == pytest.approx(their_metrics["actor/pg_clipfrac"])
    assert our_metrics["actor/ppo_kl"] == pytest.approx(their_metrics["actor/ppo_kl"])
    assert our_metrics["actor/turns_per_row"] == pytest.approx(1.0)


def test_several_turns_per_row_is_not_verls_gspo():
    """The limit above is only evidence if the two disagree away from it -- otherwise it
    would be satisfied by a `turn_gspo` that ignored `turn_id` entirely."""
    from verl.trainer.ppo.core_algos import compute_policy_loss_gspo

    old, new, adv, mask = _batch()
    turn_id = torch.where(mask, torch.tensor([0, 0, 0, 1, 1, 1, -1, -1]), -1)
    cfg = _Cfg(global_batch_size=3)

    ours, _ = compute_policy_loss_turn_gspo(
        old_log_prob=old, log_prob=new, advantages=adv, response_mask=mask,
        config=cfg, turn_id=turn_id,
    )
    theirs, _ = compute_policy_loss_gspo(
        old_log_prob=old, log_prob=new, advantages=adv, response_mask=mask, config=cfg,
    )
    assert abs(ours.item() - theirs.item()) > 1e-4, (
        "two turns per row gave verl's row-level answer -- turn_id is being ignored"
    )


# ------------------------------------------------------------- limit 2: vanilla PPO

def test_one_token_per_turn_gives_each_token_its_own_ratio():
    """★ The second exact limit. A geometric mean over a single element is that element,
    so the importance ratio must be exactly ``exp(log_prob - old_log_prob)`` -- vanilla
    PPO's -- and the clip must fire on exactly the same tokens."""
    old, new, adv, mask = _batch()
    per_token = torch.arange(adv.shape[1]).expand_as(adv).clone()
    turn_id = torch.where(mask, per_token, torch.full_like(per_token, -1))

    log_ratio = new - old
    means, lengths = turn_geometric_mean_log_ratio(log_ratio, turn_id, mask)

    assert torch.allclose(means[mask], log_ratio[mask], atol=1e-6), (
        "the turn ratio is not the token's own ratio when each turn holds one token"
    )
    assert torch.all(lengths[mask] == 1.0)

    # And so the clip decision matches vanilla PPO's, token for token.
    ratio = torch.exp(log_ratio)
    vanilla_clipped = (ratio > 1.2) | (ratio < 0.8)
    ours_clipped = (torch.exp(means) > 1.2) | (torch.exp(means) < 0.8)
    assert torch.equal(vanilla_clipped[mask], ours_clipped[mask])


# --------------------------------------------------------------- the aggregation

def test_the_aggregation_is_seq_mean_of_turn_sum_of_token_mean():
    """Computed the slow, obvious way and compared with the vectorised one."""
    loss = torch.tensor([[2.0, 4.0, 10.0, 20.0, 0.0], [1.0, 1.0, 7.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    turn_id = torch.tensor([[0, 0, 1, 1, -1], [0, 0, 1, -1, -1]])
    _, lengths = turn_geometric_mean_log_ratio(torch.zeros_like(loss), turn_id, mask)

    got = aggregate_seq_mean_turn_sum_token_mean(loss, mask, lengths, global_batch_size=2)

    # row 0: turn 0 mean (2+4)/2 = 3, turn 1 mean (10+20)/2 = 15  -> sum 18
    # row 1: turn 0 mean (1+1)/2 = 1, turn 1 mean 7/1 = 7         -> sum 8
    assert got.item() == pytest.approx((18.0 + 8.0) / 2)


def test_a_long_turn_does_not_get_more_gradient_than_a_short_one():
    """★ What the ``1 / L_t`` is for. Without it a turn's contribution grows with its
    length, so the model can enlarge its own update by writing more rather than by
    writing better -- and length is far easier to change than correctness."""
    short = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    long = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    mask_s = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    mask_l = torch.ones((1, 4), dtype=torch.bool)
    ids_s = torch.tensor([[0, 0, -1, -1]])
    ids_l = torch.tensor([[0, 0, 0, 0]])

    _, len_s = turn_geometric_mean_log_ratio(torch.zeros_like(short), ids_s, mask_s)
    _, len_l = turn_geometric_mean_log_ratio(torch.zeros_like(long), ids_l, mask_l)

    a = aggregate_seq_mean_turn_sum_token_mean(short, mask_s, len_s, global_batch_size=1)
    b = aggregate_seq_mean_turn_sum_token_mean(long, mask_l, len_l, global_batch_size=1)
    assert a.item() == pytest.approx(b.item()), (
        f"a 4-token turn contributed {b.item()} where a 2-token one contributed "
        f"{a.item()}; the 1/L_t is missing"
    )


def test_an_episode_with_more_turns_contributes_more():
    """The counterpart. Across turns the policy gradient is a *sum* -- ten decisions are
    ten terms -- so turn-mean here would silently discount long episodes."""
    loss = torch.ones((1, 4))
    mask = torch.ones((1, 4), dtype=torch.bool)
    one_turn = torch.zeros((1, 4), dtype=torch.long)
    two_turns = torch.tensor([[0, 0, 1, 1]])

    _, l1 = turn_geometric_mean_log_ratio(loss, one_turn, mask)
    _, l2 = turn_geometric_mean_log_ratio(loss, two_turns, mask)
    a = aggregate_seq_mean_turn_sum_token_mean(loss, mask, l1, global_batch_size=1)
    b = aggregate_seq_mean_turn_sum_token_mean(loss, mask, l2, global_batch_size=1)
    assert b.item() == pytest.approx(2 * a.item())


def test_a_local_row_count_is_refused_under_data_parallelism():
    """Each rank would divide by its own row count and the reduced gradient would be a
    mean of means -- which is not the mean over the batch unless the ranks happen to hold
    equally many rows. Silent, and worth a few percent."""
    loss, mask = torch.ones((2, 3)), torch.ones((2, 3), dtype=torch.bool)
    _, lengths = turn_geometric_mean_log_ratio(loss, torch.zeros_like(loss, dtype=torch.long), mask)
    with pytest.raises(ValueError, match="global row count"):
        aggregate_seq_mean_turn_sum_token_mean(loss, mask, lengths, global_batch_size=None, dp_size=4)


# ------------------------------------------------------------------ the wiring

def test_the_loss_refuses_to_run_without_turn_ids():
    """★ Falling back to verl's `gspo` would be the bug this file exists to fix, wearing
    this file's name. Under concat that fallback treats a whole episode as one action."""
    old, new, adv, mask = _batch()
    with pytest.raises(ValueError, match="turn_id"):
        compute_policy_loss_turn_gspo(
            old_log_prob=old, log_prob=new, advantages=adv, response_mask=mask,
            config=_Cfg(global_batch_size=3), turn_id=None,
        )


def test_verl_passes_turn_id_through_to_a_loss_that_wants_it():
    """The column is useless if `ppo_loss` does not forward it, and `PolicyLossFn`'s
    signature has no such argument -- so this is decided by reflection, and reflection is
    the kind of thing that stops matching without failing."""
    import inspect

    from verl.workers.utils import losses

    src = inspect.getsource(losses.ppo_loss)
    assert 'has_turn_id = "turn_id" in data.keys()' in src
    assert 'inspect.signature(policy_loss_fn).parameters' in src
    assert "**extra_args" in src, "the argument is computed but never passed"
    assert "turn_id" in inspect.signature(compute_policy_loss_turn_gspo).parameters


def test_the_estimators_publish_the_column_the_loss_needs():
    """Both halves of the same contract, checked together: an estimator that stopped
    publishing would leave `turn_gspo` raising at the first step of a real run."""
    import inspect

    from vagen.algorithms._common import trajectory_algos

    src = inspect.getsource(trajectory_algos._Packed.emit)
    assert '"turn_id"' in src, "emit no longer publishes turn_id"


# ---------------------------------------------------- refusing a config that cannot work

def _trainer(loss_mode="turn_gspo", estimator="turn_level_gae", external_lib="vagen.training.losses"):
    from omegaconf import OmegaConf

    from vagen.training.trainer.mixin import VagenLogicMixin

    t = object.__new__(type("T", (VagenLogicMixin,), {}))
    t.config = OmegaConf.create({
        "algorithm": {"adv_estimator": estimator, "gamma": 1.0},
        "actor_rollout_ref": {
            "actor": {"policy_loss": {"loss_mode": loss_mode}},
            "model": {"external_lib": external_lib},
        },
    })
    return t


def test_a_workable_turn_gspo_config_is_accepted():
    _trainer()._vagen_check_turn_level_loss_has_what_it_needs()


def test_turn_gspo_without_the_worker_import_is_refused_here_not_minutes_in():
    """★ The registry lives in the actor worker's process, so registering `turn_gspo` on
    the driver proves nothing. Left unchecked the run comes up, rolls out, and dies on
    'Unsupported loss mode' at the first update."""
    with pytest.raises(ValueError, match="external_lib=vagen.training.losses"):
        _trainer(external_lib=None)._vagen_check_turn_level_loss_has_what_it_needs()


def test_turn_gspo_with_an_estimator_that_publishes_no_turns_is_refused():
    with pytest.raises(ValueError, match="turn_id"):
        _trainer(estimator="gae")._vagen_check_turn_level_loss_has_what_it_needs()


def test_other_losses_are_left_alone():
    """The check must not become a tax on every run: `vanilla` and verl's `gspo` need
    neither the column nor the import."""
    _trainer(loss_mode="vanilla", estimator="gae", external_lib=None)\
        ._vagen_check_turn_level_loss_has_what_it_needs()


def test_the_check_runs_at_startup():
    import inspect

    from vagen.training.trainer.mixin import VagenLogicMixin

    assert "_vagen_check_turn_level_loss_has_what_it_needs()" in inspect.getsource(
        VagenLogicMixin._vagen_init
    )


# ------------------------------------------- what the turn-sum costs, pinned

def test_the_ratio_is_the_geometric_mean_and_not_the_raw_product():
    """★ ``turn_new / turn_old`` written literally is ``prod_j r_j``, and that is *not*
    what this loss uses. GSPO substitutes its ``L_t``-th root.

    The substitution is the whole reason the loss is usable: ``log R_t`` is a sum of
    ``L_t`` token log-ratios, so its spread grows like ``sqrt(L_t)`` -- at 80 tokens and a
    per-token jitter of 0.02, ``R_t`` already covers ``[0.84, 1.20]`` against a clip range
    of ``[0.8, 1.2]``, and the clip stops discriminating. It is a different surrogate, not
    an unbiased estimate of the product objective, and this test exists so that nobody
    reads ``turn_gspo`` as computing the product.
    """
    delta = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    turn_id = torch.zeros((1, 4), dtype=torch.long)
    mask = torch.ones((1, 4), dtype=torch.bool)

    mean_log, lengths = turn_geometric_mean_log_ratio(delta, turn_id, mask)
    product = torch.exp(delta.sum()).item()          # the literal turn ratio
    used = torch.exp(mean_log[0, 0]).item()          # what turn_gspo clips

    assert product == pytest.approx(2.71828, rel=1e-4)
    assert used == pytest.approx(product ** (1 / 4), rel=1e-6)
    assert used == pytest.approx(1.28403, rel=1e-4)
    assert lengths[0, 0].item() == 4.0


def test_the_ratio_is_constant_within_a_turn_so_the_clip_is_one_decision():
    """The point of a turn-level ratio: the turn is accepted or rejected as a whole,
    rather than each of its tokens being judged separately -- which is the token-ratio
    noise GSPO exists to remove."""
    old, new, _, mask = _batch(rows=1, width=8)
    turn_id = torch.where(mask, torch.tensor([0, 0, 0, 0, 1, 1, -1, -1]), -1)

    means, _ = turn_geometric_mean_log_ratio(new - old, turn_id, mask)
    assert len(set(means[0, :4].tolist())) == 1, "the first turn's tokens disagree"
    assert len(set(means[0, 4:6].tolist())) == 1, "the second turn's tokens disagree"
    assert means[0, 0].item() != pytest.approx(means[0, 4].item()), "the two turns agree"


def test_the_pg_term_grows_with_the_turn_count_while_the_regularisers_do_not():
    """★ The hyperparameter hazard, pinned so it cannot change silently.

    verl aggregates entropy and KL with ``loss_agg_mode`` in a separate call, so they stay
    at O(1) while this loss returns something proportional to the turn count. That divides
    the *effective* ``entropy_coeff`` and ``kl_loss_coef`` by the mean turns per row --
    and that mean moves as the policy's verbosity moves.

    This is not a bug to fix here (turn-sum is what the policy gradient over an episode's
    actions is); it is a number the caller has to know. If this test starts failing, the
    advice in the module docstring is wrong and has to be rewritten with it.
    """
    from verl.trainer.ppo.core_algos import agg_loss

    loss = torch.ones((2, 20))
    mask = torch.ones((2, 20), dtype=torch.bool)
    regulariser = agg_loss(
        loss_mat=loss, loss_mask=mask, loss_agg_mode="token-mean",
        batch_num_tokens=int(mask.sum()),
    ).item()

    for turns in (1, 2, 5, 10):
        turn_id = (torch.arange(20) // (20 // turns)).expand(2, 20).contiguous()
        _, lengths = turn_geometric_mean_log_ratio(loss, turn_id, mask)
        pg = aggregate_seq_mean_turn_sum_token_mean(
            loss, mask, lengths, global_batch_size=2
        ).item()
        assert pg / regulariser == pytest.approx(float(turns), rel=1e-5), (
            f"{turns} turns/row gave a pg:regulariser ratio of {pg / regulariser:.3f}, "
            f"not {turns}; the docstring's coefficient advice is now wrong"
        )


def test_turns_per_row_is_reported_because_it_is_that_factor():
    """The metric is not decoration -- it is the number the caller needs to rescale the
    other two coefficients by."""
    old, new, adv, mask = _batch(rows=2, width=8)
    turn_id = torch.where(mask, torch.tensor([0, 0, 0, 1, 1, 1, -1, -1]), -1)
    _, metrics = compute_policy_loss_turn_gspo(
        old_log_prob=old, log_prob=new, advantages=adv, response_mask=mask,
        config=_Cfg(global_batch_size=2), turn_id=turn_id,
    )
    assert metrics["actor/turns_per_row"] == pytest.approx(2.0)
    assert metrics["actor/turn_len_mean"] == pytest.approx(3.0)


# ================================================================= turn_ppo

from vagen.training.losses.turn_gspo import compute_policy_loss_turn_ppo, turn_log_ratio  # noqa: E402


def test_turn_ppo_clips_the_literal_product_ratio():
    """★ ``turn_ppo`` is the objective ``turn_gspo`` approximates: it clips
    ``R_t = prod_j r_j``, the actual ``pi(a_t|s_t) / pi_old(a_t|s_t)``, with no root."""
    delta = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    turn_id = torch.zeros((1, 4), dtype=torch.long)
    mask = torch.ones((1, 4), dtype=torch.bool)

    summed, _ = turn_log_ratio(delta, turn_id, mask, reduce="sum")
    assert torch.exp(summed[0, 0]).item() == pytest.approx(torch.exp(delta.sum()).item(), rel=1e-6)
    assert torch.exp(summed[0, 0]).item() == pytest.approx(2.71828, rel=1e-4)


def test_the_two_turn_losses_differ_by_exactly_the_length_root():
    """★ The single relationship that defines the pair: ``log R_t = L_t * log s_t``. If
    this holds, ``turn_ppo`` and ``turn_gspo`` really are the exact objective and its
    geometric-mean surrogate, and not two unrelated things."""
    old, new, _, mask = _batch(rows=2, width=8)
    turn_id = torch.where(mask, torch.tensor([0, 0, 0, 1, 1, 1, -1, -1]), -1)
    delta = new - old

    geo, lengths = turn_log_ratio(delta, turn_id, mask, reduce="mean")
    prod, _ = turn_log_ratio(delta, turn_id, mask, reduce="sum")
    assert torch.allclose(prod[mask], (geo * lengths)[mask], atol=1e-6)


def test_turn_ppos_ratio_spread_grows_with_length_while_gspos_shrinks():
    """★ Why GSPO exists, measured rather than asserted. Same per-token jitter, longer
    turns: ``std(log R_t)`` grows like ``sqrt(L)`` and ``std(log s_t)`` falls like
    ``1/sqrt(L)``. Once the first passes ``log(1 + eps) ~ 0.18`` the clip stops
    discriminating -- which is the whole reason ``turn_ppo`` needs a wider clip range.
    """
    g = torch.Generator().manual_seed(7)
    spreads = {}
    for L in (4, 16, 64, 256):
        delta = torch.randn((256, L), generator=g) * 0.02
        turn_id = torch.zeros((256, L), dtype=torch.long)
        mask = torch.ones((256, L), dtype=torch.bool)
        prod, _ = turn_log_ratio(delta, turn_id, mask, reduce="sum")
        geo, _ = turn_log_ratio(delta, turn_id, mask, reduce="mean")
        spreads[L] = (prod[:, 0].std().item(), geo[:, 0].std().item())

    ppo = [spreads[L][0] for L in (4, 16, 64, 256)]
    gspo = [spreads[L][1] for L in (4, 16, 64, 256)]
    assert ppo[0] < ppo[1] < ppo[2] < ppo[3], f"turn_ppo spread did not grow: {ppo}"
    assert gspo[0] > gspo[1] > gspo[2] > gspo[3], f"turn_gspo spread did not shrink: {gspo}"
    # Each 4x in length should move both by ~2x, in opposite directions.
    assert ppo[3] / ppo[0] == pytest.approx(8.0, rel=0.25)
    assert gspo[0] / gspo[3] == pytest.approx(8.0, rel=0.25)
    # And at L=256 the product ratio's spread is well past the clip range's log(1.2).
    assert ppo[3] > 0.18 > gspo[3]


def test_turn_ppo_at_one_token_per_turn_is_vanilla_ppos_ratio():
    """★ ``turn_ppo``'s own exact limit. A product over one element is that element, so
    with each token its own turn the ratio must be exactly vanilla PPO's."""
    old, new, _, mask = _batch()
    per_token = torch.arange(mask.shape[1]).expand_as(mask).clone()
    turn_id = torch.where(mask, per_token, torch.full_like(per_token, -1))

    delta = new - old
    prod, lengths = turn_log_ratio(delta, turn_id, mask, reduce="sum")
    assert torch.allclose(prod[mask], delta[mask], atol=1e-6)
    assert torch.all(lengths[mask] == 1.0)


def test_turn_ppo_sums_where_turn_gspo_averages_in_the_aggregation():
    """★ The coupling that makes this one implementation rather than two. ``turn_ppo``
    clips ``R_t``, whose gradient carries no ``1 / L_t``, so its tokens must be *summed*;
    ``turn_gspo`` clips ``s_t``, whose gradient does, so its tokens are *averaged*. A pair
    that mismatched here would still train -- on the gradient of nothing in particular."""
    old, new, adv, mask = _batch(rows=1, width=8)
    turn_id = torch.where(mask, torch.zeros_like(adv, dtype=torch.long), -1)
    cfg = _Cfg(global_batch_size=1)

    gspo, _ = compute_policy_loss_turn_gspo(
        old_log_prob=old, log_prob=new, advantages=adv, response_mask=mask,
        config=cfg, turn_id=turn_id,
    )
    ppo, m = compute_policy_loss_turn_ppo(
        old_log_prob=old, log_prob=new, advantages=adv, response_mask=mask,
        config=cfg, turn_id=turn_id,
    )
    # One turn of 6 tokens: turn_ppo sums the same-shaped matrix turn_gspo averages, so
    # the aggregation alone accounts for a factor of L even before the ratios differ.
    assert m["actor/turn_len_mean"] == pytest.approx(6.0)
    assert abs(ppo.item()) > abs(gspo.item())


def test_turn_ppo_reports_the_spread_that_says_whether_the_clip_range_is_usable():
    old, new, adv, mask = _batch(rows=2, width=8)
    turn_id = torch.where(mask, torch.tensor([0, 0, 0, 1, 1, 1, -1, -1]), -1)
    _, m = compute_policy_loss_turn_ppo(
        old_log_prob=old, log_prob=new, advantages=adv, response_mask=mask,
        config=_Cfg(global_batch_size=2), turn_id=turn_id,
    )
    assert "actor/turn_log_ratio_std" in m and m["actor/turn_log_ratio_std"] >= 0.0
    assert m["actor/turns_per_row"] == pytest.approx(2.0)


def test_turn_ppo_also_refuses_without_turn_ids():
    old, new, adv, mask = _batch()
    with pytest.raises(ValueError, match=r"turn_ppo needs a `turn_id`"):
        compute_policy_loss_turn_ppo(
            old_log_prob=old, log_prob=new, advantages=adv, response_mask=mask,
            config=_Cfg(global_batch_size=3), turn_id=None,
        )


def test_both_turn_losses_are_registered():
    from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY

    assert "turn_gspo" in POLICY_LOSS_REGISTRY
    assert "turn_ppo" in POLICY_LOSS_REGISTRY
