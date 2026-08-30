"""Turn-level policy losses whose unit of action is neither a token nor a row.

★ What this is for. ``turn_level_gae`` says a turn is one decision. That claim is only
half-made if the *advantage* is turn-level while the importance ratio and the clip stay
per-token -- the objective then still asks each token to be individually acceptable, and
a handful of outlier tokens can reshape a whole turn's gradient. This is the other half.

The derivation, and the one place it is not exact
-------------------------------------------------
Written out honestly, "a turn is one action" gives::

    R_t(theta) = pi_theta(a_t | s_t) / pi_old(a_t | s_t) = prod_j r_{t,j}

and PPO's clipped objective on that ratio. It is the correct objective and it is
numerically unusable: ``log R_t`` is a sum of ``L_t`` token log-ratios, so its standard
deviation grows like ``sqrt(L_t)``. At ``L_t = 512`` and a per-token jitter of 0.02,
``R_t`` swings over roughly ``[0.64, 1.57]`` while the clip range is ``[0.8, 1.2]`` --
the clip either always fires or never does, and the fraction of samples with a non-zero
gradient collapses.

GSPO (arXiv:2507.18071) replaces it with the geometric mean::

    s_t(theta) = R_t(theta) ** (1 / L_t)

whose log-variance *falls* with length instead of rising, and lands on the same scale for
long and short turns -- so one epsilon means the same thing for both. This is **not** an
unbiased estimate of the objective above; it is a different surrogate that agrees with it
in gradient direction at ``theta = theta_old``. Everything else in this module is exact;
this substitution is not, and it is the reason the module is named after GSPO rather than
after the objective in the first display.

Why verl's own ``gspo`` is not this
-----------------------------------
verl has GSPO already, and its "sequence" is **one row**::

    seq_lengths = torch.sum(response_mask, dim=-1)        # the whole row

Under ``no_concat`` a row is a turn, so it is accidentally right. Under ``concat`` a row
is an entire episode and under ``compact`` a whole conversation, so the geometric mean is
taken over every turn at once: one ratio, one clip decision, for the episode. Nothing
raises -- the numbers are plausible and the algorithm is simply not the one named.

Aggregation, and why it is three levels
---------------------------------------
``1 / L_t`` is not a normalisation choice here, it is the derivative of ``s_t``::

    grad s_t = s_t * (1 / L_t) * sum_j grad log pi(y_{t,j})

so the token-mean *inside* a turn is mandatory. Across turns the policy gradient is a
**sum** -- an episode with ten decisions contributes ten terms -- and only the outermost
level, across rows, is the batch's estimate of an expectation and therefore a mean::

    seq-mean ( turn-sum ( token-mean ) )

Two exact limits follow, and ``tests/test_turn_gspo.py`` pins both:

* one turn per row  ==> elementwise identical to verl's ``gspo``
* every token its own turn ==> ratios and clipping elementwise identical to vanilla PPO

They are the only way to falsify this file, so they are not optional.

★ What this does to ``entropy_coeff`` and ``kl_loss_coef``
----------------------------------------------------------
The turn-sum is correct about the objective and it has a price. verl aggregates the
entropy and KL terms *separately*, with ``config.loss_agg_mode`` -- ``token-mean`` by
default -- so those stay at ``O(1)`` while this loss returns something proportional to
the number of turns in a row. Measured, with an all-ones loss matrix::

    turns/row      turn_gspo pg     token-mean(entropy, kl)     ratio
            1             1.000                       1.000      1.0x
            2             2.000                       1.000      2.0x
            5             5.000                       1.000      5.0x
           10            10.000                       1.000     10.0x

So switching ``gspo -> turn_gspo`` divides the *effective* entropy and KL coefficients by
roughly the mean turn count, and that count moves as the policy learns to be more or less
verbose. **Multiply both coefficients by the expected turns per row when switching**, and
watch ``actor/turns_per_row``, which is reported for exactly this reason.

Not "fixed" by normalising here: dividing by a batch's mean turn count would make the
objective depend on a batch statistic, which is the same drifting-hyperparameter problem
wearing a different hat. ``tests/test_turn_gspo.py`` pins the relationship so it cannot
change silently.

★ The KL *penalty* needs no turn-level version
-----------------------------------------------
Only the importance ratio does. KL between two autoregressive distributions is additive
over the factorisation::

    KL(pi(a_t|s_t) || pi_ref(a_t|s_t)) = sum_j E[ KL(pi(.|s,y_<j) || pi_ref(.|s,y_<j)) ]

so the per-token KL already sums to the turn-level KL, and verl's existing per-token
``kl_penalty`` is the right quantity untouched. The ratio is different because ``clip``
is not linear: ``clip(prod_j r_j)`` cannot be recovered from the individual
``clip(r_j)``, which is the entire reason this file exists.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import register_policy_loss
from verl.workers.config import ActorConfig

#: Marks a position the model did not emit. ``turn_id`` uses it instead of 0 so that
#: observation tokens cannot be folded into turn 0 -- see ``_Packed.turn_ids``.
NO_TURN = -1


def turn_log_ratio(
    log_ratio: torch.Tensor,
    turn_id: torch.Tensor,
    response_mask: torch.Tensor,
    reduce: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce ``log_ratio`` within each turn and broadcast the result back to its tokens.

    ``reduce`` is the entire difference between the two turn-level losses in this module:

    * ``"sum"`` gives ``log R_t = sum_j log r_{t,j}``, i.e. the literal
      ``pi(a_t|s_t) / pi_old(a_t|s_t)`` -- what ``turn_ppo`` clips.
    * ``"mean"`` gives ``log s_t = log R_t / L_t``, the geometric mean -- what
      ``turn_gspo`` clips.

    Returns ``(per_token_turn_value, turn_lengths_per_token)``. Both are shaped like the
    input; the second is how many model-output tokens the position's turn holds, which
    the aggregation needs and which is cheaper to return than to recompute.

    Turns are identified by ``(row, turn_id)``. A row belongs to exactly one trajectory,
    so ids cannot collide inside one, and scattering per row keeps the whole thing a
    single pass with no python loop over turns.
    """
    if reduce not in ("mean", "sum"):
        raise ValueError(f"reduce must be 'mean' or 'sum', got {reduce!r}")
    mask = response_mask.to(log_ratio.dtype)
    # `turn_id` is -1 where the model was silent. Shift so every index is >= 0 for
    # scatter_add_, and rely on the mask -- not the index -- to exclude those positions.
    idx = (turn_id + 1).clamp(min=0).to(torch.long)
    width = int(idx.max().item()) + 1 if idx.numel() else 1

    sums = torch.zeros((log_ratio.shape[0], width), dtype=log_ratio.dtype, device=log_ratio.device)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(1, idx, log_ratio * mask)
    counts.scatter_add_(1, idx, mask)

    reduced = sums if reduce == "sum" else sums / counts.clamp(min=1.0)
    return torch.gather(reduced, 1, idx), torch.gather(counts, 1, idx)


def turn_geometric_mean_log_ratio(log_ratio, turn_id, response_mask):
    """GSPO's turn ratio: ``log s_t = log R_t / L_t``. See :func:`turn_log_ratio`."""
    return turn_log_ratio(log_ratio, turn_id, response_mask, reduce="mean")


def aggregate_seq_mean_turn_sum(
    loss_mat: torch.Tensor,
    response_mask: torch.Tensor,
    per_token_scale: torch.Tensor | float,
    global_batch_size: Optional[int] = None,
    dp_size: int = 1,
) -> torch.Tensor:
    """Scale each token, sum the row, average over rows.

    ``per_token_scale`` is the only thing that differs between the two turn-level losses,
    and it is not a normalisation choice in either -- it is the derivative of whatever
    ratio that loss clips:

    * ``turn_gspo`` clips ``s_t = R_t ** (1/L_t)``, whose gradient carries ``1 / L_t``,
      so the scale is ``1 / turn_lengths`` and a turn contributes its token-*mean*.
    * ``turn_ppo`` clips ``R_t`` itself, whose gradient carries no such factor, so the
      scale is ``1`` and a turn contributes its token-*sum*.

    Summing across turns within a row is the policy gradient over an episode's actions --
    ten decisions are ten terms -- and only the outermost level, across rows, estimates an
    expectation and is therefore a mean.

    ``global_batch_size`` is the number of rows across all data-parallel ranks, which the
    caller gets from verl's ``global_batch_info``. It has to be the global count, not the
    local one: with a local count each rank would divide by its own row count and the
    all-reduced gradient would be the mean of per-rank means, which differs from the mean
    over the batch whenever the ranks hold different numbers of rows.
    """
    mask = response_mask.to(loss_mat.dtype)
    per_row = torch.sum(loss_mat * mask * per_token_scale, dim=-1)

    row_alive = (torch.sum(mask, dim=-1) > 0).to(loss_mat.dtype)
    if global_batch_size is None:
        if dp_size > 1:
            raise ValueError(
                "a turn-level loss needs the global row count to average over rows; with "
                "dp_size > 1 a local count makes each rank divide by its own number of "
                "rows and the reduced gradient is then a mean of means."
            )
        global_batch_size = row_alive.sum()
    return torch.sum(per_row * row_alive) / global_batch_size * dp_size


def aggregate_seq_mean_turn_sum_token_mean(
    loss_mat, response_mask, turn_lengths, global_batch_size=None, dp_size=1
):
    """``turn_gspo``'s aggregation. See :func:`aggregate_seq_mean_turn_sum`."""
    return aggregate_seq_mean_turn_sum(
        loss_mat, response_mask, 1.0 / turn_lengths.clamp(min=1.0), global_batch_size, dp_size
    )


def _turn_level_policy_loss(
    *, old_log_prob, log_prob, advantages, response_mask, config, rollout_is_weights,
    turn_id, reduce: str, name: str,
):
    """Shared body of ``turn_gspo`` and ``turn_ppo``.

    They differ in exactly two coupled places, and the coupling is why this is one
    function rather than two: ``reduce`` picks which turn-level ratio is clipped, and the
    per-token scale in the aggregation must be that ratio's derivative factor. Written
    separately, the pair would drift -- and a mismatched pair still trains, just on a
    gradient that is not the gradient of anything.
    """
    assert config is not None and isinstance(config, ActorConfig)
    if turn_id is None:
        raise ValueError(
            f"{name} needs a `turn_id` column and the batch has none. It is published by "
            "the trajectory advantage estimators (default_gae, token_level_gae, "
            "turn_level_gae); with verl's own gae or grpo there is nothing "
            "that says "
            "where a turn starts. Without it this loss cannot tell a turn from a row, "
            "which is exactly the bug it exists to fix -- so it refuses rather than "
            "falling back to verl's `gspo`."
        )

    clip_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob
    turn_lr, turn_lengths = turn_log_ratio(
        negative_approx_kl, turn_id, response_mask, reduce=reduce
    )

    # GSPO-token: numerically the importance equals the turn's ratio, but the gradient is
    # sg[ratio] * grad log pi_j. Writing it this way is what lets the advantage vary
    # within a turn while the ratio and the clip stay turn-level -- which is what a
    # per-token advantage paired with a turn-level action needs, and what a plain
    # broadcast could not give.
    log_importance = log_prob - log_prob.detach() + turn_lr.detach()
    log_importance = torch.clamp(log_importance, max=10.0)  # as verl's gspo does
    importance = torch.exp(log_importance)

    losses_unclipped = -advantages * importance
    losses_clipped = -advantages * torch.clamp(importance, 1 - clip_low, 1 + clip_high)
    pg_losses = torch.maximum(losses_unclipped, losses_clipped)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # The derivative factor of the ratio being clipped -- see aggregate_seq_mean_turn_sum.
    scale = (1.0 / turn_lengths.clamp(min=1.0)) if reduce == "mean" else 1.0
    pg_loss = aggregate_seq_mean_turn_sum(
        pg_losses, response_mask, scale,
        global_batch_size=config.global_batch_info.get("global_batch_size"),
        dp_size=config.global_batch_info.get("dp_size", 1),
    )

    clipped = torch.gt(losses_clipped, losses_unclipped).float()
    turns_per_row = [
        int(torch.unique(row[m]).numel()) for row, m in zip(turn_id, response_mask.to(bool))
    ]
    # ★ The spread of the turn log-ratio against the clip range, which is the number that
    # decides whether this loss is doing anything. `turn_ppo` sums L_t token log-ratios,
    # so this grows like sqrt(L_t); once it passes log(1+eps) ~ 0.18 the clip stops
    # discriminating and nearly every sample lands on the clipped branch. `turn_gspo`
    # divides by L_t and this *shrinks* with length instead.
    lr_std = verl_F.masked_mean(
        (turn_lr - verl_F.masked_mean(turn_lr, response_mask)) ** 2, response_mask
    ).sqrt()
    metrics = {
        "actor/pg_clipfrac": verl_F.masked_mean(clipped, response_mask).detach().item(),
        "actor/ppo_kl": verl_F.masked_mean(-negative_approx_kl, response_mask).detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
        # ★ At one turn per row this loss and verl's `gspo` are identical, so a run
        # reporting 1.0 here is paying for machinery it is not using -- and a number far
        # from the expected turn count means the boundaries are being detected wrong.
        "actor/turns_per_row": (sum(turns_per_row) / len(turns_per_row)) if turns_per_row else 0.0,
        "actor/turn_len_mean": verl_F.masked_mean(turn_lengths, response_mask).detach().item(),
        "actor/turn_log_ratio_std": lr_std.detach().item(),
    }
    return pg_loss, metrics


@register_policy_loss("turn_gspo")
def compute_policy_loss_turn_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    turn_id: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """GSPO with the turn, rather than the row, as the sequence.

    Clips ``s_t = R_t ** (1 / L_t)``, the geometric mean of the turn's token ratios.

    ``loss_agg_mode`` is accepted because verl's signature has it and ignored because the
    aggregation is forced by the derivative of ``s_t`` -- see the module docstring.
    """
    return _turn_level_policy_loss(
        old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
        response_mask=response_mask, config=config, rollout_is_weights=rollout_is_weights,
        turn_id=turn_id, reduce="mean", name="turn_gspo",
    )


@register_policy_loss("turn_ppo")
def compute_policy_loss_turn_ppo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    turn_id: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """PPO with the turn as one action, on the *literal* ratio.

    Clips ``R_t = prod_j r_{t,j} = pi(a_t|s_t) / pi_old(a_t|s_t)``. This is the exact
    objective that ``turn_gspo`` approximates -- the one place in this module where no
    surrogate is involved -- and it is here so the cost of that approximation can be
    measured rather than assumed.

    ★ **Expect to widen the clip range.** ``log R_t`` is a sum of ``L_t`` token
    log-ratios, so its spread grows like ``sqrt(L_t)`` while ``log s_t``'s shrinks like
    ``1 / sqrt(L_t)``. At a per-token jitter of 0.02 the two differ by ``L_t`` -- 80x on a
    typical Sokoban turn. With ``clip_ratio`` left at 0.2, ``actor/pg_clipfrac`` will sit
    near 1 and almost every sample will contribute the clipped branch, which is the
    failure GSPO exists to avoid. ``actor/turn_log_ratio_std`` against ``log(1 + eps)``
    tells you immediately whether the range is usable.

    ★ **The reported ``actor/pg_loss`` is inflated by ``L_t``.** The per-token form has
    the correct *gradient* -- summing ``L_t`` copies of ``-A_t R_t`` reproduces
    ``grad(-A_t R_t)`` exactly, because ``grad R_t = R_t * sum_j grad log pi_j`` carries no
    ``1 / L_t``. Its *value* is ``L_t`` times the objective. Both cannot be right at once
    with this construction, and the gradient is the one that has to be. Divide the logged
    number by ``actor/turn_len_mean`` to compare it against ``turn_gspo``'s.
    """
    return _turn_level_policy_loss(
        old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
        response_mask=response_mask, config=config, rollout_is_weights=rollout_is_weights,
        turn_id=turn_id, reduce="sum", name="turn_ppo",
    )
