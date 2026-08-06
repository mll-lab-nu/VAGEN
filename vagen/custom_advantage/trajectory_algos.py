"""Advantage estimators that score a trajectory however its rows are laid out.

Both work under concat and no-concat: :class:`TrajectoryView` presents a trajectory as
an ordered list of rows, and a concat trajectory is one whose list has length one.

* ``traj_token_gae`` -- token-level PPO. The critic values every model-output token and
  GAE runs backward over the trajectory's tokens, skipping everything that is not model
  output and carrying the recursion across row boundaries.
* ``traj_grpo`` -- one advantage for the whole trajectory, normalised against the other
  trajectories of its prompt group and broadcast to all of its tokens.

Neither writes sentinel returns: token-level supervises every model-output token, and
GRPO needs no critic. Only turn-level GAE leaves positions unsupervised, which is what
``value_mask`` exists for -- see ``registry.py``.
"""

from __future__ import annotations

def rewards_for_advantage(batch) -> "torch.Tensor":
    """The per-token reward the advantage should be built from.

    ``token_level_rewards`` when it exists, ``token_level_scores`` otherwise. verl writes
    the KL-penalised reward into the former and leaves the latter untouched, so reading
    only the scores makes ``algorithm.use_kl_in_reward=True`` a silent no-op -- the
    penalty is computed, stored, and never read.
    """
    rewards = batch.get("token_level_rewards")
    return batch["token_level_scores"] if rewards is None else rewards


import numpy as np
import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import register_adv_est

from vagen.custom_advantage.registry import register_sentinel_adv_est
from vagen.custom_advantage.trajectory import TrajectoryView, to_int64_codes
from vagen.trainer.logic import IGNORE_RETURN


def _sequence_index(view: TrajectoryView, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten each trajectory's model-output tokens into one padded sequence.

    Returns ``(index, valid)`` of shape ``(n_traj, max_len)``. ``index`` addresses a
    flattened ``(n_rows, width)`` tensor, so a single gather collects a trajectory's
    tokens across all of its rows in turn order. Padding sits on the right and is
    excluded by ``valid``.
    """
    device = view.mask.device
    sequences = []
    for rows in view.trajectories:
        # Rows already arrive in turn order, so concatenating their token positions
        # yields the trajectory's tokens in the order the model produced them.
        positions = [r * width + view.mask[r].nonzero(as_tuple=True)[0] for r in rows]
        sequences.append(torch.cat(positions) if positions else torch.empty(0, dtype=torch.long, device=device))

    max_len = max((len(s) for s in sequences), default=0)
    index = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    valid = torch.zeros((len(sequences), max_len), dtype=torch.bool, device=device)
    for i, seq in enumerate(sequences):
        index[i, : len(seq)] = seq
        valid[i, : len(seq)] = True
    return index, valid


@register_adv_est("traj_token_gae")
def compute_traj_token_gae(*, batch, non_tensor_batch, config=None, **kwargs):
    """Token-level GAE over a trajectory, across whatever rows it occupies."""
    gamma, lam = float(config.gamma), float(config.lam)
    scores = rewards_for_advantage(batch)
    values = batch.get("values", torch.zeros_like(scores))
    response_mask = batch["response_mask"]

    with torch.no_grad():
        view = TrajectoryView.build(response_mask, non_tensor_batch)
        width = scores.shape[1]
        rows_scores = view.gather(scores)
        rows_values = view.gather(values)
        mask_f = view.mask.to(rows_scores.dtype)

        index, valid = _sequence_index(view, width)
        flat_scores, flat_values = rows_scores.reshape(-1), rows_values.reshape(-1)
        seq_r = torch.where(valid, flat_scores[index], torch.zeros_like(flat_scores[index]))
        seq_v = torch.where(valid, flat_values[index], torch.zeros_like(flat_values[index]))

        n_traj, max_len = index.shape
        seq_adv = torch.zeros_like(seq_v)
        nextvalues = torch.zeros(n_traj, dtype=seq_v.dtype, device=seq_v.device)
        lastgaelam = torch.zeros_like(nextvalues)

        # Backward over token position, vectorised across trajectories. Padding is on
        # the right, so the first steps of the loop must leave the recursion untouched
        # rather than folding zeros into it.
        for t in reversed(range(max_len)):
            live = valid[:, t]
            delta = seq_r[:, t] + gamma * nextvalues - seq_v[:, t]
            lastgaelam = torch.where(live, delta + gamma * lam * lastgaelam, lastgaelam)
            seq_adv[:, t] = torch.where(live, lastgaelam, torch.zeros_like(lastgaelam))
            nextvalues = torch.where(live, seq_v[:, t], nextvalues)

        advantages = torch.zeros_like(rows_values).reshape(-1)
        returns = torch.zeros_like(rows_values).reshape(-1)
        advantages[index[valid]] = seq_adv[valid]
        returns[index[valid]] = (seq_adv + seq_v)[valid]
        advantages = advantages.view_as(rows_values)
        returns = returns.view_as(rows_values)

        advantages = verl_F.masked_whiten(advantages, mask_f) * mask_f
        return view.broadcast(advantages), view.broadcast(returns)


@register_adv_est("traj_grpo")
def compute_traj_grpo(*, batch, non_tensor_batch, config=None, **kwargs):
    """One advantage per trajectory, normalised within its prompt group.

    Needs no critic, so ``returns`` mirrors ``advantages`` -- verl's own GRPO does the
    same, and nothing reads ``returns`` when the critic is disabled.
    """
    scores = rewards_for_advantage(batch)
    response_mask = batch["response_mask"]
    norm_by_std = True if config is None else config.get("norm_adv_by_std_in_grpo", True)

    with torch.no_grad():
        view = TrajectoryView.build(response_mask, non_tensor_batch)
        rows_scores = view.gather(scores)
        mask_f = view.mask.to(rows_scores.dtype)

        # A trajectory's return is every reward it collected, wherever those rows sit.
        row_totals = (rows_scores * mask_f).sum(dim=1)
        traj_return = torch.stack([row_totals[rows].sum() for rows in view.trajectories])

        group_codes = to_int64_codes(non_tensor_batch["group_idx"], factorize_if_non_numeric=True)
        # Every row of a trajectory shares its group, so the first row identifies it.
        traj_group = torch.as_tensor(
            np.asarray([group_codes[view.rows[rows[0]].item()] for rows in view.trajectories]),
            device=traj_return.device,
        )

        traj_adv = torch.zeros_like(traj_return)
        for g in torch.unique(traj_group).tolist():
            sel = traj_group == g
            rewards = traj_return[sel]
            centred = rewards - rewards.mean()
            if norm_by_std:
                # A group whose trajectories all scored the same carries no signal;
                # dividing by its zero std would produce NaNs rather than zeros.
                std = rewards.std(unbiased=False)
                centred = centred / (std + 1e-6) if std > 0 else torch.zeros_like(centred)
            traj_adv[sel] = centred

        advantages = torch.zeros_like(rows_scores)
        for j, rows in enumerate(view.trajectories):
            for r in rows:
                advantages[r] = traj_adv[j]
        advantages = advantages * mask_f

        return view.broadcast(advantages), view.broadcast(advantages.clone())


def _is_turn_boundary(index: torch.Tensor, valid: torch.Tensor, width: int) -> torch.Tensor:
    """True at the last model-output token of each turn.

    A turn ends either because the environment interrupts -- leaving a gap in the
    positions within a row -- or because the trajectory continues in the next row. The
    row change has to be tested explicitly: ``row * width + position`` runs on unbroken
    across a row boundary whenever a row's model output reaches its end, so a gap test
    alone silently merges two turns into one.
    """
    boundary = torch.zeros_like(valid)
    gap = index[:, 1:] != index[:, :-1] + 1
    new_row = index[:, 1:] // width != index[:, :-1] // width
    boundary[:, :-1] = valid[:, 1:] & (gap | new_row)
    # The last valid token of a trajectory ends the last turn.
    last = valid & ~torch.roll(valid, shifts=-1, dims=1)
    last[:, -1] = valid[:, -1]
    return boundary | last


@register_adv_est("traj_bilevel_gae")
def compute_traj_bilevel_gae(*, batch, non_tensor_batch, config=None, **kwargs):
    """GAE with one lambda inside a turn and another across turns.

    The two views of a multi-turn episode -- one step per token, one step per turn --
    optimise the same objective when gamma is 1, so neither is more correct; they differ
    in the bias/variance their lambda buys. Token-level GAE weights a turn by
    ``lam ** turn_length``, which for variable-length turns cannot equal a fixed
    per-turn weight, so the two are genuinely different estimators and there is room
    between them.

    This is a lambda-return with a position-dependent lambda (Sutton & Barto's variable
    lambda), which is a legitimate lambda-return rather than two estimators added
    together::

        A_j = delta_j + gamma * lam_j * A_{j+1}
        lam_j = lam_low   inside a turn
                lam       at the token that ends one

    Its limits are exact, which is what makes it testable rather than merely plausible:

    * ``lam_low == lam`` reproduces token-level GAE token for token.
    * ``lam_low == 1`` reproduces turn-level GAE at the first token of every turn. With
      no intra-turn reward the deltas telescope:
      ``sum_j delta_j = r_t + V(s_{t+1}) - V(s_t)``, which is the turn-level delta, so
      the recursion becomes the turn-level one. Tokens after the first then carry the
      same credit minus the value drift accumulated within the turn -- the refinement
      the token level is there to add.

    Unlike turn-level GAE this supervises every model-output token, so it emits no
    sentinel returns and needs no ``value_mask``.
    """
    gamma = float(config.gamma)
    lam_high = float(config.lam)
    lam_low = float(config.get("lam_low", 1.0)) if hasattr(config, "get") else 1.0

    scores = rewards_for_advantage(batch)
    values = batch.get("values", torch.zeros_like(scores))
    response_mask = batch["response_mask"]

    with torch.no_grad():
        view = TrajectoryView.build(response_mask, non_tensor_batch)
        width = scores.shape[1]
        rows_scores, rows_values = view.gather(scores), view.gather(values)
        mask_f = view.mask.to(rows_scores.dtype)

        index, valid = _sequence_index(view, width)
        flat_scores, flat_values = rows_scores.reshape(-1), rows_values.reshape(-1)
        zeros = torch.zeros_like(flat_values[index])
        seq_r = torch.where(valid, flat_scores[index], zeros)
        seq_v = torch.where(valid, flat_values[index], zeros)

        seq_lam = torch.where(_is_turn_boundary(index, valid, width), lam_high, lam_low).to(seq_v.dtype)

        n_traj, max_len = index.shape
        seq_adv = torch.zeros_like(seq_v)
        nextvalues = torch.zeros(n_traj, dtype=seq_v.dtype, device=seq_v.device)
        lastgaelam = torch.zeros_like(nextvalues)

        for t in reversed(range(max_len)):
            live = valid[:, t]
            delta = seq_r[:, t] + gamma * nextvalues - seq_v[:, t]
            lastgaelam = torch.where(live, delta + gamma * seq_lam[:, t] * lastgaelam, lastgaelam)
            seq_adv[:, t] = torch.where(live, lastgaelam, torch.zeros_like(lastgaelam))
            nextvalues = torch.where(live, seq_v[:, t], nextvalues)

        advantages = torch.zeros_like(rows_values).reshape(-1)
        returns = torch.zeros_like(rows_values).reshape(-1)
        advantages[index[valid]] = seq_adv[valid]
        returns[index[valid]] = (seq_adv + seq_v)[valid]
        advantages = advantages.view_as(rows_values)
        returns = returns.view_as(rows_values)

        advantages = verl_F.masked_whiten(advantages, mask_f) * mask_f
        return view.broadcast(advantages), view.broadcast(returns)


@register_sentinel_adv_est("traj_turn_gae")
def compute_traj_turn_gae(*, batch, non_tensor_batch, config=None, ignore_value: float = IGNORE_RETURN, **kwargs):
    """Turn-level GAE: one decision per turn, whatever rows the turn occupies.

    The recursion runs over turns rather than tokens -- a turn's reward is everything it
    collected, its value is the critic at its *first* model-output token -- and the
    resulting advantage is broadcast to every token of that turn.

    ★ First, not last. ``V(s_t)`` is the value of the state the turn acts from, which is
    the position before any of its tokens have been emitted. The critic at the turn's
    last token has already seen nearly the whole turn and is answering a different
    question. That broadcast is not
    an approximation: an autoregressive policy factorises as
    ``log pi(turn) = sum_j log pi(token_j)``, so the per-token coefficient in the
    turn-level policy gradient is exactly the turn's advantage.

    Returns are written only at each turn's first token, the rest left at the sentinel,
    because the critic is being asked for a turn-level value and only that position
    carries one. ``value_mask`` is what stops it training on the rest.

    Unlike ``no_concat_gae``, which this replaces, turns are found from the token stream
    rather than assumed to be rows, so it works under either layout.
    """
    gamma, lam = float(config.gamma), float(config.lam)
    scores = rewards_for_advantage(batch)
    values = batch.get("values", torch.zeros_like(scores))
    response_mask = batch["response_mask"]

    with torch.no_grad():
        view = TrajectoryView.build(response_mask, non_tensor_batch)
        width = scores.shape[1]
        rows_scores, rows_values = view.gather(scores), view.gather(values)
        mask_f = view.mask.to(rows_scores.dtype)

        index, valid = _sequence_index(view, width)
        flat_scores, flat_values = rows_scores.reshape(-1), rows_values.reshape(-1)
        zeros = torch.zeros_like(flat_values[index])
        seq_r = torch.where(valid, flat_scores[index], zeros)
        seq_v = torch.where(valid, flat_values[index], zeros)
        boundary = _is_turn_boundary(index, valid, width) & valid

        # Turn index of every token: how many turns ended strictly before it.
        turn_of = (boundary.cumsum(dim=1) - boundary.long()).clamp(min=0)
        # A turn's first token: the one after a boundary, plus the sequence's own start.
        start = torch.zeros_like(boundary)
        start[:, 1:] = boundary[:, :-1]
        start[:, 0] = valid[:, 0]
        start = start & valid
        n_turns = int(boundary.sum(dim=1).max().item()) or 1
        n_traj = index.shape[0]

        turn_r = torch.zeros((n_traj, n_turns), dtype=seq_v.dtype, device=seq_v.device)
        turn_r.scatter_add_(1, turn_of.clamp(max=n_turns - 1), seq_r)
        # scatter_add_, not scatter_: with one index per token, several tokens land in
        # the same turn slot and a plain scatter keeps whichever is written last -- a
        # zero from a non-start token, overwriting the value that was wanted.
        turn_v = torch.zeros_like(turn_r)
        turn_v.scatter_add_(1, turn_of.clamp(max=n_turns - 1), torch.where(start, seq_v, torch.zeros_like(seq_v)))
        turn_count = torch.zeros_like(turn_r)
        turn_count.scatter_add_(1, turn_of.clamp(max=n_turns - 1), boundary.to(turn_r.dtype))
        turn_alive = turn_count > 0

        turn_adv = torch.zeros_like(turn_r)
        nextvalue = torch.zeros(n_traj, dtype=seq_v.dtype, device=seq_v.device)
        lastgaelam = torch.zeros_like(nextvalue)
        for t in reversed(range(n_turns)):
            live = turn_alive[:, t]
            delta = turn_r[:, t] + gamma * nextvalue - turn_v[:, t]
            lastgaelam = torch.where(live, delta + gamma * lam * lastgaelam, lastgaelam)
            turn_adv[:, t] = torch.where(live, lastgaelam, torch.zeros_like(lastgaelam))
            nextvalue = torch.where(live, turn_v[:, t], nextvalue)

        seq_adv = torch.gather(turn_adv, 1, turn_of.clamp(max=n_turns - 1)) * valid
        seq_ret = torch.gather(turn_adv + turn_v, 1, turn_of.clamp(max=n_turns - 1))

        advantages = torch.zeros_like(rows_values).reshape(-1)
        returns = torch.full_like(rows_values, float(ignore_value)).reshape(-1)
        advantages[index[valid]] = seq_adv[valid]
        # Only the turn's first token carries a turn-level return -- that is the state
        # the value was asked about. The rest stay at the sentinel and are excluded from
        # the critic loss by value_mask.
        returns[index[start]] = seq_ret[start]
        advantages = advantages.view_as(rows_values)
        returns = returns.view_as(rows_values)

        advantages = verl_F.masked_whiten(advantages, mask_f) * mask_f
        return view.broadcast(advantages), view.broadcast(returns)
