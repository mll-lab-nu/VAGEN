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

import numpy as np
import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import register_adv_est

from vagen.custom_advantage.trajectory import TrajectoryView, to_int64_codes


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
    scores = batch["token_level_scores"]
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
    scores = batch["token_level_scores"]
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
