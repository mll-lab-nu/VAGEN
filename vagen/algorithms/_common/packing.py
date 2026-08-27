"""Shared trajectory packing, boundary detection, and GAE recursion primitives.

Concrete estimators own their reward placement and recursion policy. This module only
contains mechanics reused by multiple estimators: gathering an episode across batch
rows, scattering outputs back, finding turn boundaries, and running a generic backward
GAE recurrence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import verl.utils.torch_functional as verl_F

from vagen.algorithms._common.inputs import AdvantageInputs, AdvantageOutputs
from vagen.algorithms._common.trajectory import TrajectoryView


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


def _last_valid(valid: torch.Tensor) -> torch.Tensor:
    """True at each sequence's final valid position; all False for a sequence with none.

    Padding is contiguous and on the right, so the last valid position is the one whose
    successor is padding, plus the final column. Three callers want this and each used to
    spell it out; the version they spelled used ``torch.roll``, which wraps the first
    column onto the last and then had to undo it -- correct, but only because of a repair
    line that is easy to drop when the expression is copied.
    """
    last = torch.zeros_like(valid)
    if valid.shape[1] == 0:
        return last
    last[:, :-1] = valid[:, :-1] & ~valid[:, 1:]
    last[:, -1] = valid[:, -1]
    return last


@dataclass
class _Packed:
    """A batch's trajectories, each flattened to one padded sequence of its own tokens.

    Several estimators need the same three mechanics: gather a trajectory's rows into a
    single ordered sequence, operate on it, then scatter the result back to its original
    rows. Algorithm-specific reward placement and recursions stay in the concrete
    implementation modules.
    """

    view: TrajectoryView
    index: torch.Tensor  # (n_traj, max_len), addressing a flattened (n_rows, width)
    valid: torch.Tensor  # (n_traj, max_len) bool -- False on the right-hand padding
    seq_r: torch.Tensor
    seq_v: torch.Tensor
    mask_f: torch.Tensor  # (n_rows, width), the model-output mask as a float
    rows_values: torch.Tensor
    width: int

    def boundary(self) -> torch.Tensor:
        """True at the last model-output token of each turn."""
        return _is_turn_boundary(self.index, self.valid, self.width)

    def seam(self, ends_with_summary) -> torch.Tensor:
        """True at the last model-output token of a compaction summary.

        Those tokens are turn boundaries by :meth:`boundary` -- a summary is a model
        emission and an action, so it ends a turn in the token stream -- but no
        environment step follows one. Marking them lets an estimator charge the seam
        differently from a real transition; see the module docstring of
        ``vagen/harness/_common/base.py`` for why the difference matters.
        """
        seam = torch.zeros_like(self.valid)
        if ends_with_summary is None or self.valid.shape[1] == 0:
            return seam
        flags = torch.as_tensor(
            np.asarray([bool(x) for x in ends_with_summary]), device=self.valid.device
        )
        on_flagged_row = flags[self.view.rows[self.index // self.width]] & self.valid

        # The summary is the row's *last* emission, so only that one token is the seam.
        # A conversation holds several turns, so `boundary()` is true several times on a
        # flagged row and intersecting with it would mark every turn in the conversation
        # as a seam -- turning the whole row's inter-turn discounting off.
        row_of = self.index // self.width
        last_on_row = torch.zeros_like(self.valid)
        last_on_row[:, :-1] = self.valid[:, 1:] & (row_of[:, 1:] != row_of[:, :-1])
        return on_flagged_row & (last_on_row | _last_valid(self.valid))

    def scatter(self, seq: torch.Tensor, where: torch.Tensor | None = None, fill: float = 0.0):
        """Sequence-shaped -> row-shaped, writing only at ``where`` (default: everywhere)."""
        where = self.valid if where is None else where
        out = torch.full_like(self.rows_values, float(fill)).reshape(-1)
        out[self.index[where]] = seq[where]
        return out.view_as(self.rows_values)

    def scatter_flag(self, where: torch.Tensor) -> torch.Tensor:
        """A 0/1 long tensor marking the row positions named by ``where``."""
        return self.scatter_int(torch.ones_like(self.index), where=where)

    def scatter_int(self, seq, where: torch.Tensor | None = None, fill: int = 0):
        """Sequence-shaped -> row-shaped, as integers. ``scatter`` but not float."""
        where = self.valid if where is None else where
        out = torch.full_like(self.rows_values, float(fill), dtype=torch.long).reshape(-1)
        out[self.index[where]] = seq[where].to(torch.long)
        return out.view_as(self.rows_values)

    def turn_ids(self) -> torch.Tensor:
        """Which turn each model-output token belongs to; ``-1`` where the model was silent.

        Numbered per trajectory rather than per row, which costs nothing -- a row belongs
        to exactly one trajectory, so ids never collide within a row -- and makes the
        column readable on its own.

        ★ This is the only channel a turn-level *loss* has. ``PolicyLossFn``'s signature
        is fixed and carries no notion of a turn, and the loss sees padded tensors rather
        than the ``response_spans`` the loop published. Deriving turn boundaries a second
        time inside the loss would mean two implementations that can disagree while both
        look right, so the estimator that already computed them publishes them.

        ``-1`` at the silent positions rather than ``0``, which is a real turn. This is
        defensive, not load-bearing: ``turn_gspo`` weights every sum by ``response_mask``,
        so folding the observation tokens into turn 0 would not change its answer -- the
        mutation is a genuine no-op there, and no test catches it because there is nothing
        to catch. The convention is for any *other* consumer, which would otherwise take
        ``0`` at face value and inflate turn 0's length -- hence shrink its ``1 / L_t`` --
        by however verbose the environment happened to be.
        """
        boundary = self.boundary() & self.valid
        turn_of = (boundary.cumsum(dim=1) - boundary.long()).clamp(min=0)
        return self.scatter_int(turn_of, fill=-1)

    def emit(self, advantages, returns, value_mask=None) -> AdvantageOutputs:
        """Whiten the advantages over the model-output mask and broadcast back to rows.

        Every estimator that packs also publishes ``turn_id``, whether or not its own
        recursion needed it: it is the actor's only way to know where a turn starts, and
        which advantage estimator is in use should not decide whether a turn-level loss
        can run. ``turn_gspo`` refuses to start without it rather than inventing one.
        """
        advantages = verl_F.masked_whiten(advantages, self.mask_f) * self.mask_f
        return AdvantageOutputs(
            advantages=self.view.broadcast(advantages),
            returns=self.view.broadcast(returns),
            value_mask=None if value_mask is None else self.view.broadcast(value_mask),
            extra={"turn_id": self.view.broadcast(self.turn_ids())},
        )


def _pack(inputs: AdvantageInputs) -> _Packed:
    view = inputs.view
    scores, values = inputs.rewards, inputs.values
    width = scores.shape[1]
    rows_scores, rows_values = view.gather(scores), view.gather(values)
    index, valid = _sequence_index(view, width)
    flat_scores, flat_values = rows_scores.reshape(-1), rows_values.reshape(-1)
    zeros = torch.zeros_like(flat_values[index])
    return _Packed(
        view=view,
        index=index,
        valid=valid,
        seq_r=torch.where(valid, flat_scores[index], zeros),
        seq_v=torch.where(valid, flat_values[index], zeros),
        mask_f=view.mask.to(rows_scores.dtype),
        rows_values=rows_values,
        width=width,
    )


def _backward_gae(seq_r, seq_v, valid, gamma: float, lam) -> torch.Tensor:
    """GAE run backward over the packed sequences, vectorised across trajectories.

    ``lam`` is either a scalar -- plain token-level GAE -- or a tensor the shape of the
    sequences, which is Sutton & Barto's variable-lambda return. Nothing in the tree passes
    a tensor today; it is what a per-turn lambda would use.

    Padding sits on the right, so the first steps of the loop must leave the recursion
    untouched rather than folding zeros into it.
    """
    n_traj, max_len = valid.shape
    seq_adv = torch.zeros_like(seq_v)
    nextvalues = torch.zeros(n_traj, dtype=seq_v.dtype, device=seq_v.device)
    lastgaelam = torch.zeros_like(nextvalues)
    lam_t = lam if torch.is_tensor(lam) else None

    for t in reversed(range(max_len)):
        live = valid[:, t]
        lam_t_now = lam_t[:, t] if lam_t is not None else lam
        delta = seq_r[:, t] + gamma * nextvalues - seq_v[:, t]
        lastgaelam = torch.where(live, delta + gamma * lam_t_now * lastgaelam, lastgaelam)
        seq_adv[:, t] = torch.where(live, lastgaelam, torch.zeros_like(lastgaelam))
        nextvalues = torch.where(live, seq_v[:, t], nextvalues)
    return seq_adv


def _is_turn_boundary(index: torch.Tensor, valid: torch.Tensor, width: int) -> torch.Tensor:
    """True at the last model-output token of each turn.

    A turn ends either because the environment interrupts -- leaving a gap in the
    positions within a row -- or because the trajectory continues in the next row. The
    row change has to be tested explicitly: ``row * width + position`` runs on unbroken
    across a row boundary whenever a row's model output reaches its end, so a gap test
    alone silently merges two turns into one.
    """
    boundary = torch.zeros_like(valid)
    if valid.shape[1] == 0:
        # No row in the batch has a single model-output token, so there are no turns to
        # bracket. Without this the `valid[:, -1]` below indexes a zero-width dimension
        # and raises IndexError instead of returning "no boundaries".
        return boundary
    gap = index[:, 1:] != index[:, :-1] + 1
    new_row = index[:, 1:] // width != index[:, :-1] // width
    boundary[:, :-1] = valid[:, 1:] & (gap | new_row)
    # The last valid token of a trajectory ends the last turn.
    return boundary | _last_valid(valid)
