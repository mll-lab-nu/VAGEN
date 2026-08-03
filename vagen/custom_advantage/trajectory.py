"""Grouping rows into trajectories, shared by every multi-turn advantage estimator.

A trajectory is one episode. How it is laid out in the batch is a separate question
from which algorithm scores it:

* concat -- one row per trajectory, every turn's tokens in that row
* no-concat -- one row per turn, so a trajectory spans several rows

Token-level PPO, turn-level PPO and GRPO all need the same thing from that layout:
the rows of each trajectory, in turn order, and which token positions in them are model
output. This module supplies exactly that, so an estimator can be written once and run
under either layout -- a concat trajectory is simply one whose row list has length one.

Two details the layout forces and every estimator would otherwise repeat:

* Padding duplicates. ``pad_dataproto_to_divisor`` repeats real rows to reach a multiple
  of the DP world size, so a ``(group_idx, traj_idx, turn_idx)`` triple can appear more
  than once. Scoring a duplicate twice would double-count it in the backward recursion,
  so the view deduplicates and offers ``broadcast`` to expand results back.
* ``group_idx`` is a uuid string, so it has to be factorized before it can be sorted or
  compared numerically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def to_int64_codes(x, factorize_if_non_numeric: bool = False) -> np.ndarray:
    """Coerce an index column to int64, factorizing strings when asked.

    Only identity matters for grouping, so the particular codes are irrelevant as long
    as equal values map to equal codes.
    """
    x = np.asarray(x)
    if x.dtype.kind in "iu":
        return x.astype(np.int64, copy=False)
    if x.dtype.kind == "f":
        if not np.all(np.isfinite(x)):
            raise ValueError("index column contains non-finite values")
        return x.astype(np.int64)
    if factorize_if_non_numeric:
        _, inverse = np.unique(x.astype(str), return_inverse=True)
        return inverse.astype(np.int64, copy=False)
    return x.astype(np.int64, copy=False)


@dataclass
class TrajectoryView:
    """Deduplicated rows, grouped into trajectories and ordered by turn.

    Attributes:
        rows: ``(n_unique,)`` index of each distinct row in the original batch.
        inverse: ``(batch_size,)`` distinct row each original row corresponds to.
        trajectories: per trajectory, its distinct-row indices in turn order.
        mask: ``(n_unique, L)`` bool, True where the token is model output.
        last_pos: ``(n_unique,)`` last True position per row, ``-1`` if the row is empty.
    """

    rows: torch.Tensor
    inverse: torch.Tensor
    trajectories: list[list[int]]
    mask: torch.Tensor
    last_pos: torch.Tensor

    @classmethod
    def build(cls, response_mask: torch.Tensor, non_tensor_batch) -> TrajectoryView:
        device = response_mask.device
        group = to_int64_codes(non_tensor_batch["group_idx"], factorize_if_non_numeric=True)
        traj = to_int64_codes(non_tensor_batch["traj_idx"])
        turn = to_int64_codes(non_tensor_batch["turn_idx"])

        key = np.stack([group, traj, turn], axis=1)
        uniq_key, first_idx, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)

        rows = torch.as_tensor(first_idx.astype(np.int64), dtype=torch.long, device=device)
        mask = response_mask.to(torch.bool).index_select(0, rows)

        # np.unique sorts lexicographically, so rows of one trajectory are already
        # contiguous and in ascending turn order; group them by (group, traj) directly.
        trajectories: list[list[int]] = []
        for i, (g, t, _turn) in enumerate(uniq_key.tolist()):
            if trajectories and (g, t) == cls._key_of(uniq_key, trajectories[-1][-1]):
                trajectories[-1].append(i)
            else:
                trajectories.append([i])

        return cls(
            rows=rows,
            inverse=torch.as_tensor(inverse.astype(np.int64), dtype=torch.long, device=device),
            trajectories=trajectories,
            mask=mask,
            last_pos=cls._last_true(mask),
        )

    @staticmethod
    def _key_of(uniq_key: np.ndarray, i: int) -> tuple[int, int]:
        g, t, _ = uniq_key[i].tolist()
        return g, t

    @staticmethod
    def _last_true(mask: torch.Tensor) -> torch.Tensor:
        """Index of the last True per row, ``-1`` for an all-False row."""
        n, length = mask.shape
        positions = torch.arange(length, device=mask.device).view(1, length).expand(n, length)
        last = (mask.long() * positions).max(dim=1).values
        return torch.where(mask.any(dim=1), last, torch.full_like(last, -1))

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Take the distinct rows out of a full-batch tensor."""
        return tensor.index_select(0, self.rows)

    def broadcast(self, tensor: torch.Tensor) -> torch.Tensor:
        """Expand a per-distinct-row result back over the padded batch."""
        return tensor.index_select(0, self.inverse)
