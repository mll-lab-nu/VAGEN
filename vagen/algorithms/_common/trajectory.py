"""Grouping rows into trajectories, shared by every multi-turn algorithm.

A trajectory is one episode. How it is laid out in the batch is a separate question
from which algorithm scores it:

* concat -- one row per trajectory, every turn's tokens in that row
* no-concat -- one row per turn, so a trajectory spans several rows

Token-level PPO, turn-level PPO and GRPO all need the same thing from that layout:
the rows of each trajectory, in turn order, and which token positions in them are model
output. This module supplies exactly that, so an estimator can be written once and run
under either layout -- a concat trajectory is simply one whose row list has length one.

* compact -- one row per conversation, a new one at every compaction

★ **What identifies a trajectory.** ``episode_id``, when the agent loop emits it: it is
minted once per ``run_episode`` and is the only column that *means* "one episode".
``(group_idx, traj_idx)`` is the fallback, and it is a weaker thing -- that pair
identifies a *rollout*, and a rollout is one episode only because the loop happens to
make it so. The two agree today only because verl's ``repeat_interleave`` and VAGEN's
positional ``traj_idx`` tile line up, which nothing asserts; and on the validation path
they already disagree, because padding a batch to a multiple of the worker count
duplicates prompts that then run as genuinely separate episodes under the same pair.
Merging two episodes into one trajectory is silent: the backward recursion simply runs
through both, and the earlier one is credited with the later one's rewards.

Two details the layout forces and every estimator would otherwise repeat:

* Padding duplicates. ``pad_dataproto_to_divisor`` repeats real rows to reach a multiple
  of the DP world size, so an identity can appear more than once. Scoring a duplicate
  twice would double-count it in the backward recursion, so the view deduplicates and
  offers ``broadcast`` to expand results back.
* ``group_idx`` and ``episode_id`` are uuid strings, so they have to be factorized before
  they can be sorted or compared numerically.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


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
        # Identity first: `episode_id` when the loop published one, because that column
        # is minted per episode and nothing else here is. The pair below identifies a
        # rollout, which coincides with an episode only by construction of the loop --
        # see this module's docstring. Two int64 columns either way, so the rest of the
        # method does not care which it got.
        if "episode_id" in non_tensor_batch:
            episode = to_int64_codes(non_tensor_batch["episode_id"], factorize_if_non_numeric=True)
            group = np.zeros(len(episode), dtype=np.int64)
        else:
            group = to_int64_codes(non_tensor_batch["group_idx"], factorize_if_non_numeric=True)
            episode = to_int64_codes(non_tensor_batch["traj_idx"])
            # ★ The fallback merges rather than fails. `(group_idx, traj_idx)` identifies a
            # *rollout*, and a rollout is one episode only because the training loop makes
            # it so -- validation runs several episodes per rollout (n_envs 30 and 60 on
            # val_navigation, 50 on val_spatial_gym), so every one of them collapses into
            # a single "trajectory" and the recursion carries credit from one episode into
            # a completely unrelated one. Nothing raises: the shapes are right and the
            # numbers are finite. Warned rather than raised because the training path
            # always publishes `episode_id` and only `concat_val_multi_turn` does not.
            logger.warning(
                "no `episode_id` column; grouping trajectories by (group_idx, traj_idx) "
                "instead. That is one rollout, not one episode -- if a rollout here holds "
                "more than one episode they are being scored as a single trajectory, with "
                "credit flowing between them."
            )
        # Concat keeps a whole trajectory in one row, so there is no turn axis and the
        # agent loop emits none. Defaulting to zero is what makes an estimator run
        # unchanged under either layout instead of needing a concat-specific branch.
        if "turn_idx" in non_tensor_batch:
            turn = to_int64_codes(non_tensor_batch["turn_idx"])
        else:
            turn = np.zeros(len(episode), dtype=np.int64)

        key = np.stack([group, episode, turn], axis=1)
        uniq_key, first_idx, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)

        # Rows sharing (episode, turn) are deduplicated: np.unique keeps the first
        # and the rest inherit its advantages and returns. That is right for the copies
        # `pad_dataproto_to_divisor` appends, which are identical to what they duplicate.
        # It is wrong for rows that merely collide -- a loop that stopped emitting
        # turn_idx collapses every row onto turn 0 -- because then a row is handed an
        # advantage computed for a different response, on a mask that marks different
        # positions. The two are distinguishable: real copies agree on the mask.
        if len(uniq_key) != len(key):
            m = response_mask.to(torch.bool)
            for i, first in enumerate(first_idx):
                same = np.flatnonzero(inverse == i)
                if len(same) > 1 and not bool(
                    m.index_select(0, torch.as_tensor(same.astype(np.int64),
                                                      dtype=torch.long, device=device))
                    .eq(m[int(first)]).all()
                ):
                    raise ValueError(
                        f"{len(same)} rows share the key {tuple(uniq_key[i].tolist())} "
                        f"(episode, turn) but have different response "
                        f"masks, so they are distinct turns rather than padding copies. "
                        f"Deduplicating would give them one row's advantages. Check that "
                        f"the agent loop emits a distinct turn_idx per row."
                    )

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
