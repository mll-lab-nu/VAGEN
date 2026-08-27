"""Trajectory-level GRPO implementation."""

import numpy as np
import torch

from vagen.algorithms._common import (
    AdvantageInputs,
    AdvantageOutputs,
    advantage_estimator,
    register_algorithm,
)
from vagen.algorithms._common.trajectory import to_int64_codes


# This estimator returns bare AdvantageOutputs rather than using the shared packed
# emitter, so it intentionally publishes no turn_id column for a turn-level loss.
@advantage_estimator("trajectory_grpo", publishes_turn_id=False)
def compute_trajectory_grpo(inputs: AdvantageInputs):
    """One advantage per trajectory, normalised within its prompt group.

    Needs no critic, so ``returns`` mirrors ``advantages`` -- verl's own GRPO does the
    same, and nothing reads ``returns`` when the critic is disabled.
    """
    scores = inputs.rewards
    norm_by_std = inputs.param("norm_adv_by_std_in_grpo", True)

    with torch.no_grad():
        view = inputs.view
        rows_scores = view.gather(scores)
        mask_f = view.mask.to(rows_scores.dtype)

        # A trajectory's return is every reward it collected, wherever those rows sit.
        row_totals = (rows_scores * mask_f).sum(dim=1)
        traj_return = torch.stack([row_totals[rows].sum() for rows in view.trajectories])

        group_codes = to_int64_codes(inputs.group_idx, factorize_if_non_numeric=True)
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

        return AdvantageOutputs(
            advantages=view.broadcast(advantages),
            returns=view.broadcast(advantages.clone()),
        )


SPEC = register_algorithm("trajectory_grpo", compute_trajectory_grpo)

__all__ = ["SPEC", "compute_trajectory_grpo"]
