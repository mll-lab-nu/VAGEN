from verl.trainer.ppo.core_algos import register_adv_est,compute_grpo_outcome_advantage
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig
from verl.trainer.ppo import core_algos



@register_adv_est("no_concat_ppo")
def compute_gae_no_concat_advantage_return(
    token_level_scores: torch.Tensor,    # (bs, L)
    token_level_rewards: torch.Tensor,   # (bs, L)  # not used here, see comments below
    values: torch.Tensor,                # (bs, L)
    response_mask: torch.Tensor,         # (bs, L), 1 for valid tokens, 0 after EOS
    gamma: torch.Tensor,                 # scalar or 0-d tensor
    lam: torch.Tensor,                   # scalar or 0-d tensor
    turn_indexs,
    traj_indexs,
    **kwargs,
):
    """
    Compute GAE in a non-concatenated PPO setting.

    Differences from standard token-level GAE:
    - Each trajectory consists of multiple turns (samples).
    - Each turn is first reduced to a scalar reward.
    - GAE is computed at the turn level (backward in time).
    - All valid tokens within the same turn share the same advantage.
    - The return is written only to the last valid token of each turn;
      all other token positions are filled with -100.

    Args:
        token_level_scores:
            (bs, L) token-level scores used to compute turn-level reward.
        token_level_rewards:
            (bs, L) token-level rewards (not used here by design).
        values:
            (bs, L) token-level value predictions.
        response_mask:
            (bs, L) mask indicating valid response tokens (1) and padding/EOS (0).
        gamma:
            Discount factor.
        lam:
            GAE lambda.
        turn_indexs:
            Turn indices for each sample in the batch.
        traj_indexs:
            Trajectory indices for each sample in the batch.

    Returns:
        advantages:
            (bs, L) token-level advantages.
            All valid tokens in the same turn share the same advantage.
        returns:
            (bs, L) token-level returns.
            Only the last valid token of each turn has a return value;
            all other positions are set to -100.
    """
    with torch.no_grad():
        device = token_level_scores.device
        bs, L = token_level_scores.shape

        # Convert mask to float and bool versions
        mask_f = response_mask.to(dtype=token_level_scores.dtype, device=device)
        mask_b = response_mask.to(dtype=torch.bool, device=device)

        # ------------------------------------------------------------------
        # 1) Compute a scalar reward per turn by aggregating token-level scores
        # ------------------------------------------------------------------
        # Turn reward = sum of token-level scores over valid tokens
        turn_rewards = (token_level_scores * mask_f).sum(dim=1)  # (bs,)

        # NOTE:
        # If you want to use token_level_rewards instead, replace the line above with:
        # turn_rewards = (token_level_rewards * mask_f).sum(dim=1)
        # Or combine both:
        # turn_rewards = ((token_level_scores + token_level_rewards) * mask_f).sum(dim=1)

        # ------------------------------------------------------------------
        # 2) Extract a single value per turn
        #    Use the value at the last valid token position
        # ------------------------------------------------------------------
        turn_values = torch.zeros(bs, dtype=values.dtype, device=device)
        last_token_pos = torch.full((bs,), -1, dtype=torch.long, device=device)

        for i in range(bs):
            valid_idx = torch.nonzero(mask_b[i], as_tuple=False).view(-1)
            if valid_idx.numel() > 0:
                last = valid_idx[-1].item()
                last_token_pos[i] = last
                turn_values[i] = values[i, last]
            else:
                # Edge case: no valid tokens in this turn
                turn_values[i] = 0.0
                last_token_pos[i] = -1

        # ------------------------------------------------------------------
        # 3) Group turns by trajectory and compute turn-level GAE
        # ------------------------------------------------------------------
        traj_ids = torch.as_tensor(traj_indexs, dtype=torch.long, device="cpu")
        turn_ids = torch.as_tensor(turn_indexs, dtype=torch.long, device="cpu")

        advantages = torch.zeros((bs, L), dtype=values.dtype, device=device)
        returns = torch.full((bs, L), -100.0, dtype=values.dtype, device=device)

        gamma_val = float(gamma.item() if isinstance(gamma, torch.Tensor) else gamma)
        lam_val = float(lam.item() if isinstance(lam, torch.Tensor) else lam)

        for traj in torch.unique(traj_ids).tolist():
            traj_mask = torch.nonzero(traj_ids == traj, as_tuple=False).view(-1)
            if traj_mask.numel() == 0:
                continue

            # Sort turns in this trajectory by turn index
            sorted_turns = traj_mask[torch.argsort(turn_ids[traj_mask])]
            sorted_turns = sorted_turns.tolist()

            next_value = 0.0
            lastgaelam = 0.0

            # Backward GAE over turns
            for b in reversed(sorted_turns):
                r = float(turn_rewards[b].item())
                v = float(turn_values[b].item())

                delta = r + gamma_val * next_value - v
                lastgaelam = delta + gamma_val * lam_val * lastgaelam

                adv = lastgaelam
                ret = adv + v

                # Assign the same advantage to all valid tokens in this turn
                if last_token_pos[b].item() >= 0:
                    advantages[b, mask_b[b]] = adv

                    # Write return only at the last valid token
                    returns[b, last_token_pos[b]] = ret

                next_value = v

        # ------------------------------------------------------------------
        # 4) Optional: masked whitening of advantages (token-level)
        # ------------------------------------------------------------------
        advantages = verl_F.masked_whiten(advantages, mask_f)

    return advantages, returns