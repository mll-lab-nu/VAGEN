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



@register_adv_est("no_concat_gae")
def compute_gae_no_concat_advantage_return(
    token_level_scores: torch.Tensor,    # (bs, L)
    token_level_rewards: torch.Tensor,   # (bs, L)  # not used here
    values: torch.Tensor,                # (bs, L)
    response_mask: torch.Tensor,         # (bs, L)
    gamma: torch.Tensor,
    lam: torch.Tensor,
    group_idx,                         # (bs,)
    turn_idx,                         # (bs,)
    traj_idx,                         # (bs,)
    **kwargs,
):
    with torch.no_grad():
        device = token_level_scores.device
        bs, L = token_level_scores.shape

        mask_f = response_mask.to(dtype=token_level_scores.dtype, device=device)
        mask_b = response_mask.to(dtype=torch.bool, device=device)

        # 1) turn-level reward
        turn_rewards = (token_level_scores * mask_f).sum(dim=1)  # (bs,)

        # 2) turn-level value = value at last valid token
        turn_values = torch.zeros(bs, dtype=values.dtype, device=device)
        last_token_pos = torch.full((bs,), -1, dtype=torch.long, device=device)

        for i in range(bs):
            valid_idx = torch.nonzero(mask_b[i], as_tuple=False).view(-1)
            if valid_idx.numel() > 0:
                last = valid_idx[-1].item()
                last_token_pos[i] = last
                turn_values[i] = values[i, last]
            else:
                turn_values[i] = 0.0
                last_token_pos[i] = -1

        # 3) group by (group_idx, traj_idx) and compute turn-level GAE
        traj_ids = torch.as_tensor(traj_idx, dtype=torch.long, device="cpu")
        group_ids = torch.as_tensor(group_idx, dtype=torch.long, device="cpu")
        turn_ids = torch.as_tensor(turn_idx, dtype=torch.long, device="cpu")

        advantages = torch.zeros((bs, L), dtype=values.dtype, device=device)
        returns = torch.full((bs, L), -100.0, dtype=values.dtype, device=device)

        gamma_val = float(gamma.item() if isinstance(gamma, torch.Tensor) else gamma)
        lam_val = float(lam.item() if isinstance(lam, torch.Tensor) else lam)

        # ---- NEW: unique keys are pairs (group_id, traj_id)
        keys = torch.stack([group_ids, traj_ids], dim=1)          # (bs, 2) on CPU
        unique_keys = torch.unique(keys, dim=0)                   # (num_groups, 2)

        for g_id, t_id in unique_keys.tolist():
            pair_mask = torch.nonzero(
                (group_ids == g_id) & (traj_ids == t_id),
                as_tuple=False
            ).view(-1)

            if pair_mask.numel() == 0:
                continue

            # Sort turns inside this (group_id, traj_id) by traj_idx
            sorted_turns = pair_mask[torch.argsort(turn_ids[pair_mask])].tolist()

            next_value = 0.0
            lastgaelam = 0.0

            for b in reversed(sorted_turns):
                r = float(turn_rewards[b].item())
                v = float(turn_values[b].item())

                delta = r + gamma_val * next_value - v
                lastgaelam = delta + gamma_val * lam_val * lastgaelam

                adv = lastgaelam
                ret = adv + v

                if last_token_pos[b].item() >= 0:
                    advantages[b, mask_b[b]] = adv
                    returns[b, last_token_pos[b]] = ret

                next_value = v

        # 4) optional whitening
        advantages = verl_F.masked_whiten(advantages, mask_f)

    return advantages, returns
