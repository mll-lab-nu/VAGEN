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



from verl.trainer.ppo.core_algos import register_adv_est
import numpy as np
import torch
import verl.utils.torch_functional as verl_F


def _to_numpy_int64(x, factorize_if_non_numeric: bool = False):
    """
    Convert input to np.ndarray(dtype=int64).

    - If x is numeric-like -> direct cast to int64.
    - If x is object/string-like and factorize_if_non_numeric=True:
        encode unique values into contiguous int64 codes (stable per batch).
      This is required when group_idx can be UUID strings.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    # Fast path: already integer
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.int64, copy=False)

    # If object/strings: try numeric cast first, otherwise factorize
    if x.dtype == np.object_ or np.issubdtype(x.dtype, np.str_):
        try:
            return x.astype(np.int64)
        except (ValueError, TypeError):
            if not factorize_if_non_numeric:
                # Give a clearer error
                sample = x.reshape(-1)[:5]
                raise ValueError(
                    f"Cannot cast to int64. Sample values: {sample}. "
                    f"Set factorize_if_non_numeric=True for string-like IDs."
                )
            # Factorize: same original value -> same code
            _, inv = np.unique(x, return_inverse=True)
            return inv.astype(np.int64, copy=False)

    # Fallback: try casting for float/bool/etc.
    return x.astype(np.int64, copy=False)


@register_adv_est("no_concat_gae")
def compute_gae_no_concat_advantage_return(
    token_level_scores: torch.Tensor,    # (bs, L)
    token_level_rewards: torch.Tensor,   # (bs, L)  # not used
    values: torch.Tensor,                # (bs, L)
    response_mask: torch.Tensor,         # (bs, L)
    gamma: torch.Tensor,
    lam: torch.Tensor,
    group_idx,                           # (bs,) may be UUID strings
    turn_idx,                            # (bs,)
    traj_idx,                            # (bs,)
    **kwargs,
):
    with torch.no_grad():
        device = token_level_scores.device
        bs, L = token_level_scores.shape

        mask_f_full = response_mask.to(dtype=token_level_scores.dtype, device=device)
        mask_b_full = response_mask.to(dtype=torch.bool, device=device)

        # ------------------------------------------------------------------
        # 0) Handle padded duplicates:
        #    instances with identical (group_idx, traj_idx, turn_idx) are padding artifacts.
        #    Compute once per unique triple, then broadcast back.
        #
        # IMPORTANT: group_idx may be UUID/string, so we factorize it.
        # ------------------------------------------------------------------
        g_np = _to_numpy_int64(group_idx, factorize_if_non_numeric=True)
        t_np = _to_numpy_int64(traj_idx, factorize_if_non_numeric=False)
        u_np = _to_numpy_int64(turn_idx, factorize_if_non_numeric=False)

        key3 = np.stack([g_np, t_np, u_np], axis=1)  # (bs, 3) int64

        uniq_key3, uniq_first_idx, inverse = np.unique(
            key3, axis=0, return_index=True, return_inverse=True
        )
        uniq_first_idx = uniq_first_idx.astype(np.int64, copy=False)
        bs_u = uniq_first_idx.shape[0]

        uniq_first_idx_t = torch.as_tensor(uniq_first_idx, dtype=torch.long, device=device)

        scores = token_level_scores.index_select(0, uniq_first_idx_t)  # (bs_u, L)
        vals   = values.index_select(0, uniq_first_idx_t)              # (bs_u, L)
        mask_f = mask_f_full.index_select(0, uniq_first_idx_t)         # (bs_u, L)
        mask_b = mask_b_full.index_select(0, uniq_first_idx_t)         # (bs_u, L)

        group_ids = torch.as_tensor(uniq_key3[:, 0], dtype=torch.long, device="cpu")
        traj_ids  = torch.as_tensor(uniq_key3[:, 1], dtype=torch.long, device="cpu")
        turn_ids  = torch.as_tensor(uniq_key3[:, 2], dtype=torch.long, device="cpu")

        # ------------------------------------------------------------------
        # 1) Turn-level reward: sum of token-level scores within the response
        # ------------------------------------------------------------------
        turn_rewards = (scores * mask_f).sum(dim=1)  # (bs_u,)

        # ------------------------------------------------------------------
        # 2) Turn-level value: value at the last valid token (vectorized)
        # ------------------------------------------------------------------
        valid_counts = mask_b.sum(dim=1)  # (bs_u,)
        arange_L = torch.arange(L, device=device).view(1, L).expand(bs_u, L)

        last_pos = (mask_b.to(torch.long) * arange_L).max(dim=1).values
        last_pos = torch.where(valid_counts > 0, last_pos, torch.full_like(last_pos, -1))

        gather_pos = last_pos.clamp(min=0).view(bs_u, 1)
        gathered = vals.gather(1, gather_pos).squeeze(1)

        turn_values = torch.zeros(bs_u, dtype=vals.dtype, device=device)
        turn_values = torch.where(last_pos >= 0, gathered, turn_values)

        # ------------------------------------------------------------------
        # 3) Compute turn-level GAE grouped by (group_idx, traj_idx)
        # ------------------------------------------------------------------
        advantages_u = torch.zeros((bs_u, L), dtype=vals.dtype, device=device)
        returns_u = torch.full((bs_u, L), -100.0, dtype=vals.dtype, device=device)

        gamma_val = float(gamma.item() if isinstance(gamma, torch.Tensor) else gamma)
        lam_val = float(lam.item() if isinstance(lam, torch.Tensor) else lam)

        key2 = torch.stack([group_ids, traj_ids], dim=1)
        unique_key2 = torch.unique(key2, dim=0)

        for g_id, tr_id in unique_key2.tolist():
            pair_mask = torch.nonzero(
                (group_ids == g_id) & (traj_ids == tr_id),
                as_tuple=False
            ).view(-1)

            if pair_mask.numel() == 0:
                continue

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

                lp = int(last_pos[b].item())
                if lp >= 0:
                    advantages_u[b, mask_b[b]] = adv
                    returns_u[b, lp] = ret

                next_value = v

        # ------------------------------------------------------------------
        # 4) Optional whitening (unique samples only)
        # ------------------------------------------------------------------
        advantages_u = verl_F.masked_whiten(advantages_u, mask_f)

        # ------------------------------------------------------------------
        # 5) Broadcast back to full batch (including padded duplicates)
        # ------------------------------------------------------------------
        inverse_t = torch.as_tensor(inverse, dtype=torch.long, device=device)
        advantages_full = advantages_u.index_select(0, inverse_t)  # (bs, L)
        returns_full = returns_u.index_select(0, inverse_t)        # (bs, L)

    return advantages_full, returns_full