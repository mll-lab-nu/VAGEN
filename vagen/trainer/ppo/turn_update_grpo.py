import numpy as np
import torch
from collections import defaultdict
import verl.utils.torch_functional as verl_F
from .utils import (
    apply_duplicate_values,
    normalize_advantages_by_groups,
    assign_advantages_to_tokens,
    identify_unique_and_duplicate_turns,
    group_turns_by_trajectory,
    compute_discounted_returns,
)


@torch.no_grad()
def grpo_state_traj_advantage(
        token_level_rewards: torch.Tensor,
        loss_mask: torch.Tensor,
        env_ids: np.ndarray,
        turn_ids: np.ndarray,
        turn_rewards: np.ndarray,
        state_ids: np.ndarray,
        uids: np.ndarray,
        high_level_gamma: float,
        turn_advantage_weight: float = 1.0,
        traj_advantage_weight: float = 1.0,
        use_std_normalization: bool = False
    ):
    """GiGPO with both state-level and trajectory-level grouping."""
    
    unique_turns, duplicate_turns = identify_unique_and_duplicate_turns(env_ids, turn_ids)
    trajectories = group_turns_by_trajectory(unique_turns)
    
    # Compute trajectory-level advantages
    turn_returns = compute_discounted_returns(
        trajectories, unique_turns, token_level_rewards, turn_rewards, loss_mask, high_level_gamma
    )
    
    uid_groups = defaultdict(list)
    for (env_id, turn_id), batch_idx in unique_turns.items():
        uid = uids[batch_idx]
        uid_groups[uid].append((batch_idx, turn_returns.get((env_id, turn_id), 0.0)))
    
    traj_advantages = normalize_advantages_by_groups(uid_groups, use_std_normalization)
    
    # Compute state-level advantages using total rewards (token + turn)
    
    state_groups = defaultdict(list)
    for (env_id, turn_id), batch_idx in unique_turns.items():
        state_id = state_ids[batch_idx]
        state_groups[state_id].append((batch_idx, turn_returns.get((env_id, turn_id), 0.0)))
    
    state_advantages = normalize_advantages_by_groups(state_groups, use_std_normalization)
    
    # Combine both advantages
    combined_advantages = {}
    for batch_idx in traj_advantages:
        traj_adv = traj_advantages.get(batch_idx, 0.0)
        state_adv = state_advantages.get(batch_idx, 0.0)
        combined_advantages[batch_idx] = traj_advantage_weight * traj_adv + turn_advantage_weight * state_adv
    
    # Assign to tokens
    advantages = assign_advantages_to_tokens(combined_advantages, loss_mask)
    returns = advantages.clone()
    # Handle duplicates
    apply_duplicate_values(duplicate_turns, unique_turns, env_ids, turn_ids, advantages, returns)
    
    # Final whitening
    advantages = verl_F.masked_whiten(advantages, loss_mask)
    
    
    return advantages, returns