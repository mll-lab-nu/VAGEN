"""Utility helpers for PPO training (VAGEN).

Currently provides:
    • seed_everything(seed: int) – set deterministic seed for Python `random`,
      NumPy, torch (CPU & CUDA).  Also sets `PYTHONHASHSEED` and forces
      deterministic cuDNN behaviour.

The helper is intentionally lightweight so it can be imported both in driver
code and inside Ray worker actors without introducing circular dependencies.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

__all__ = ["seed_everything"]


def seed_everything(seed: int, *, deterministic: bool = True, warn: bool = True) -> None:
    """Seed *all* common PRNGs to make experiment deterministic.

    Parameters
    ----------
    seed : int
        The random seed to set.
    deterministic : bool, default True
        If *True* will apply extra flags to make cuDNN deterministic
        (slower but reproducible).  Can be disabled if performance is
        preferred over bit-wise reproducibility.
    warn : bool, default True
        If *True* prints a short message when called multiple times.
    """
    # Prevent accidental reseeding in the same process unless user opts out.
    if hasattr(seed_everything, "_seeded") and seed_everything._seeded and warn:
        print("[seed_everything] WARNING: reseeding the same Python process.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything._seeded = True  # type: ignore[attr-defined]



# ============= Utility Functions for advantage computation =============

def identify_unique_and_duplicate_turns(env_ids: np.ndarray, turn_ids: np.ndarray):
    """
    Identify unique turns and duplicates from batch data.
    
    Returns:
        unique_turns: dict mapping (env_id, turn_id) to batch index
        duplicate_turns: list of batch indices that are duplicates
    """
    unique_turns = {}
    duplicate_turns = []
    
    batch_size = len(env_ids)
    for b in range(batch_size):
        env_id = env_ids[b]
        turn_id = int(turn_ids[b])
        turn_key = (env_id, turn_id)
        
        if turn_key not in unique_turns:
            unique_turns[turn_key] = b
        else:
            duplicate_turns.append(b)
    
    return unique_turns, duplicate_turns


def group_turns_by_trajectory(unique_turns: dict):
    """
    Group turns by trajectory (env_id) and sort by turn_id.
    
    Returns:
        trajectories: dict mapping env_id to sorted list of (turn_id, batch_idx) tuples
    """
    trajectories = {}
    for (env_id, turn_id), batch_idx in unique_turns.items():
        if env_id not in trajectories:
            trajectories[env_id] = []
        trajectories[env_id].append((turn_id, batch_idx))
    
    # Sort turns within each trajectory by turn_id
    for env_id in trajectories:
        trajectories[env_id].sort(key=lambda x: x[0])
    
    return trajectories


def get_valid_positions_with_check(loss_mask: torch.Tensor, batch_idx: int, 
                                  env_id, turn_id: int):
    """
    Get valid positions from loss_mask with assertion check.
    
    Returns:
        valid_positions: tensor of valid position indices
    """
    valid_positions = loss_mask[batch_idx].nonzero(as_tuple=True)[0]
    assert len(valid_positions) > 0, (
        f"No valid positions found for turn "
        f"(env_id={env_id}, turn_id={turn_id}, batch_idx={batch_idx})"
    )
    return valid_positions


def apply_duplicate_values(duplicate_turns: list, unique_turns: dict, 
                         env_ids: np.ndarray, turn_ids: np.ndarray,
                         advantages: torch.Tensor, returns: torch.Tensor):
    """
    Copy computed values from unique turns to duplicate turns.
    """
    for dup_idx in duplicate_turns:
        env_id = env_ids[dup_idx]
        turn_id = int(turn_ids[dup_idx])
        orig_idx = unique_turns[(env_id, turn_id)]
        
        advantages[dup_idx] = advantages[orig_idx]
        returns[dup_idx] = returns[orig_idx]


# ============= Utility Functions for turn update grpo advantage computation =============



def compute_discounted_returns(trajectories: dict, unique_turns: dict, 
                              token_level_rewards: torch.Tensor, turn_rewards: np.ndarray,
                              loss_mask: torch.Tensor, high_level_gamma: float):
    """Compute discounted cumulative returns for each turn."""
    turn_returns = {}
    
    for env_id, turns in trajectories.items():
        cumulative_return = 0.0
        
        for i in range(len(turns) - 1, -1, -1):
            turn_id, batch_idx = turns[i]
            
            # Compute total reward for this turn using loss_mask
            masked_token_rewards = token_level_rewards[batch_idx] * loss_mask[batch_idx]
            token_rewards_sum = masked_token_rewards.sum().item()
            turn_reward = float(turn_rewards[batch_idx]) + token_rewards_sum
            
            # Update cumulative return with discounting
            cumulative_return = turn_reward + high_level_gamma * cumulative_return
            turn_returns[(env_id, turn_id)] = cumulative_return
    
    return turn_returns

def compute_turn_total_rewards(unique_turns: dict, token_level_rewards: torch.Tensor, 
                              turn_rewards: np.ndarray, loss_mask: torch.Tensor):
    """Compute total reward (token + turn) for each turn."""
    turn_total_rewards = {}
    
    for (env_id, turn_id), batch_idx in unique_turns.items():
        # Sum token rewards using loss_mask and add turn reward
        masked_token_rewards = token_level_rewards[batch_idx] * loss_mask[batch_idx]
        token_rewards_sum = masked_token_rewards.sum().item()
        total_reward = float(turn_rewards[batch_idx]) + token_rewards_sum
        turn_total_rewards[(env_id, turn_id)] = total_reward
    
    return turn_total_rewards

def normalize_advantages_by_groups(groups: dict, use_std: bool = False, epsilon: float = 1e-8):
    """Normalize advantages within each group."""
    normalized_advantages = {}
    
    for group_id, group_data in groups.items():
        if len(group_data) == 1:
            batch_idx, value = group_data[0]
            normalized_advantages[batch_idx] = 0.0
        else:
            group_values = [value for _, value in group_data]
            mean_value = np.mean(group_values)
            
            if use_std:
                std_value = np.std(group_values)
                if std_value > epsilon:
                    for batch_idx, value in group_data:
                        normalized_advantages[batch_idx] = (value - mean_value) / std_value
                else:
                    for batch_idx, value in group_data:
                        normalized_advantages[batch_idx] = 0.0
            else:
                for batch_idx, value in group_data:
                    normalized_advantages[batch_idx] = value - mean_value
    
    return normalized_advantages

def assign_advantages_to_tokens(normalized_advantages: dict, loss_mask: torch.Tensor):
    """Assign advantage values to all valid tokens."""
    batch_size, response_length = loss_mask.shape
    advantages = torch.zeros(batch_size, response_length)
    
    for batch_idx, advantage_value in normalized_advantages.items():
        valid_positions = loss_mask[batch_idx].nonzero(as_tuple=True)[0]
        advantages[batch_idx, valid_positions] = advantage_value
    
    return advantages