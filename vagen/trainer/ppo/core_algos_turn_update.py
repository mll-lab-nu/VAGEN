import numpy as np
import torch
from collections import defaultdict
from verl import DataProto
import verl.utils.torch_functional as verl_F
import torch.nn.functional as F

# ============= Utility Functions =============

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


def compute_turn_level_advantages(trajectories: dict, values: torch.Tensor, 
                                 loss_mask: torch.Tensor, turn_rewards: np.ndarray,
                                 high_level_gamma: float, lam: float,
                                 include_token_rewards: bool = False,
                                 token_level_rewards: torch.Tensor = None):
    """
    Compute turn-level advantages for all trajectories.
    
    Args:
        include_token_rewards: Whether to include token-level rewards in turn reward
        token_level_rewards: Required if include_token_rewards is True
    
    Returns:
        turn_level_advantages: dict mapping (env_id, turn_id) to advantage
        turn_level_returns: dict mapping (env_id, turn_id) to return
    """
    turn_level_advantages = {}
    turn_level_returns = {}
    
    for env_id, turns in trajectories.items():
        lastgaelam = 0.0
        
        for i in range(len(turns) - 1, -1, -1):
            turn_id, batch_idx = turns[i]
            
            # Get valid positions
            valid_positions = get_valid_positions_with_check(
                loss_mask, batch_idx, env_id, turn_id
            )
            
            # Use the value at the first valid position
            first_valid_pos = valid_positions[0].item()
            turn_reward = float(turn_rewards[batch_idx])
            
            # Optionally include token-level rewards
            if include_token_rewards:
                token_rewards_sum = token_level_rewards[batch_idx][valid_positions].sum().item()
                turn_reward += token_rewards_sum
            
            turn_value = values[batch_idx, first_valid_pos].item()
            
            # Get next turn's value
            if i < len(turns) - 1:
                next_turn_id, next_batch_idx = turns[i + 1]
                next_valid_positions = get_valid_positions_with_check(
                    loss_mask, next_batch_idx, env_id, next_turn_id
                )
                next_first_valid_pos = next_valid_positions[0].item()
                next_value = values[next_batch_idx, next_first_valid_pos].item()
            else:
                next_value = 0.0
            
            # Calculate turn-level advantage
            delta = turn_reward + high_level_gamma * next_value - turn_value
            lastgaelam = delta + high_level_gamma * lam * lastgaelam
            
            turn_level_advantages[(env_id, turn_id)] = lastgaelam
            turn_level_returns[(env_id, turn_id)] = lastgaelam + turn_value
    
    return turn_level_advantages, turn_level_returns


# ============= Main GAE Functions =============

def compute_turn_update_gae_advantage_return(
        token_level_rewards: torch.Tensor,
        values: torch.Tensor,
        loss_mask: torch.Tensor,
        env_ids: np.ndarray,
        turn_ids: np.ndarray,
        turn_rewards: np.ndarray,
        gamma: float,
        lam: float,
    ):
    """Turn-based GAE calculation that treats the entire trajectory as one sequence."""
    with torch.no_grad():
        batch_size, response_length = values.shape
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        
        # Create updated rewards
        updated_rewards = token_level_rewards.clone()
        for b in range(batch_size):
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            if len(valid_positions) > 0:
                last_valid_pos = valid_positions[-1].item()
                updated_rewards[b, last_valid_pos] += float(turn_rewards[b])
        
        # Use utility functions
        unique_turns, duplicate_turns = identify_unique_and_duplicate_turns(env_ids, turn_ids)
        trajectories = group_turns_by_trajectory(unique_turns)
        
        # Process each trajectory
        for env_id, turns in trajectories.items():
            lastgaelam = 0.0
            next_value = 0.0
            
            for i in range(len(turns) - 1, -1, -1):
                turn_id, batch_idx = turns[i]
                valid_positions = loss_mask[batch_idx].nonzero(as_tuple=True)[0]
                
                # Backward pass through valid tokens
                for j in range(len(valid_positions) - 1, -1, -1):
                    curr_pos = valid_positions[j].item()
                    delta = updated_rewards[batch_idx, curr_pos].item() + gamma * next_value - values[batch_idx, curr_pos].item()
                    lastgaelam = delta + gamma * lam * lastgaelam
                    
                    advantages[batch_idx, curr_pos] = lastgaelam
                    returns[batch_idx, curr_pos] = lastgaelam + values[batch_idx, curr_pos].item()
        
        # Handle duplicates
        apply_duplicate_values(duplicate_turns, unique_turns, env_ids, turn_ids, 
                             advantages, returns)
        
        # Whiten advantages
        advantages = verl_F.masked_whiten(advantages, loss_mask)
    
    return advantages, returns


def compute_turn_update_high_level_gae_advantage_return(
        token_level_rewards: torch.Tensor,
        values: torch.Tensor,
        loss_mask: torch.Tensor,
        env_ids: np.ndarray,
        turn_ids: np.ndarray,
        turn_rewards: np.ndarray,
        high_level_gamma: float,
        lam: float,
    ):
    """High-level GAE calculation that computes advantages at the turn level only."""
    with torch.no_grad():
        batch_size, response_length = values.shape
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        
        # Use utility functions
        unique_turns, duplicate_turns = identify_unique_and_duplicate_turns(env_ids, turn_ids)
        trajectories = group_turns_by_trajectory(unique_turns)
        
        # Compute turn-level advantages (including token rewards)
        turn_level_advantages, turn_level_returns = compute_turn_level_advantages(
            trajectories, values, loss_mask, turn_rewards, 
            high_level_gamma, lam, 
            include_token_rewards=True,
            token_level_rewards=token_level_rewards
        )
        
        # Apply turn-level advantages to all tokens in each turn
        for (env_id, turn_id), batch_idx in unique_turns.items():
            turn_advantage = turn_level_advantages.get((env_id, turn_id), 0.0)
            turn_return = turn_level_returns.get((env_id, turn_id), 0.0)
            
            valid_positions = get_valid_positions_with_check(
                loss_mask, batch_idx, env_id, turn_id
            )
            
            # Set the same advantage for all valid tokens
            for pos in valid_positions:
                advantages[batch_idx, pos] = turn_advantage
            
            # Set return only at the first valid position
            first_valid_pos = valid_positions[0].item()
            returns[batch_idx, first_valid_pos] = turn_return
        
        # Handle duplicates
        apply_duplicate_values(duplicate_turns, unique_turns, env_ids, turn_ids, 
                             advantages, returns)
        
        # Whiten advantages
        advantages = verl_F.masked_whiten(advantages, loss_mask)
    
    return advantages, returns


def compute_turn_update_bi_level_gae_advantage_return(
        token_level_rewards: torch.Tensor,
        values: torch.Tensor,
        loss_mask: torch.Tensor,
        env_ids: np.ndarray,
        turn_ids: np.ndarray,
        turn_rewards: np.ndarray,
        high_level_gamma: float,
        gamma: float,
        lam: float,
        token_reward_type: str = "return"
    ):
    """Bi-level GAE calculation with turn-wise updates."""
    with torch.no_grad():
        batch_size, response_length = values.shape
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        updated_rewards = token_level_rewards.clone()
        
        # Use utility functions
        unique_turns, duplicate_turns = identify_unique_and_duplicate_turns(env_ids, turn_ids)
        trajectories = group_turns_by_trajectory(unique_turns)
        
        # Compute turn-level advantages (without token rewards)
        turn_level_advantages, _ = compute_turn_level_advantages(
            trajectories, values, loss_mask, turn_rewards, 
            high_level_gamma, lam, 
            include_token_rewards=False
        )
        
        # Update rewards with turn-level results
        for (env_id, turn_id), batch_idx in unique_turns.items():
            valid_positions = get_valid_positions_with_check(
                loss_mask, batch_idx, env_id, turn_id
            )
            
            first_valid_pos = valid_positions[0].item()
            last_valid_pos = valid_positions[-1].item()
            turn_value = values[batch_idx, first_valid_pos].item()
            turn_advantage = turn_level_advantages.get((env_id, turn_id), 0.0)
            
            if token_reward_type == "return":
                updated_rewards[batch_idx, last_valid_pos] = turn_advantage + turn_value
            elif token_reward_type == "advantage":
                updated_rewards[batch_idx, last_valid_pos] = turn_advantage
            else:
                raise ValueError(f"token_reward_type must be 'advantage' or 'return', got {token_reward_type}")
        
        # Compute token-level advantages
        for (env_id, turn_id), batch_idx in unique_turns.items():
            valid_positions = get_valid_positions_with_check(
                loss_mask, batch_idx, env_id, turn_id
            )
            
            lastgaelam = 0.0
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i].item()
                
                if i < len(valid_positions) - 1:
                    next_pos = valid_positions[i + 1].item()
                    nextvalue = values[batch_idx, next_pos].item()
                else:
                    nextvalue = 0.0
                    lastgaelam = 0.0
                
                delta = updated_rewards[batch_idx, curr_pos].item() + gamma * nextvalue - values[batch_idx, curr_pos].item()
                lastgaelam = delta + gamma * lam * lastgaelam
                
                advantages[batch_idx, curr_pos] = lastgaelam
                returns[batch_idx, curr_pos] = lastgaelam + values[batch_idx, curr_pos].item()
        
        # Handle duplicates
        apply_duplicate_values(duplicate_turns, unique_turns, env_ids, turn_ids, 
                             advantages, returns)
        
        # Whiten advantages
        advantages = verl_F.masked_whiten(advantages, loss_mask)
    
    return advantages, returns