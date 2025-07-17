import numpy as np
import torch
from collections import defaultdict
from verl import DataProto
import verl.utils.torch_functional as verl_F
import torch.nn.functional as F
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
    """Turn-based GAE calculation that treats the entire trajectory as one sequence.
    
    This version first assigns turn rewards to the last valid token of each turn,
    then computes GAE backward from the last valid token to the first, skipping
    positions where loss_mask is 0.
    
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length). Per-token rewards (e.g., KL penalties).
        values: `(torch.Tensor)`
            shape: (bs, response_length). Value estimates for each token.
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for valid tokens, 0 for padding/prompts.
        env_ids: `(np.ndarray)`
            Environment/trajectory identifiers (can be any hashable type).
        turn_ids: `(np.ndarray)`
            Step within trajectory (must be numeric).
        turn_rewards: `(np.ndarray)`
            Turn-level rewards (scalar per turn).
        gamma: `(float)`
            Discount factor for rewards.
        lam: `(float)`
            Lambda value for GAE computation.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        batch_size, response_length = values.shape
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        
        # Step 1: Create a copy of token_level_rewards and add turn rewards to last valid positions
        updated_rewards = token_level_rewards.clone()
        
        for b in range(batch_size):
            # Find the last valid position in this turn
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            if len(valid_positions) > 0:
                last_valid_pos = valid_positions[-1].item()
                # Add turn reward to the last valid token
                updated_rewards[b, last_valid_pos] += float(turn_rewards[b])
        
        # Step 2: Identify unique turns (handle duplicates from padding)
        unique_turns = {}
        duplicate_turns = []
        
        for b in range(batch_size):
            env_id = env_ids[b]
            turn_id = int(turn_ids[b])
            turn_key = (env_id, turn_id)
            
            if turn_key not in unique_turns:
                unique_turns[turn_key] = b
            else:
                duplicate_turns.append(b)
        
        # Step 3: Group turns by trajectory (env_id)
        trajectories = {}
        for (env_id, turn_id), batch_idx in unique_turns.items():
            if env_id not in trajectories:
                trajectories[env_id] = []
            trajectories[env_id].append((turn_id, batch_idx))
        
        # Sort turns within each trajectory by turn_id
        for env_id in trajectories:
            trajectories[env_id].sort(key=lambda x: x[0])
        
        # Step 4: Process each trajectory
        for env_id, turns in trajectories.items():
            
            
            lastgaelam = 0.0
            next_value = 0.0
            for i in range(len(turns) - 1, -1, -1):
                turn_id, batch_idx = turns[i]
                
                valid_positions = loss_mask[batch_idx].nonzero(as_tuple=True)[0]
                
                # Backward pass through valid tokens within this turn
                for j in range(len(valid_positions) - 1, -1, -1):
                    curr_pos = valid_positions[j].item()
                    # Calculate advantage using the updated rewards
                    delta = updated_rewards[batch_idx, curr_pos].item() + gamma * next_value - values[batch_idx, curr_pos].item()
                    lastgaelam = delta + gamma * lam * lastgaelam
                    
                    advantages[batch_idx, curr_pos] = lastgaelam
                    returns[batch_idx, curr_pos] = lastgaelam + values[batch_idx, curr_pos].item()
        
        # Step 5: Handle duplicate turns by copying computed values
        for dup_idx in duplicate_turns:
            env_id = env_ids[dup_idx]
            turn_id = int(turn_ids[dup_idx])
            orig_idx = unique_turns[(env_id, turn_id)]
            
            advantages[dup_idx] = advantages[orig_idx]
            returns[dup_idx] = returns[orig_idx]
        
        # Step 6: Whiten advantages
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
    """High-level GAE calculation that computes advantages at the turn level only.
    
    This version computes advantages only between turns (not between tokens),
    then assigns the same advantage to all tokens within each turn.
    Returns are only meaningful at the first token of each turn.
    
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length). Per-token rewards (e.g., KL penalties).
        values: `(torch.Tensor)`
            shape: (bs, response_length). Value estimates for each token.
            IMPORTANT: values[i] is the value estimate BEFORE generating token[i].
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for valid tokens, 0 for padding/prompts.
        env_ids: `(np.ndarray)`
            Environment/trajectory identifiers (can be any hashable type).
        turn_ids: `(np.ndarray)`
            Step within trajectory (must be numeric).
        turn_rewards: `(np.ndarray)`
            Turn-level rewards (scalar per turn).
        high_level_gamma: `(float)`
            Discount factor for turn-level rewards.
        lam: `(float)`
            Lambda value for GAE computation.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length). Same advantage value for all tokens in a turn.
        returns: `(torch.Tensor)`
            shape: (bs, response_length). Non-zero only at the first token of each turn.
    """
    with torch.no_grad():
        batch_size, response_length = values.shape
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        
        # Step 1: Identify unique turns (handle duplicates from padding)
        unique_turns = {}
        duplicate_turns = []
        
        for b in range(batch_size):
            env_id = env_ids[b]
            turn_id = int(turn_ids[b])
            turn_key = (env_id, turn_id)
            
            if turn_key not in unique_turns:
                unique_turns[turn_key] = b
            else:
                duplicate_turns.append(b)
        
        # Step 2: Group turns by trajectory (env_id)
        trajectories = {}
        for (env_id, turn_id), batch_idx in unique_turns.items():
            if env_id not in trajectories:
                trajectories[env_id] = []
            trajectories[env_id].append((turn_id, batch_idx))
        
        # Sort turns within each trajectory by turn_id
        for env_id in trajectories:
            trajectories[env_id].sort(key=lambda x: x[0])
        
        # Step 3: Compute high-level advantages for each trajectory
        turn_level_advantages = {}
        turn_level_returns = {}
        
        for env_id, turns in trajectories.items():
            # For each trajectory, compute advantages backward through turns
            lastgaelam = 0.0
            
            for i in range(len(turns) - 1, -1, -1):
                turn_id, batch_idx = turns[i]
                
                # Get valid positions for this turn
                valid_positions = loss_mask[batch_idx].nonzero(as_tuple=True)[0]
                assert len(valid_positions) > 0, f"No valid positions found for turn (env_id={env_id}, turn_id={turn_id}, batch_idx={batch_idx})"
                
                # Use the value at the first valid position (before generating this turn)
                first_valid_pos = valid_positions[0].item()
                turn_reward = float(turn_rewards[batch_idx])
                
                # Add token-level rewards (e.g., KL penalties) to turn reward
                # Sum all token rewards for this turn
                token_rewards_sum = token_level_rewards[batch_idx][valid_positions].sum().item()
                total_turn_reward = turn_reward + token_rewards_sum
                
                turn_value = values[batch_idx, first_valid_pos].item()
                
                # Get next turn's value
                if i < len(turns) - 1:
                    next_turn_id, next_batch_idx = turns[i + 1]
                    next_valid_positions = loss_mask[next_batch_idx].nonzero(as_tuple=True)[0]
                    assert len(next_valid_positions) > 0, f"No valid positions found for next turn (env_id={env_id}, turn_id={next_turn_id}, batch_idx={next_batch_idx})"
                    # Use the first valid position of next turn
                    next_first_valid_pos = next_valid_positions[0].item()
                    next_value = values[next_batch_idx, next_first_valid_pos].item()
                else:
                    next_value = 0.0
                
                # Calculate turn-level advantage
                delta = total_turn_reward + high_level_gamma * next_value - turn_value
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                
                turn_level_advantages[(env_id, turn_id)] = lastgaelam
                turn_level_returns[(env_id, turn_id)] = lastgaelam + turn_value
        
        # Step 4: Apply turn-level advantages to all tokens in each turn
        for (env_id, turn_id), batch_idx in unique_turns.items():
            turn_advantage = turn_level_advantages.get((env_id, turn_id), 0.0)
            turn_return = turn_level_returns.get((env_id, turn_id), 0.0)
            
            # Find valid positions for this turn
            valid_positions = loss_mask[batch_idx].nonzero(as_tuple=True)[0]
            assert len(valid_positions) > 0, f"No valid positions found for turn (env_id={env_id}, turn_id={turn_id}, batch_idx={batch_idx})"
            
            # Set the same advantage for all valid tokens in this turn
            for pos in valid_positions:
                advantages[batch_idx, pos] = turn_advantage
            
            # Set return only at the first valid position
            first_valid_pos = valid_positions[0].item()
            returns[batch_idx, first_valid_pos] = turn_return
        
        # Step 5: Handle duplicate turns by copying computed values
        for dup_idx in duplicate_turns:
            env_id = env_ids[dup_idx]
            turn_id = int(turn_ids[dup_idx])
            orig_idx = unique_turns[(env_id, turn_id)]
            
            advantages[dup_idx] = advantages[orig_idx]
            returns[dup_idx] = returns[orig_idx]
        
        # Step 6: Whiten advantages
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
        token_reward_type: str = "advantage"  # "advantage" or "return"
    ):
    """Bi-level GAE calculation with turn-wise updates.
    
    First computes turn-level advantages using only turn rewards,
    then uses these advantages/returns as rewards to compute token-level advantages.
    
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length). Per-token rewards (e.g., KL penalties).
        values: `(torch.Tensor)`
            shape: (bs, response_length). Value estimates for each token.
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for valid tokens, 0 for padding/prompts.
        env_ids: `(np.ndarray)`
            Environment/trajectory identifiers (can be any hashable type).
        turn_ids: `(np.ndarray)`
            Step within trajectory (must be numeric).
        turn_rewards: `(np.ndarray)`
            Turn-level rewards (scalar per turn).
        high_level_gamma: `(float)`
            Discount factor for turn-level rewards.
        gamma: `(float)`
            Discount factor for token-level rewards.
        lam: `(float)`
            Lambda value for GAE computation.
        token_reward_type: `(str)`
            "advantage": use turn advantage as reward for token-level computation
            "return": use turn return as reward for token-level computation
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        batch_size, response_length = values.shape
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        
        # Step 1: Identify unique turns (handle duplicates from padding)
        unique_turns = {}
        duplicate_turns = []
        
        for b in range(batch_size):
            env_id = env_ids[b]
            turn_id = int(turn_ids[b])
            turn_key = (env_id, turn_id)
            
            if turn_key not in unique_turns:
                unique_turns[turn_key] = b
            else:
                duplicate_turns.append(b)
        
        # Step 2: Group turns by trajectory (env_id)
        trajectories = {}
        for (env_id, turn_id), batch_idx in unique_turns.items():
            if env_id not in trajectories:
                trajectories[env_id] = []
            trajectories[env_id].append((turn_id, batch_idx))
        
        # Sort turns within each trajectory by turn_id
        for env_id in trajectories:
            trajectories[env_id].sort(key=lambda x: x[0])
        
        # Step 3: Compute high-level (turn-level) advantages using only turn rewards
        turn_level_advantages = {}
        turn_level_returns = {}
        
        for env_id, turns in trajectories.items():
            # For each trajectory, compute advantages backward through turns
            lastgaelam = 0.0
            
            for i in range(len(turns) - 1, -1, -1):
                turn_id, batch_idx = turns[i]
                
                # Get the last valid position for this turn
                valid_positions = loss_mask[batch_idx].nonzero(as_tuple=True)[0]
                if len(valid_positions) == 0:
                    print(f"[DEBUG] No valid positions for turn {turn_id} in batch {batch_idx}")
                    continue
                
                # Use the value at the last valid position
                last_valid_pos = valid_positions[-1].item()
                turn_reward = float(turn_rewards[batch_idx])
                turn_value = values[batch_idx, last_valid_pos].item()
                
                # Get next turn's value
                if i < len(turns) - 1:
                    next_turn_id, next_batch_idx = turns[i + 1]
                    next_valid_positions = loss_mask[next_batch_idx].nonzero(as_tuple=True)[0]
                    if len(next_valid_positions) > 0:
                        # Use the first valid position of next turn
                        next_first_valid_pos = next_valid_positions[0].item()
                        next_value = values[next_batch_idx, next_first_valid_pos].item()
                    else:
                        next_value = 0.0
                else:
                    next_value = 0.0
                
                # Calculate turn-level advantage
                delta = turn_reward + high_level_gamma * next_value - turn_value
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                
                turn_level_advantages[(env_id, turn_id)] = lastgaelam
                turn_level_returns[(env_id, turn_id)] = lastgaelam + turn_value
        
        # Step 4: Compute token-level advantages using turn-level results as rewards
        for (env_id, turn_id), batch_idx in unique_turns.items():
            turn_advantage = turn_level_advantages.get((env_id, turn_id), 0.0)
            turn_return = turn_level_returns.get((env_id, turn_id), 0.0)
            
            # Find valid positions for this turn
            valid_positions = loss_mask[batch_idx].nonzero(as_tuple=True)[0]
            if len(valid_positions) == 0:
                print(f"[DEBUG] No valid positions for turn {turn_id} in batch {batch_idx}")
                continue
            
            # Determine what to use as the reward for token-level computation
            if token_reward_type == "advantage":
                turn_level_reward = turn_advantage
            elif token_reward_type == "return":
                turn_level_reward = turn_return
            else:
                raise ValueError(f"token_reward_type must be 'advantage' or 'return', got {token_reward_type}")
            
            # Create rewards for token-level computation
            # Combine token-level rewards (KL penalties) with turn-level reward at last position
            token_rewards = token_level_rewards[batch_idx].clone()
            last_valid_pos = valid_positions[-1].item()
            token_rewards[last_valid_pos] += turn_level_reward
            
            # Compute token-level advantages within this turn
            lastgaelam = 0.0
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i].item()
                
                # Get next value
                if i < len(valid_positions) - 1:
                    next_pos = valid_positions[i + 1].item()
                    nextvalue = values[batch_idx, next_pos].item()
                else:
                    # Last token in turn
                    nextvalue = 0.0
                
                # Calculate token-level advantage
                delta = token_rewards[curr_pos].item() + gamma * nextvalue - values[batch_idx, curr_pos].item()
                lastgaelam = delta + gamma * lam * lastgaelam
                
                advantages[batch_idx, curr_pos] = lastgaelam
                returns[batch_idx, curr_pos] = lastgaelam + values[batch_idx, curr_pos].item()
        
        # Step 5: Handle duplicate turns by copying computed values
        for dup_idx in duplicate_turns:
            env_id = env_ids[dup_idx]
            turn_id = int(turn_ids[dup_idx])
            orig_idx = unique_turns[(env_id, turn_id)]
            
            advantages[dup_idx] = advantages[orig_idx]
            returns[dup_idx] = returns[orig_idx]
        
        # Step 6: Whiten advantages
        advantages = verl_F.masked_whiten(advantages, loss_mask)
    
    return advantages, returns
