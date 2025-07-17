# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict
from verl import DataProto
import verl.utils.torch_functional as verl_F
import torch.nn.functional as F

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return_with_loss_mask(token_level_rewards: torch.Tensor, values: torch.Tensor, 
                                 loss_mask: torch.Tensor, gamma: float, lam: float):
    """Modified GAE calculation that handle multi-turn with loss mask
    Here we should also ensure that the trajectory score is given at the last valid token instead of last token
    Seems it's true in reward manager
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        
        for b in range(batch_size):
            lastgaelam = 0.0
            
            # Find the valid token positions (where loss_mask is 1)
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            
            if len(valid_positions) == 0:
                print(f"[DEBUG] No valid positions for batch {b}")
                continue
                
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                
                # Get the next value
                if i < len(valid_positions) - 1:
                    # Next valid position
                    next_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_pos]
                    
                else:
                    # Last valid position
                    nextvalue = 0.0
                
                # Calculate delta using the next valid token
                delta = token_level_rewards[b, curr_pos] + gamma * nextvalue - values[b, curr_pos]
                
                # Update advantage estimate
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
            
            # Calculate returns for valid positions
            for i, pos in enumerate(valid_positions):
                returns[b, pos] = advantages[b, pos] + values[b, pos]
        

        advantages = verl_F.masked_whiten(advantages, loss_mask)
        
    return advantages, returns



def compute_bi_level_gae_advantage_return(
        token_level_rewards: torch.Tensor,
        values: torch.Tensor, 
        eos_mask: torch.Tensor,
        sos_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        lam: float,
        high_level_gamma: float,
        gamma: float,
    ):
    """Modified GAE calculation that compute two level of advantage and return:
    high level: per-turn wise
    low level: token wise
    there're two level of MDP, where high level is the agentic MDP and low level is the token MDP
    Args:
        token_level_rewards: `(torch.Tensor)` (multi-turn reward, per turn reward is given at eos token for each response token sequence)
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for eos token, 0 for other tokens
        sos_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for sos token, 0 for other tokens
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        gamma: `(float)`
            discounted factor used in RL for token rewards
        high_level_gamma: `(float)`
            discounted factor used in RL for per-turn reward
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        batch_size, gen_len = token_level_rewards.shape
        advantages_turn = torch.zeros_like(token_level_rewards)
        advantages_token = torch.zeros_like(token_level_rewards)
        returns_token = torch.zeros_like(token_level_rewards)
        updated_reward = token_level_rewards.clone()
        
        for b in range(batch_size):
            sos_positions = sos_mask[b].nonzero(as_tuple=True)[0]
            eos_positions = eos_mask[b].nonzero(as_tuple=True)[0]

            # Debug: Check if sos_positions and eos_positions have the same length
            if len(sos_positions) != len(eos_positions):
                print(f"[DEBUG] Batch {b}: sos_positions length ({len(sos_positions)}) != eos_positions length ({len(eos_positions)})")
                print(f"[DEBUG] sos_positions: {sos_positions}")
                print(f"[DEBUG] eos_positions: {eos_positions}")
            # We use sos positions to calculate turn-level advantage
            # We assign the turn-level advantage to the last valid token of each turn
            
            lastgaelam = 0.0
            for i in range(len(sos_positions) - 1, -1, -1):
                cur_sos_pos = sos_positions[i]
                cur_eos_pos = eos_positions[i]
                
                # Get the next value
                if i < len(sos_positions) - 1:
                    # Next valid position
                    
                    next_sos_pos = sos_positions[i + 1]
                    nextvalue = values[b, next_sos_pos]
                    
                else:
                    # Last valid position
                    nextvalue = 0.0
                
                
                reward_for_current_turn = updated_reward[b, cur_eos_pos] # reward is assigned to the last valid token of each turn
                delta = reward_for_current_turn + high_level_gamma * nextvalue - values[b, cur_sos_pos] # value before current tokens are taken
                
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                advantages_turn[b, cur_eos_pos] = lastgaelam

                updated_reward[b, cur_eos_pos] = advantages_turn[b, cur_eos_pos] + values[b, cur_sos_pos]  
                
            
            # Then, calculate low level advantage and return for each token using gamma
            lastgaelam = 0.0
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                if curr_pos not in eos_positions:
                    # Next valid position
                    next_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_pos]
                else:
                    # Last valid position
                    nextvalue = 0.0
                    lastgaelam = 0.0
                delta = updated_reward[b, curr_pos] + gamma * nextvalue - values[b, curr_pos]
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages_token[b, curr_pos] = lastgaelam
                returns_token[b, curr_pos] = lastgaelam + values[b, curr_pos]
        
        advantages_token = verl_F.masked_whiten(advantages_token, loss_mask)
    
    return advantages_token, returns_token


def compute_high_level_gae_advantage_return(
        token_level_rewards: torch.Tensor,
        values: torch.Tensor,
        eos_mask: torch.Tensor,
        sos_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        lam: float,
        high_level_gamma: float
    ):
    
    """
    Compute high-level GAE (Generalized Advantage Estimation) advantage and return for each turn.
    
    This function implements a hierarchical advantage estimation where:
    1. Turn-level advantages are computed using high-level gamma for temporal discounting
    2. Token-level advantages are computed within each turn
    3. The final advantage combines both turn-level and token-level components
    
    Args:
        token_level_rewards (torch.Tensor): Token-level rewards of shape (b, response_len)
        values (torch.Tensor): Value estimates of shape (b, response_len)
        reward_mask (torch.Tensor): Binary mask indicating reward positions of shape (b, response_len)
                                   Each row should have exactly one 1, others are 0
        loss_mask (torch.Tensor): Binary mask indicating valid tokens for loss computation of shape (b, response_len)
        lam (float): GAE lambda parameter for advantage estimation
        high_level_gamma (float): Discount factor for turn-level temporal discounting
    
    Returns:
        tuple: (advantages_token, returns_token) where both are tensors of shape (b, response_len)
               - advantages_token: Computed token-level advantages for each token
               - returns_token: Computed token-level returns for each token
    
    Note:
        - start_mask identifies the beginning of each turn (where loss_mask transitions from 0 to 1)
        - reward_mask identifies the end of each turn (where rewards are assigned)
        - The function processes each batch item separately to handle variable turn lengths
    """
    with torch.no_grad():
        batch_size, gen_len = token_level_rewards.shape
        token_advantages = torch.zeros_like(token_level_rewards)
        token_returns = torch.zeros_like(token_level_rewards)
        
        
        
        for b in range(batch_size):
            sos_positions = sos_mask[b].nonzero(as_tuple=True)[0]
            eos_positions=eos_mask[b].nonzero(as_tuple=True)[0]
            
            # We use sos positions to calculate turn-level advantage
            # We assign the turn-level advantage to the last valid token of each turn
            
            lastgaelam = 0.0
            for i in range(len(sos_positions) - 1, -1, -1):
                cur_sos_pos = sos_positions[i]
                cur_eos_pos = eos_positions[i]
                
                # Get the next value
                if i < len(sos_positions) - 1:
                    # Next valid position
                    
                    next_sos_pos = sos_positions[i + 1]
                    nextvalue = values[b, next_sos_pos]
                    
                else:
                    # Last valid position
                    nextvalue = 0.0
                
                
                
                delta = token_level_rewards[b, cur_eos_pos] + high_level_gamma * nextvalue - values[b, cur_sos_pos] # value before current tokens are taken
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                token_returns[b, cur_sos_pos] = lastgaelam + values[b, cur_sos_pos]
              
                token_advantages[b, cur_sos_pos:cur_eos_pos+1] = lastgaelam # Tokes shares the turn advantage
                
        
        token_advantages = verl_F.masked_whiten(token_advantages, loss_mask)
    
    return token_advantages, token_returns
                    
        
        

def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0 
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t] # TD error
            lastgaelam = delta + gamma * lam * lastgaelam # gae
            advantages_reversed.append(lastgaelam) # store the gae
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores

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
            # Build a map for quick lookup of next turn
            turn_id_to_next = {}
            for i in range(len(turns) - 1):
                curr_turn_id, curr_batch_idx = turns[i]
                next_turn_id, next_batch_idx = turns[i + 1]
                turn_id_to_next[curr_turn_id] = (next_turn_id, next_batch_idx)
            
            # Process turns backward, maintaining lastgaelam across turns
            lastgaelam = 0.0
            
            for i in range(len(turns) - 1, -1, -1):
                turn_id, batch_idx = turns[i]
                
                # Get valid positions for this turn
                valid_positions = loss_mask[batch_idx].nonzero(as_tuple=True)[0]
                if len(valid_positions) == 0:
                    print(f"[DEBUG] No valid positions for turn {turn_id} in batch {batch_idx}")
                    continue
                
                # Check if there's a next turn to connect to
                if turn_id in turn_id_to_next:
                    next_turn_id, next_batch_idx = turn_id_to_next[turn_id]
                    next_valid_positions = loss_mask[next_batch_idx].nonzero(as_tuple=True)[0]
                    if len(next_valid_positions) > 0:
                        # Connect to the first valid position of next turn
                        next_first_pos = next_valid_positions[0].item()
                        next_value_for_continuation = values[next_batch_idx, next_first_pos].item()
                    else:
                        next_value_for_continuation = 0.0
                else:
                    # This is the last turn in trajectory
                    next_value_for_continuation = 0.0
                    lastgaelam = 0.0  # Reset for the last turn
                
                # Backward pass through valid tokens within this turn
                for j in range(len(valid_positions) - 1, -1, -1):
                    curr_pos = valid_positions[j].item()
                    
                    # Get next value
                    if j < len(valid_positions) - 1:
                        # Next token within the same turn
                        next_pos = valid_positions[j + 1].item()
                        nextvalue = values[batch_idx, next_pos].item()
                    else:
                        # Last token of this turn - connect to next turn if exists
                        nextvalue = next_value_for_continuation
                    
                    # Calculate advantage using the updated rewards
                    delta = updated_rewards[batch_idx, curr_pos].item() + gamma * nextvalue - values[batch_idx, curr_pos].item()
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
    Returns are only meaningful at the last token of each turn (marked by reward_mask).
    
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
        lam: `(float)`
            Lambda value for GAE computation.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length). Same advantage value for all tokens in a turn.
        returns: `(torch.Tensor)`
            shape: (bs, response_length). Non-zero only at the last token of each turn.
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
                
                # Get the last valid position for this turn
                valid_positions = loss_mask[batch_idx].nonzero(as_tuple=True)[0]
                if len(valid_positions) == 0:
                    
                    print(f"[DEBUG] No valid positions for turn {turn_id} in batch {batch_idx}")
                    continue
                
                # Use the value at the last valid position
                last_valid_pos = valid_positions[-1].item()
                turn_reward = float(turn_rewards[batch_idx])
                
                # Add token-level rewards (e.g., KL penalties) to turn reward
                # Sum all token rewards for this turn
                token_rewards_sum = token_level_rewards[batch_idx][valid_positions].sum().item()
                total_turn_reward = turn_reward + token_rewards_sum
                
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
            if len(valid_positions) == 0:
                print(f"[DEBUG] No valid positions for turn {turn_id} in batch {batch_idx}")
                continue
            
            # Set the same advantage for all valid tokens in this turn
            for pos in valid_positions:
                advantages[batch_idx, pos] = turn_advantage
            
            # Set return only at the last valid position
            last_valid_pos = valid_positions[-1].item()
            returns[batch_idx, last_valid_pos] = turn_return
        
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


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
