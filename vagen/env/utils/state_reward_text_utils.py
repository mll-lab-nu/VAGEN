# Process Reward for Grounding and World Modeling
#
# This file helps to give process rewards for environments that support "grounding", 
# "worldmodeling", or "grounding_worldmodeling" response formats.
#
# Process rewards are calculated in two ways:
# 1. Observation process reward: Evaluates how well the agent grounds its observations
# 2. Prediction process reward: Evaluates the accuracy of the agent's world model predictions
#
# Requirements:
# - The environment must implement get_env_state() method that returns a text-based 
#   description of the current environment state
# - This state description is used as ground truth for calculating process rewards
from typing import List, Dict, Any
import asyncio
import time
from .llm_judge import run_llm_judge

def env_state_reward_wrapper(step_func):
    """
    Decorator function that enhances the step method to include state rewards.
    
    This wrapper:
    1. Captures the state before and after executing the original step function
    2. Updates the info dictionary with appropriate content and state keys based on prompt format
    3. Handles accumulated rewards if configured
    
    Args:
        step_func: The original step function to be wrapped
        
    Returns:
        The wrapped step function with enhanced state reward functionality
    """
    def wrapped_step(self, action_str):
        if hasattr(self, 'config') and self.config.get("use_state_reward", False):
            pre_state = self.get_env_state()
            obs, reward, done, info = step_func(self, action_str)
            post_state = self.get_env_state()
            
            # Get the mapping based on prompt format
            prompt_format = self.config.get("prompt_format", None)
            if prompt_format is None:
                raise ValueError("Prompt format is not specified in the config.")
            
            if "observation_content" in info and info["observation_content"]:
                info["observation_state"] = pre_state
            if "prediction_content" in info and info["prediction_content"]:
                info["prediction_state"] = post_state
            info["use_state_reward"] = True
            return obs, reward, done, info
        else:
            return step_func(self, action_str)
    return wrapped_step

def service_state_reward_wrapper(step_batch_func):
    """
    Decorator to wrap the step_batch function to calculate and apply rewards.
    
    Args:
        step_batch_func: Original step_batch function
        
    Returns:
        Wrapped step_batch function with reward calculation
    """
    def wrapped_step_batch(self, ids2actions):
        # Call the original step_batch function
        step_batch_results = step_batch_func(self, ids2actions)
        input_to_llm = []
        for id, result in step_batch_results.items():
            obs, reward, done, info = result
            if info.get("use_state_reward", False):
                if info.get("observation_content", None) and info.get("observation_state", None):
                    input_to_llm.append({
                        "id": id,
                        "content": info["observation_content"],
                        "state": info["observation_state"],
                        "type": "observation"
                    })
                if info.get("prediction_content", None) and info.get("prediction_state", None):
                    input_to_llm.append({
                        "id": id,
                        "content": info["prediction_content"],
                        "state": info["prediction_state"],
                        "type": "prediction"
                    })
                    
        if len(input_to_llm) > 0:
            # Use synchronous batch processing
            scores = run_llm_judge(input_to_llm)
        else:
            return step_batch_results
        
        new_step_batch_results = {id: list(result) for id, result in step_batch_results.items()}
        
        for item, score in zip(input_to_llm, scores):
            id = item["id"]
            env_config = self.env_configs[id]
            if "metrics" not in new_step_batch_results[id][3]:
                new_step_batch_results[id][3]["metrics"] = {"turn_metrics": {}, "traj_metrics": {}}
            if "turn_metrics" not in new_step_batch_results[id][3]["metrics"]:
                new_step_batch_results[id][3]["metrics"]["turn_metrics"] = {}
            if item["type"] == "observation":
                new_step_batch_results[id][3]["metrics"]["turn_metrics"]["grounding_reward"] = score * env_config.get("grounding_reward_weight", 0.5)
                new_step_batch_results[id][1] += score * env_config.get("grounding_reward_weight", 0.5)
            elif item["type"] == "prediction":
                new_step_batch_results[id][3]["metrics"]["turn_metrics"]["worldmodeling_reward"] = score * env_config.get("worldmodeling_reward_weight", 0.5)
                new_step_batch_results[id][1] += score * env_config.get("worldmodeling_reward_weight", 0.5)
        
        return {id: tuple(result) for id, result in new_step_batch_results.items()}
                
    return wrapped_step_batch