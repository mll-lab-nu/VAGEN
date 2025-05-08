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

