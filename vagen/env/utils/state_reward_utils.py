STATE_REWARD_MAP={
    "grounding": {
        "content_keys": ["observation"],
        "state_keys": ["observation_state"],
    },
    "worldmodeling": {
        "content_keys": ["prediction_content"],
        "state_keys": ["prediction_state"],
    },
    "grounding_worldmodeling": {
        "content_keys": ["observation", "prediction_content"],
        "state_keys": ["observation_state", "prediction_state"],
    },
    "grounding_structured":{
        "content_keys": ["observation_content"],
        "state_keys": ["observation_state"],
    },
    "worldmodeling_structured": {
        "content_keys": ["prediction_content"],
        "state_keys": ["prediction_state"],
    },
    "grounding_worldmodeling_structured": {
        "content_keys": ["observation_content", "prediction_content"],
        "state_keys": ["observation_state", "prediction_state"],
    },
    "grounding_symbolic": {
        "content_keys": ["observation_content"],
        "state_keys": ["observation_state"],
    },
    "worldmodeling_symbolic": {
        "content_keys": ["prediction_content"],
        "state_keys": ["prediction_state"],
    },
    "grounding_worldmodeling_symbolic": {
        "content_keys": ["observation_content", "prediction_content"],
        "state_keys": ["observation_state", "prediction_state"],
    },
}

def state_reward_wrapper(step_func):
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
        if hasattr(self, 'config') and getattr(self.config, 'use_state_reward', False):
            pre_state = self.get_env_state()
            obs, reward, done, info = step_func(self, action_str)
            post_state = self.get_env_state()
            
            # Get the mapping based on prompt format
            prompt_format = getattr(self.config, 'prompt_format', None)
            mapping = STATE_REWARD_MAP(prompt_format)
            content_keys = mapping["content_keys"]
            state_keys = mapping["state_keys"]
            
            # Update info with content and state pairs based on the mapping
            for content_key,state_key in zip(content_keys,state_keys):
                info[content_key] = info["rst"][content_key] 
                if "observation" in state_key:
                    info[state_key] = pre_state
                elif "prediction" in state_key:
                    info[state_key] = post_state
                else:
                    raise ValueError(f"Unknown state key: {state_key}")
            
            info["use_state_reward"] = True
            return obs, reward, done, info
        else:
            return step_func(self, action_str)
    return wrapped_step