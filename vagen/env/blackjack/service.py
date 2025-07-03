from typing import Dict, List, Tuple, Optional, Any, Union
from vagen.env.base.base_service import BaseService
from vagen.env.base.base_service_config import BaseServiceConfig
from vagen.server.serial import serialize_observation

from .env import BlackjackEnv
from .env_config import BlackjackEnvConfig
from vagen.env.utils.state_reward_text_utils import service_state_reward_wrapper_v3 as service_state_reward_wrapper
from vagen.env.utils.top_string_tracker import TopKStringTracker

class BlackjackService(BaseService):
    
    def __init__(self, config: BaseServiceConfig):
        self.environments = {}
        self.env_configs = {}
        self.config = config
        
        # Initialize state reward components if enabled
        if self.config.use_state_reward:
            self.top_strings_tracker_decision = TopKStringTracker(self.config.top_strings_m)
            self.top_strings_tracker_reasoning = TopKStringTracker(self.config.top_strings_m)
    
    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        for env_id, config in ids2configs.items():
            env_config_dict = config.get('env_config', {})
            env_config = BlackjackEnvConfig(**env_config_dict)
            env = BlackjackEnv(env_config)
            self.environments[env_id] = env
            self.env_configs[env_id] = env_config
    
    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        results = {}
        
        for env_id, seed in ids2seeds.items():
            env = self.environments[env_id]
            observation, info = env.reset(seed=seed)
            serialized_observation = serialize_observation(observation)
            results[env_id] = (serialized_observation, info)
        
        return results
    
    @service_state_reward_wrapper
    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        results = {}
        
        for env_id, action in ids2actions.items():
            env = self.environments[env_id]
            observation, reward, done, info = env.step(action)
            serialized_observation = serialize_observation(observation)
            results[env_id] = (serialized_observation, reward, done, info)
        
        return results
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[Any, float]:
        results = {}
        
        for env_id in env_ids:
            env = self.environments[env_id]
            results[env_id] = env.compute_reward()
        
        return results
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[Any, str]:
        results = {}
        
        for env_id in env_ids:
            env = self.environments[env_id]
            results[env_id] = env.system_prompt()
        
        return results
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        if env_ids is None:
            env_ids = list(self.environments.keys())
        
        for env_id in env_ids:
            env = self.environments[env_id]
            env.close()
            
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)
    
    def gen_decision_reasoning_prompt(self, content, **kwargs) -> str:
        """
        Generate prompt for evaluating decision-making reasoning in Blackjack.
        This method is called by the state reward wrapper when use_state_reward is enabled.
        """
        return f"""Evaluate the quality of this Blackjack decision and reasoning:

Decision/Reasoning: {content}

Please assess:
1. Is the decision strategically sound?
2. Does the reasoning demonstrate understanding of Blackjack probability?
3. Are key factors (dealer card, hand total, bust risk) considered?

Respond with JSON: {{"decision_quality": 0.0-1.0, "reasoning_quality": 0.0-1.0}}"""
    
    def calculate_decision_reasoning_reward(self, **kwargs) -> float:
        """
        Calculate reward for Blackjack decision making.
        
        Args:
            response: The LLM judge response (should contain quality score)
            content: The original content being judged
            r_type: Type of reasoning being evaluated
        
        Returns:
            A float representing the calculated reward (0.0 to 1.0)
        """
        response = kwargs.get("response")
        content = kwargs.get("content")
        r_type = kwargs.get("r_type", "decision")
        
        # Parse the LLM judge response to get quality score
        try:
            if isinstance(response, dict):
                # Assume the judge returns a quality score
                reward = response.get("quality", 0.0)
            else:
                # Simple fallback
                reward = 0.5
        except:
            reward = 0.0
        
        # Check for repetitive responses (anti-cheating mechanism)
        if r_type == "decision":
            top_k_strings = self.top_strings_tracker_decision.get_top_k(self.config.top_strings_k)
        else:
            top_k_strings = self.top_strings_tracker_reasoning.get_top_k(self.config.top_strings_k)
        
        # Penalize repetitive low-quality responses
        if content in top_k_strings and reward < 0.6:
            return -0.1
            
        return reward   