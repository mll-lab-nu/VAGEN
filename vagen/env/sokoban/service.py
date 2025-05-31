from typing import Dict, List, Tuple, Optional, Any, Union
from vagen.env.base.base_service import BaseService
from vagen.env.base.base_service_config import BaseServiceConfig
from vagen.env.utils.state_reward_text_utils import service_state_reward_wrapper_new as service_state_reward_wrapper
from vagen.server.serial import serialize_observation

from .env import SokobanEnv
from .env_config import SokobanEnvConfig
from .prompt import visual_reasoning_reward_prompt
from vagen.env.utils.state_matching import calculate_visual_reasoning_reward_bipartite
class SokobanService(BaseService):
    
    def __init__(self, config: BaseServiceConfig):
        self.environments = {}
        self.env_configs = {}
        self.config = config
    
    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        for env_id, config in ids2configs.items():
            env_config_dict = config.get('env_config', {})
            env_config = SokobanEnvConfig(**env_config_dict)
            env = SokobanEnv(env_config)
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
    
    def gen_visual_reasoning_prompt(self, content,**kwargs) -> str:
        return visual_reasoning_reward_prompt.format(prediction=content)
    
    def calculate_visual_reasoning_reward(self, response,state) -> float:
        """
        Calculate the visual reasoning reward based on the response and state.
        
        Args:
            response: The response from the LLM.
            state: The current state of the environment.
        
        Returns:
            A float representing the calculated reward.
        """
        object_weights={"target": 0.5,"box": 0.5}
        
        return calculate_visual_reasoning_reward_bipartite(response, state,object_weights)
        