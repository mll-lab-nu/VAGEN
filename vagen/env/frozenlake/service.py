from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from vagen.env.base.base_service import BaseService
from vagen.server.serial import serialize_observation

from .env import FrozenLakeEnv
from .env_config import FrozenLakeEnvConfig
from ..base.base_service_config import BaseServiceConfig
from vagen.env.utils.state_reward_text_utils import service_state_reward_wrapper_v2 as service_state_reward_wrapper
from .prompt import visual_reasoning_reward_prompt
from vagen.env.utils.state_matching import calculate_visual_reasoning_reward_bipartite,calculate_f1_with_max_matching
from vagen.env.utils.top_string_tracker import TopKStringTracker
class FrozenLakeService(BaseService):
    """
    Service class for FrozenLake environments.
    Implements batch operations with parallel processing for efficiency.
    """
    
    def __init__(self, config:BaseServiceConfig):
        """
        Initialize the FrozenLakeService.
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.max_workers = config.get('max_workers', 10)
        self.environments = {}
        self.env_configs = {}
        self.config= config
        if self.config.use_state_reward:
            self.top_strings_tracker_grounding = TopKStringTracker(self.config.top_strings_m)
            self.top_strings_tracker_worldmodeling = TopKStringTracker(self.config.top_strings_m)
        
    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        """
        Create multiple FrozenLake environments in parallel.
        
        Args:
            ids2configs: A dictionary where each key is an environment ID and the corresponding
                        value is the configuration for that environment.
                Each config should contain:
                - env_name: Should be "frozenlake"
                - env_config: FrozenLake specific configuration
        """
        # Define worker function
        def create_single_env(env_id, config):
            # Verify environment type
            env_name = config.get('env_name', 'frozenlake')
            if env_name != 'frozenlake':
                return env_id, None, f"Expected environment type 'frozenlake', got '{env_name}'"
            
            try:
                # Get FrozenLake specific configuration
                env_config_dict = config.get('env_config', {})
                
                # Create environment config
                env_config = FrozenLakeEnvConfig(**env_config_dict)
                
                # Create environment
                env = FrozenLakeEnv(env_config)
                
                return env_id, (env, env_config), None
            except Exception as e:
                return env_id, None, str(e)
        
        # Use ThreadPoolExecutor for parallel creation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all environment creation tasks
            futures = {
                executor.submit(create_single_env, env_id, config): env_id 
                for env_id, config in ids2configs.items()
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error creating environment {env_id}: {error}")
                    continue
                
                env, env_config = result
                self.environments[env_id] = env
                self.env_configs[env_id] = env_config
    
    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        """
        Reset multiple FrozenLake environments in parallel.
        
        Args:
            ids2seeds: A dictionary where each key is an environment ID and the corresponding
                     value is a seed value (or None for using default seeding behavior).
            
        Returns:
            A dictionary mapping environment IDs to tuples of the form (observation, info)
        """
        results = {}
        
        # Define worker function
        def reset_single_env(env_id, seed):
            try:
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                observation, info = env.reset(seed=seed)
                serialized_observation = serialize_observation(observation)
                return env_id, (serialized_observation, info), None
            except Exception as e:
                return env_id, None, str(e)
        
        # Use ThreadPoolExecutor for parallel reset
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all reset tasks
            futures = {
                executor.submit(reset_single_env, env_id, seed): env_id 
                for env_id, seed in ids2seeds.items()
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error resetting environment {env_id}: {error}")
                    results[env_id] = ({}, {"error": error})
                else:
                    results[env_id] = result
        
        return results
    
    @service_state_reward_wrapper
    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        """
        Take a step in multiple FrozenLake environments in parallel.
        
        Args:
            ids2actions: A dictionary where each key is an environment ID and the corresponding
                       value is the action to execute in that environment.
            
        Returns:
            A dictionary mapping environment IDs to tuples of the form (observation, reward, done, info)
        """
        results = {}
        
        # Define worker function
        def step_single_env(env_id, action):
            try:
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                observation, reward, done, info = env.step(action)
                serialized_observation = serialize_observation(observation)
                return env_id, (serialized_observation, reward, done, info), None
            except Exception as e:
                return env_id, None, str(e)
        
        # Use ThreadPoolExecutor for parallel step
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all step tasks
            futures = {
                executor.submit(step_single_env, env_id, action): env_id 
                for env_id, action in ids2actions.items()
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error stepping environment {env_id}: {error}")
                    results[env_id] = ({}, 0.0, True, {"error": error})
                else:
                    results[env_id] = result
        
        return results
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[Any, float]:
        """
        Compute the total reward for multiple FrozenLake environments in parallel.
        
        Args:
            env_ids: A list of environment IDs
            
        Returns:
            A dictionary mapping each environment ID to its computed total reward
        """
        results = {}
        
        # Define worker function
        def compute_reward_single_env(env_id):
            try:
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                return env_id, env.compute_reward(), None
            except Exception as e:
                return env_id, None, str(e)
        
        # Use ThreadPoolExecutor for parallel computation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all computation tasks
            futures = {
                executor.submit(compute_reward_single_env, env_id): env_id 
                for env_id in env_ids
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error computing reward for environment {env_id}: {error}")
                    results[env_id] = 0.0
                else:
                    results[env_id] = result
        
        return results
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[Any, str]:
        """
        Get system prompts for multiple FrozenLake environments in parallel.
        
        Args:
            env_ids: A list of environment IDs
            
        Returns:
            A dictionary mapping each environment ID to its corresponding system prompt string
        """
        results = {}
        
        # Define worker function
        def get_system_prompt_single_env(env_id):
            try:
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                return env_id, env.system_prompt(), None
            except Exception as e:
                return env_id, None, str(e)
        
        # Use ThreadPoolExecutor for parallel retrieval
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all retrieval tasks
            futures = {
                executor.submit(get_system_prompt_single_env, env_id): env_id 
                for env_id in env_ids
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error getting system prompt for environment {env_id}: {error}")
                    results[env_id] = ""
                else:
                    results[env_id] = result
        
        return results
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """
        Close multiple FrozenLake environments and clean up resources in parallel.
        
        Args:
            env_ids: Optional list of environment IDs to close. If None, close all environments.
        """
        # If no env_ids provided, close all environments
        if env_ids is None:
            env_ids = list(self.environments.keys())
        
        # Define worker function
        def close_single_env(env_id):
            try:
                if env_id not in self.environments:
                    return f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                env.close()
                return None
            except Exception as e:
                return str(e)
        
        # Use ThreadPoolExecutor for parallel closing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all closing tasks
            futures = [executor.submit(close_single_env, env_id) for env_id in env_ids]
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                error = future.result()
                if error:
                    print(f"Error closing environment: {error}")
        
        # Remove closed environments from dictionaries
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)
    
    def gen_visual_reasoning_prompt(self, content,**kwargs) -> str:
        return visual_reasoning_reward_prompt.format(prediction=content)
    
    def calculate_visual_reasoning_reward(self, **kwargs) -> float:
        """
        Calculate the visual reasoning reward based on the response and state.
        e.g. [{"object_id": "target", "vertical_relation":above,"horizontal_relation":left}, 
            {"object_id": "hole", "vertical_relation":above,"horizontal_relation":left}]
        Args:
            response: The output of the llm judge (structured state).
            state: The current state of the environment.
            content: The input to the llm judge (natural lanagugae state).
        
        Returns:
            A float representing the calculated reward.
        """
        object_weights={"target": 0.7,"hole": 0.3}
        response = kwargs.get('response', [])
        state = kwargs.get('state', [])
        return calculate_visual_reasoning_reward_bipartite(response, state,object_weights)

        # content = kwargs.get('content', '')
        # r_type = kwargs.get('r_type', 'grounding')
        # if r_type not in ["grounding", "worldmodeling"]:
        #     raise ValueError("r_type must be either 'grounding' or 'worldmodeling'")
        
        # target_result = calculate_f1_with_max_matching(
        #     [item for item in state if item['object_id'] == 'target'] if state else [],
        #     [item for item in response if item['object_id'] == 'target'] if response else [],
        #     match_func=lambda x, y: x['vertical_relation'] == y['vertical_relation'] and x['horizontal_relation'] == y['horizontal_relation']
        # )
        # # check hole reward
        # hole_result =calculate_f1_with_max_matching(
        #     [item for item in state if item['object_id'] == 'hole'] if state else [],
        #     [item for item in response if item['object_id'] == 'hole'] if response else [],
        #     match_func=lambda x, y: x['vertical_relation'] == y['vertical_relation'] and x['horizontal_relation'] == y['horizontal_relation']
        # )
        # target_reward = target_result['f1']
        # hole_reward = hole_result['f1']
        # if r_type=="grounding":
        #     top_k_strings = self.top_strings_tracker_grounding.get_top_k(self.config.top_strings_k)
        # if r_type=="worldmodeling":
        #     top_k_strings = self.top_strings_tracker_worldmodeling.get_top_k(self.config.top_strings_k)
        
        # if content in top_k_strings and target_reward<0.5:
        #     return -0.1
        # return target_reward*0.7 + hole_reward*0.3