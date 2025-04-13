from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from vagen.env.base_service import BaseService
from vagen.env.frozenlake.env import FrozenLakeEnv
from vagen.env.frozenlake.config import FrozenLakeConfig
from vagen.utils.serial import serialize_observation

class FrozenLakeService(BaseService):
    """
    Service class for FrozenLake environments.
    Implements batch operations with parallel processing for efficiency.
    """
    
    def __init__(self, max_workers: int = 10):
        """
        Initialize the FrozenLakeService.
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.max_workers = max_workers
        self.environments = {}
        self.env_configs = {}
    
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
                env_config = FrozenLakeConfig(**env_config_dict)
                
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
    
    # FrozenLake specific batch methods - can be kept as additional functionality
    
    def get_maps_batch(self, env_ids: List[str]) -> Dict[Any, Optional[List[List[str]]]]:
        """
        Get maps for multiple FrozenLake environments in parallel.
        
        Args:
            env_ids: A list of environment IDs
            
        Returns:
            A dictionary mapping each environment ID to its map, or None where maps are not available
        """
        results = {}
        
        # Define worker function
        def get_map_single_env(env_id):
            try:
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                if not hasattr(env, "gym_env") or not hasattr(env.gym_env, "desc"):
                    return env_id, None, "Map not available for this environment"
                
                # Convert bytes to strings
                map_data = [[cell.decode('utf-8') for cell in row] for row in env.gym_env.desc]
                return env_id, map_data, None
            except Exception as e:
                return env_id, None, str(e)
        
        # Use ThreadPoolExecutor for parallel retrieval
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all retrieval tasks
            futures = {
                executor.submit(get_map_single_env, env_id): env_id 
                for env_id in env_ids
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error getting map for environment {env_id}: {error}")
                    results[env_id] = None
                else:
                    results[env_id] = result
        
        return results
    
    def get_player_positions_batch(self, env_ids: List[str]) -> Dict[Any, Optional[Tuple[int, int]]]:
        """
        Get player positions for multiple FrozenLake environments in parallel.
        
        Args:
            env_ids: A list of environment IDs
            
        Returns:
            A dictionary mapping each environment ID to its player position, 
            or None where positions are not available
        """
        results = {}
        
        # Define worker function
        def get_position_single_env(env_id):
            try:
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                if not hasattr(env, "_get_player_position"):
                    return env_id, None, "Player position not available for this environment"
                
                position = env._get_player_position()
                # Convert to tuple of ints
                position = tuple(int(x) for x in position)
                return env_id, position, None
            except Exception as e:
                return env_id, None, str(e)
        
        # Use ThreadPoolExecutor for parallel retrieval
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all retrieval tasks
            futures = {
                executor.submit(get_position_single_env, env_id): env_id 
                for env_id in env_ids
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error getting player position for environment {env_id}: {error}")
                    results[env_id] = None
                else:
                    results[env_id] = result
        
        return results
    
    def check_success_batch(self, env_ids: List[str]) -> Dict[Any, Optional[bool]]:
        """
        Check if agents have reached goals in multiple FrozenLake environments in parallel.
        
        Args:
            env_ids: A list of environment IDs
            
        Returns:
            A dictionary mapping each environment ID to its success state, 
            or None where success check is not available
        """
        results = {}
        
        # Define worker function
        def check_success_single_env(env_id):
            try:
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                if not hasattr(env, "_success"):
                    return env_id, None, "Success check not available for this environment"
                
                success = env._success()
                return env_id, success, None
            except Exception as e:
                return env_id, None, str(e)
        
        # Use ThreadPoolExecutor for parallel checks
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all check tasks
            futures = {
                executor.submit(check_success_single_env, env_id): env_id 
                for env_id in env_ids
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error checking success for environment {env_id}: {error}")
                    results[env_id] = None
                else:
                    results[env_id] = result
        
        return results
    
    def is_done_batch(self, env_ids: List[str]) -> Dict[Any, Optional[bool]]:
        """
        Check if episodes are done in multiple FrozenLake environments in parallel.
        
        Args:
            env_ids: A list of environment IDs
            
        Returns:
            A dictionary mapping each environment ID to its done state, 
            or None where done check is not available
        """
        results = {}
        
        # Define worker function
        def is_done_single_env(env_id):
            try:
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                if not hasattr(env, "_finished"):
                    return env_id, None, "Done check not available for this environment"
                
                done = env._finished()
                return env_id, done, None
            except Exception as e:
                return env_id, None, str(e)
        
        # Use ThreadPoolExecutor for parallel checks
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all check tasks
            futures = {
                executor.submit(is_done_single_env, env_id): env_id 
                for env_id in env_ids
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error checking done state for environment {env_id}: {error}")
                    results[env_id] = None
                else:
                    results[env_id] = result
        
        return results