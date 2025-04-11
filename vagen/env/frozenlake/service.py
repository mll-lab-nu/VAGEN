from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from vagen.env.base_service import BaseService
from vagen.env.frozenlake.env import FrozenLakeEnv
from vagen.env.frozenlake.config import FrozenLakeConfig

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
        super().__init__(max_workers=max_workers)
    
    def create_environments_batch(self, configs: List[Dict[str, Any]]) -> List[str]:
        """
        Create multiple FrozenLake environments in parallel.
        
        Args:
            configs: List of environment configurations
                Each config should contain:
                - env_name: Should be "frozenlake"
                - env_config: FrozenLake specific configuration
                
        Returns:
            List of environment IDs
        """
        env_ids = []
        created_envs = {}
        
        # Define worker function
        def create_single_env(config_index, config):
            # Verify environment type
            env_name = config.get('env_name', 'frozenlake')
            if env_name != 'frozenlake':
                return config_index, None, f"Expected environment type 'frozenlake', got '{env_name}'"
            
            try:
                # Get FrozenLake specific configuration
                env_config_dict = config.get('env_config', {})
                
                # Create environment config
                env_config = FrozenLakeConfig(**env_config_dict)
                
                # Create environment
                env = FrozenLakeEnv(env_config)
                
                # Generate environment ID
                env_id = str(uuid.uuid4())
                
                return config_index, (env_id, env, env_config), None
            except Exception as e:
                return config_index, None, str(e)
        
        # Use ThreadPoolExecutor for parallel creation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all environment creation tasks
            futures = {
                executor.submit(create_single_env, i, config): i 
                for i, config in enumerate(configs)
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                idx, result, error = future.result()
                if error:
                    print(f"Error creating environment from config {idx}: {error}")
                    continue
                
                env_id, env, env_config = result
                created_envs[idx] = env_id
                self.environments[env_id] = env
                self.env_configs[env_id] = env_config
                env_ids.append(env_id)
        
        # Make sure env_ids are in the same order as input configs
        ordered_env_ids = []
        for i in range(len(configs)):
            if i in created_envs:
                ordered_env_ids.append(created_envs[i])
        
        return ordered_env_ids
    
    def reset_batch(self, env_ids: List[str], seeds: Optional[List[Optional[int]]] = None) -> List[Tuple[Dict, Dict]]:
        """
        Reset multiple FrozenLake environments in parallel.
        
        Args:
            env_ids: List of environment IDs to reset
            seeds: Optional list of seeds for resetting environments
            
        Returns:
            List of (observation, info) tuples
        """
        results = [None] * len(env_ids)
        
        # Prepare seeds list
        if seeds is None:
            seeds = [None] * len(env_ids)
        elif len(seeds) < len(env_ids):
            seeds = seeds + [None] * (len(env_ids) - len(seeds))
        
        # Define worker function
        def reset_single_env(index, env_id, seed):
            try:
                if env_id not in self.environments:
                    return index, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                return index, env.reset(seed=seed), None
            except Exception as e:
                return index, None, str(e)
        
        # Use ThreadPoolExecutor for parallel reset
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all reset tasks
            futures = {
                executor.submit(reset_single_env, i, env_id, seeds[i]): i 
                for i, env_id in enumerate(env_ids)
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                idx, result, error = future.result()
                if error:
                    print(f"Error resetting environment {env_ids[idx]}: {error}")
                    results[idx] = ({}, {"error": error})
                else:
                    results[idx] = result
        
        return results
    
    def step_batch(self, env_ids: List[str], actions: List[str]) -> List[Tuple[Dict, float, bool, Dict]]:
        """
        Take a step in multiple FrozenLake environments in parallel.
        
        Args:
            env_ids: List of environment IDs
            actions: List of actions to take in each environment
            
        Returns:
            List of (observation, reward, done, info) tuples
        """
        if len(env_ids) != len(actions):
            raise ValueError("Number of environment IDs must match number of actions")
            
        results = [None] * len(env_ids)
        
        # Define worker function
        def step_single_env(index, env_id, action):
            try:
                if env_id not in self.environments:
                    return index, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                return index, env.step(action), None
            except Exception as e:
                return index, None, str(e)
        
        # Use ThreadPoolExecutor for parallel step
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all step tasks
            futures = {
                executor.submit(step_single_env, i, env_id, actions[i]): i 
                for i, env_id in enumerate(env_ids)
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                idx, result, error = future.result()
                if error:
                    print(f"Error stepping environment {env_ids[idx]}: {error}")
                    results[idx] = ({}, 0.0, True, {"error": error})
                else:
                    results[idx] = result
        
        return results
    
    def compute_reward_batch(self, env_ids: List[str]) -> List[float]:
        """
        Compute the total reward for multiple FrozenLake environments in parallel.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            List of reward values
        """
        results = [0.0] * len(env_ids)
        
        # Define worker function
        def compute_reward_single_env(index, env_id):
            try:
                if env_id not in self.environments:
                    return index, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                return index, env.compute_reward(), None
            except Exception as e:
                return index, None, str(e)
        
        # Use ThreadPoolExecutor for parallel computation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all computation tasks
            futures = {
                executor.submit(compute_reward_single_env, i, env_id): i 
                for i, env_id in enumerate(env_ids)
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                idx, result, error = future.result()
                if error:
                    print(f"Error computing reward for environment {env_ids[idx]}: {error}")
                    results[idx] = 0.0
                else:
                    results[idx] = result
        
        return results
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> List[str]:
        """
        Get system prompts for multiple FrozenLake environments in parallel.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            List of system prompt strings
        """
        results = [""] * len(env_ids)
        
        # Define worker function
        def get_system_prompt_single_env(index, env_id):
            try:
                if env_id not in self.environments:
                    return index, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                return index, env.system_prompt(), None
            except Exception as e:
                return index, None, str(e)
        
        # Use ThreadPoolExecutor for parallel retrieval
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all retrieval tasks
            futures = {
                executor.submit(get_system_prompt_single_env, i, env_id): i 
                for i, env_id in enumerate(env_ids)
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                idx, result, error = future.result()
                if error:
                    print(f"Error getting system prompt for environment {env_ids[idx]}: {error}")
                    results[idx] = ""
                else:
                    results[idx] = result
        
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
    
    # FrozenLake specific batch methods
    
    def get_maps_batch(self, env_ids: List[str]) -> List[Optional[List[List[str]]]]:
        """
        Get maps for multiple FrozenLake environments in parallel.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            List of maps, or None for environments where maps are not available
        """
        results = [None] * len(env_ids)
        
        # Define worker function
        def get_map_single_env(index, env_id):
            try:
                if env_id not in self.environments:
                    return index, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                if not hasattr(env, "gym_env") or not hasattr(env.gym_env, "desc"):
                    return index, None, "Map not available for this environment"
                
                # Convert bytes to strings
                map_data = [[cell.decode('utf-8') for cell in row] for row in env.gym_env.desc]
                return index, map_data, None
            except Exception as e:
                return index, None, str(e)
        
        # Use ThreadPoolExecutor for parallel retrieval
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all retrieval tasks
            futures = {
                executor.submit(get_map_single_env, i, env_id): i 
                for i, env_id in enumerate(env_ids)
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                idx, result, error = future.result()
                if error:
                    print(f"Error getting map for environment {env_ids[idx]}: {error}")
                    results[idx] = None
                else:
                    results[idx] = result
        
        return results
    
    def get_player_positions_batch(self, env_ids: List[str]) -> List[Optional[Tuple[int, int]]]:
        """
        Get player positions for multiple FrozenLake environments in parallel.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            List of player positions, or None for environments where positions are not available
        """
        results = [None] * len(env_ids)
        
        # Define worker function
        def get_position_single_env(index, env_id):
            try:
                if env_id not in self.environments:
                    return index, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                if not hasattr(env, "_get_player_position"):
                    return index, None, "Player position not available for this environment"
                
                position = env._get_player_position()
                # Convert to tuple of ints
                position = tuple(int(x) for x in position)
                return index, position, None
            except Exception as e:
                return index, None, str(e)
        
        # Use ThreadPoolExecutor for parallel retrieval
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all retrieval tasks
            futures = {
                executor.submit(get_position_single_env, i, env_id): i 
                for i, env_id in enumerate(env_ids)
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                idx, result, error = future.result()
                if error:
                    print(f"Error getting player position for environment {env_ids[idx]}: {error}")
                    results[idx] = None
                else:
                    results[idx] = result
        
        return results
    
    def check_success_batch(self, env_ids: List[str]) -> List[Optional[bool]]:
        """
        Check if agents have reached goals in multiple FrozenLake environments in parallel.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            List of success states, or None for environments where success check is not available
        """
        results = [None] * len(env_ids)
        
        # Define worker function
        def check_success_single_env(index, env_id):
            try:
                if env_id not in self.environments:
                    return index, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                if not hasattr(env, "_success"):
                    return index, None, "Success check not available for this environment"
                
                success = env._success()
                return index, success, None
            except Exception as e:
                return index, None, str(e)
        
        # Use ThreadPoolExecutor for parallel checks
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all check tasks
            futures = {
                executor.submit(check_success_single_env, i, env_id): i 
                for i, env_id in enumerate(env_ids)
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                idx, result, error = future.result()
                if error:
                    print(f"Error checking success for environment {env_ids[idx]}: {error}")
                    results[idx] = None
                else:
                    results[idx] = result
        
        return results
    
    def is_done_batch(self, env_ids: List[str]) -> List[Optional[bool]]:
        """
        Check if episodes are done in multiple FrozenLake environments in parallel.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            List of done states, or None for environments where done check is not available
        """
        results = [None] * len(env_ids)
        
        # Define worker function
        def is_done_single_env(index, env_id):
            try:
                if env_id not in self.environments:
                    return index, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                if not hasattr(env, "_finished"):
                    return index, None, "Done check not available for this environment"
                
                done = env._finished()
                return index, done, None
            except Exception as e:
                return index, None, str(e)
        
        # Use ThreadPoolExecutor for parallel checks
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all check tasks
            futures = {
                executor.submit(is_done_single_env, i, env_id): i 
                for i, env_id in enumerate(env_ids)
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                idx, result, error = future.result()
                if error:
                    print(f"Error checking done state for environment {env_ids[idx]}: {error}")
                    results[idx] = None
                else:
                    results[idx] = result
        
        return results