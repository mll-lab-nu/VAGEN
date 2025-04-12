from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from vagen.env.base_service import BaseService
from vagen.env.svg.env import SVGEnv
from vagen.env.svg.config import SVGConfig
from vagen.utils.serial import serialize_observation, serialize_step_result
from vagen.env.svg.score import calculate_total_score
from vagen.env.svg.dino import get_dino_model
from vagen.env.svg.svg_utils import process_and_rasterize_svg
import logging

class SVGService(BaseService):
    """
    Service class for SVG environments.
    Implements batch operations with parallel processing for efficiency.
    Integrates DINO scoring model directly within the service.
    """
    
    def __init__(self, max_workers: int = 10, model_size: str = "small"):
        """
        Initialize the SVGService.
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
            model_size: Size of the DINO model to use ("small", "base", or "large")
        """
        self.max_workers = max_workers
        self.environments = {}
        self.env_configs = {}
        self.cache = {}
        
        # Load the DINO model directly in the service
        # This allows all environments to share the same model instance
        self.model_size = model_size
        self.dino_model = None  # Will be loaded on first use
        
        # Store device for model inference
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"SVGService initialized with {max_workers} workers, model_size={model_size}, device={self.device}")
    
    def _get_dino_model(self):
        """
        Get or initialize the DINO model.
        Uses lazy loading to avoid loading the model until needed.
        """
        if self.dino_model is None:
            logging.info(f"Loading DINO model (size={self.model_size}, device={self.device})")
            self.dino_model = get_dino_model(self.model_size, self.device)
        return self.dino_model
    
    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        """
        Create multiple SVG environments in parallel.
        
        Args:
            ids2configs: A dictionary where each key is an environment ID and the corresponding
                        value is the configuration for that environment.
                Each config should contain:
                - env_name: Should be "SVG"
                - env_config: SVG specific configuration
        """
        # Define worker function
        def create_single_env(env_id, config):
            # Verify environment type
            env_name = config.get('env_name', 'svg')
            if env_name != 'svg':
                return env_id, None, f"Expected environment type 'SVG', got '{env_name}'"
            
            try:
                # Get SVG specific configuration
                env_config_dict = config.get('env_config', {})
                
                # Create environment config
                env_config = SVGConfig(**env_config_dict)
                                
                # Create environment
                env = SVGEnv(env_config)
                
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
                    logging.error(f"Error creating environment {env_id}: {error}")
                    continue
                
                env, env_config = result
                self.environments[env_id] = env
                self.env_configs[env_id] = env_config
                # Initialize cache for this environment
                self.cache[env_id] = {}
                logging.info(f"Environment {env_id} created successfully")
    
    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        """
        Reset multiple SVG environments in parallel.
        
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
                
                # Cache current state for this environment
                if env_id in self.cache:
                    self.cache[env_id] = {
                        'gt_image': env.gt_image, 
                        'gt_svg_code': env.gt_svg_code,
                        'gen_image': None,
                        'gen_svg_code': None,
                        'scores': None
                    }
                
                # Serialize the observation for return
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
                    logging.error(f"Error resetting environment {env_id}: {error}")
                    results[env_id] = ({}, {"error": error})
                else:
                    results[env_id] = result
        
        return results
    
    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        """
        Take a step in multiple SVG environments in parallel.
        Computes SVG scores directly using the integrated DINO model.
        
        Args:
            ids2actions: A dictionary where each key is an environment ID and the corresponding
                       value is the SVG action to execute in that environment.
            
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
                
                dino_model = self._get_dino_model()

                observation, reward, done, info = env.step(action, dino_model=dino_model)
                
                if env_id in self.cache:
                    self.cache[env_id]['gen_image'] = getattr(env, 'gen_image', None)
                    self.cache[env_id]['gen_svg_code'] = getattr(env, 'gen_svg_code', None)
                    self.cache[env_id]['scores'] = info.get('scores', None)
                
                serialized_result = serialize_step_result((observation, reward, done, info))
                return env_id, serialized_result, None
            
            except Exception as e:
                logging.error(f"Error in step_single_env for {env_id}: {str(e)}")
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
                    logging.error(f"Error stepping environment {env_id}: {error}")
                    results[env_id] = ({}, 0.0, True, {"error": error})
                else:
                    results[env_id] = result
        
        return results
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[Any, float]:
        """
        Compute the total reward for multiple SVG environments in parallel.
        
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
                    logging.error(f"Error computing reward for environment {env_id}: {error}")
                    results[env_id] = 0.0
                else:
                    results[env_id] = result
        
        return results
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[Any, str]:
        """
        Get system prompts for multiple SVG environments in parallel.
        
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
                    logging.error(f"Error getting system prompt for environment {env_id}: {error}")
                    results[env_id] = ""
                else:
                    results[env_id] = result
        
        return results
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """
        Close multiple SVG environments and clean up resources in parallel.
        
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
                    logging.error(f"Error closing environment: {error}")
        
        # Remove closed environments from dictionaries
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)
            self.cache.pop(env_id, None)