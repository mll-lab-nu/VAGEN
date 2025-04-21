from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from vagen.env.base.base_service import BaseService
from vagen.env.svg.env import SVGEnv
from vagen.env.svg.env_config import SvgEnvConfig
from vagen.server.serial import serialize_observation, serialize_step_result
from vagen.env.svg.score import calculate_total_score, calculate_total_score_batch
from vagen.env.svg.dino import get_dino_model
from vagen.env.svg.svg_utils import process_and_rasterize_svg, is_valid_svg
from vagen.env.utils.context_utils import parse_llm_raw_response, convert_numpy_to_PIL
from .service_config import SVGServiceConfig

class SVGService(BaseService):
    """
    Service class for SVG environments.
    Implements batch operations with parallel processing for efficiency.
    Integrates DINO scoring model directly within the service.
    """
    
    def __init__(self, config: SVGServiceConfig):
        """
        Initialize the SVGService.
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
            model_size: Size of the DINO model to use ("small", "base", or "large")
        """
        self.config= config
        self.max_workers = self.config.max_workers
        self.environments = {}
        self.env_configs = {}
        self.cache = {}
        
        # Load the DINO model directly in the service
        # This allows all environments to share the same model instance
        self.model_size = self.config.model_size
        self.dino_model = None  # Will be loaded on first use

        # Add DreamSim model support
        self.dreamsim_model = None  # Will be loaded on first use if enabled
        
        # Store device for model inference
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"SVGService initialized with {self.max_workers} workers, model_size={self.model_size}, device={self.device}")
    
    def _get_dino_model(self):
        """
        Get or initialize the DINO model.
        Uses lazy loading to avoid loading the model until needed.
        """
        if self.dino_model is None:
            self.dino_model = get_dino_model(self.model_size, self.device)
        return self.dino_model

    def _get_dreamsim_model(self):
        """
        Get or initialize the DreamSim model.
        Uses lazy loading to avoid loading the model until needed.
        """
        if self.dreamsim_model is None:
            from vagen.env.svg.dreamsim import get_dreamsim_model
            self.dreamsim_model = get_dreamsim_model(self.device)
        return self.dreamsim_model
    
    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        """
        Create multiple SVG environments in parallel.
        
        Args:
            ids2configs: A dictionary where each key is an environment ID and the corresponding
                        value is the configuration for that environment.
        """
        def create_single_env(env_id, config):
            env_name = config.get('env_name', 'svg')
            if env_name != 'svg':
                return env_id, None, f"Expected environment type 'SVG', got '{env_name}'"
            
            # Get SVG specific configuration
            env_config_dict = config.get('env_config', {})
            
            # Create environment config
            env_config = SvgEnvConfig(**env_config_dict)
                            
            # Create environment
            env = SVGEnv(env_config)
            
            return env_id, (env, env_config), None
        
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
                if not error:
                    env, env_config = result
                    self.environments[env_id] = env
                    self.env_configs[env_id] = env_config
                    # Initialize cache for this environment
                    self.cache[env_id] = {}
    
    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        """
        Reset multiple SVG environments in parallel.
        
        Args:
            ids2seeds: A dictionary mapping environment IDs to seed values.
            
        Returns:
            A dictionary mapping environment IDs to tuples of the form (observation, info)
        """
        results = {}
        
        def reset_single_env(env_id, seed):
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
                    results[env_id] = ({}, {"error": error})
                else:
                    results[env_id] = result
        
        return results
    
    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        """
        Take a step in multiple SVG environments with optimized batch processing.
        
        Args:
            ids2actions: A dictionary mapping environment IDs to actions
            
        Returns:
            A dictionary mapping environment IDs to (observation, reward, done, info) tuples
        """
        results = {}
        # Step 1: Process SVG actions for all environments
        env_processing_results, error_results = self._process_svg_actions_batch(ids2actions)
        results.update(error_results)
        
        # Step 2: Collect valid images for batch processing
        valid_env_ids = []
        gt_images = []
        gen_images = []
        gt_codes = []
        gen_codes = []
        score_configs = []
        
        for env_id, result in env_processing_results.items():
            if result["valid"] and result["gen_image"] is not None:
                valid_env_ids.append(env_id)
                gt_images.append(result["env"].gt_image)
                gen_images.append(result["gen_image"])
                gt_codes.append(result["env"].gt_svg_code)
                gen_codes.append(result["gen_svg_code"])
                score_configs.append(result["env"].config.get_score_config())
        
        # Step 3: Batch process scores if there are valid images
        if valid_env_ids:
            # Get DINO model
            dino_model = self._get_dino_model()
            dreamsim_model = self._get_dreamsim_model()

            # Calculate all scores at once
            batch_results = calculate_total_score_batch(
                gt_images, gen_images, gt_codes, gen_codes, score_configs, dino_model, dreamsim_model
            )
            
            # Process results directly using the index mapping
            for i, env_id in enumerate(valid_env_ids):
                result = env_processing_results[env_id]
                env = result["env"]
                scores = batch_results[i]
                
                # Update reward
                env.reward += scores["total_score"]
                env.total_reward += env.reward
                
                # Update metrics and prepare info
                result["metrics"]["turn_metrics"]["action_is_effective"] = scores["total_score"] > 0
                info = result["rst"].copy()
                info["scores"] = scores
                info["metrics"] = result["metrics"]
                
                # Update cache if needed
                if env_id in self.cache:
                    self.cache[env_id]['gen_image'] = env.gen_image
                    self.cache[env_id]['gen_svg_code'] = env.gen_svg_code
                    self.cache[env_id]['scores'] = scores
                
                # Create final result
                observation = env._render(init_obs=False)
                results[env_id] = serialize_step_result((observation, env.reward, False, info))
        
        # Step 4: Process invalid or failed generations
        for env_id, result in env_processing_results.items():
            if env_id not in results:
                env = result["env"]
                
                info = result["rst"].copy() if "rst" in result else {}
                
                if "metrics" not in info:
                    info["metrics"] = {"turn_metrics": {}, "traj_metrics": {}}
                elif "turn_metrics" not in info["metrics"]:
                    info["metrics"]["turn_metrics"] = {}
                elif "traj_metrics" not in info["metrics"]:
                    info["metrics"]["traj_metrics"] = {}
                    
                info["metrics"]["turn_metrics"]["action_is_valid"] = False
                info["metrics"]["turn_metrics"]["action_is_effective"] = False
                
                if "scores" not in info:
                    info["scores"] = {
                        "dino_score": 0.0,
                        "structural_score": 0.0,
                        "total_score": 0.0
                    }
                
                reward = 0.0
                
                if hasattr(env.config, "format_penalty"):
                    reward = env.config.format_penalty
                    
                env.reward = reward
                env.total_reward += reward
                env.gen_svg_code = None
                env.gen_image = None
                
                observation = env._render(init_obs=False)
                
                if env_id in self.cache:
                    self.cache[env_id]['gen_image'] = None
                    self.cache[env_id]['gen_svg_code'] = None
                    self.cache[env_id]['scores'] = info["scores"]
                
                results[env_id] = serialize_step_result((observation, reward, False, info))
        
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
        
        def compute_reward_single_env(env_id):
            if env_id not in self.environments:
                return env_id, None, f"Environment {env_id} not found"
            
            env = self.environments[env_id]
            return env_id, env.compute_reward(), None
        
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
        
        def get_system_prompt_single_env(env_id):
            if env_id not in self.environments:
                return env_id, None, f"Environment {env_id} not found"
            
            env = self.environments[env_id]
            return env_id, env.system_prompt(), None
        
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
        
        def close_single_env(env_id):
            if env_id not in self.environments:
                return f"Environment {env_id} not found"
            
            env = self.environments[env_id]
            env.close()
            return None
        
        # Use ThreadPoolExecutor for parallel closing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all closing tasks
            futures = [executor.submit(close_single_env, env_id) for env_id in env_ids]
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                future.result()
        
        # Remove closed environments from dictionaries
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)
            self.cache.pop(env_id, None)
    
    def _process_svg_actions_batch(self, ids2actions):
        """
        Process SVG actions for all environments in parallel using ThreadPoolExecutor.
        
        Args:
            ids2actions (Dict): A dictionary mapping environment IDs to actions
        
        Returns:
            Dict: A dictionary of processing results keyed by environment ID
            Dict: A dictionary of error results directly ready for return keyed by environment ID
        """
        env_processing_results = {}
        error_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            def process_action(env_id, action):
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                # Parse and extract SVG code from action
                rst = parse_llm_raw_response(
                    response=action,
                    special_token_list=env.config.get('special_token_list', None),
                    action_sep=env.config.get("action_sep", ","),
                    max_actions=env.config.get("max_actions_per_step", 1)
                )
                
                # Handle SVG code extraction
                svg_code = None
                svg_is_valid = False
                
                # First, try to extract SVG code from the response
                if not rst['actions']:
                    svg_code = env._extract_svg_code(action)
                    if svg_code:
                        svg_is_valid = is_valid_svg(svg_code)
                        # Even if SVG is invalid, still keep it for training purposes
                        rst['actions'] = [svg_code]
                else:
                    svg_code = env._extract_svg_code(rst['actions'][0])
                    if svg_code:
                        svg_is_valid = is_valid_svg(svg_code)
                        # Always keep extracted SVG code regardless of validity
                        rst['actions'] = [svg_code]
                    else:
                        rst['actions'] = []
                
                # Initialize metrics - track validity separately from action presence
                metrics = {
                    "turn_metrics": {
                        "action_is_valid": rst['actions'] != [],  # Action exists
                        "svg_is_valid": svg_is_valid,  # SVG syntax is valid
                        "action_is_effective": False,
                    },
                    "traj_metrics": {
                        "success": False,
                    }
                }
                
                # Handle case where no SVG code could be extracted
                if not rst['actions']:
                    env.reward = env.config.format_penalty
                    env.total_reward += env.reward
                    env.gen_svg_code = None
                    env.valid_actions = []
                    info = rst.copy()
                    info["metrics"] = metrics
                    return env_id, {
                        "env": env,
                        "gen_image": None,
                        "gen_svg_code": None,
                        "rst": rst,
                        "metrics": metrics,
                        "info": info,
                        "valid": False,
                        "done": False
                    }, None
                
                # Process SVG (valid or invalid)
                env.reward = env.config.format_reward if svg_is_valid else env.config.format_penalty
                env.total_reward += env.reward
                env.gen_svg_code = rst['actions'][0]
                env.valid_actions = rst['actions']

                _, env.gen_image = process_and_rasterize_svg(env.gen_svg_code)
                
                return env_id, {
                    "env": env,
                    "gen_image": env.gen_image,
                    "gen_svg_code": env.gen_svg_code,
                    "rst": rst,
                    "metrics": metrics,
                    "valid": True,
                    "done": False
                }, None
            
            # Submit all action processing tasks
            futures = {
                executor.submit(process_action, env_id, action): env_id 
                for env_id, action in ids2actions.items()
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    error_results[env_id] = ({}, 0.0, False, {"error": error})
                else:
                    env_processing_results[env_id] = result
        
        return env_processing_results, error_results