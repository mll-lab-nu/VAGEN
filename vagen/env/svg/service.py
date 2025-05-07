from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from vagen.env.base.base_service import BaseService
from vagen.env.svg.env import SVGEnv
from vagen.env.svg.env_config import SvgEnvConfig
from vagen.server.serial import serialize_observation, serialize_step_result
from vagen.env.svg.score import calculate_total_score, calculate_total_score_batch
from vagen.env.svg.svg_utils import process_and_rasterize_svg, is_valid_svg
from vagen.env.utils.context_utils import parse_llm_raw_response, convert_numpy_to_PIL
from .service_config import SVGServiceConfig
from vagen.env.svg.svg_utils import (process_and_rasterize_svg, is_valid_svg, load_svg_dataset)
import os

class SVGService(BaseService):
    """Service class for SVG environments with centralized model management."""
    
    def __init__(self, config: SVGServiceConfig):
        self.config = config
        self.max_workers = self.config.max_workers
        self.environments = {}
        self.env_configs = {}
        self.cache = {}
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset = {}
        
        # Store device configuration
        self.devices = {
            "dino": "cuda:0",
            "dreamsim": "cuda:0"
        }
        if hasattr(config, "device") and isinstance(config.device, dict):
            for key, value in config.device.items():
                if isinstance(value, (int, float)):
                    self.devices[key] = f"cuda:{int(value)}"
                else:
                    self.devices[key] = value
        
        # Initialize model parameters
        self.model_size = self.config.model_size
        self._models = {}
        
        # Pre-initialize models if configured
        if getattr(self.config, "preload_models", False):
            self._initialize_models()
            
        print(f"SVGService initialized with {self.max_workers} workers, model_size={self.model_size}")
    
    def _initialize_models(self):
        """Initialize models if they haven't been created yet"""
        if "dino" not in self._models:
            from vagen.env.svg.dino import DINOScoreCalculator
            self._models["dino"] = DINOScoreCalculator(
                model_size=self.model_size, 
                device=self.devices["dino"]
            )
            print(f"Initialized DINO model (size={self.model_size}) on {self.devices['dino']}")
        
        if "dreamsim" not in self._models:
            from vagen.env.svg.dreamsim import DreamSimScoreCalculator
            self._models["dreamsim"] = DreamSimScoreCalculator(
                device=self.devices["dreamsim"]
            )
            print(f"Initialized DreamSim model on {self.devices['dreamsim']}")
    
    def get_dino_model(self):
        """Get the DINO model instance, initializing if necessary"""
        if "dino" not in self._models:
            from vagen.env.svg.dino import DINOScoreCalculator
            self._models["dino"] = DINOScoreCalculator(
                model_size=self.model_size, 
                device=self.devices["dino"]
            )
        return self._models["dino"]
    
    def get_dreamsim_model(self):
        """Get the DreamSim model instance, initializing if necessary"""
        if "dreamsim" not in self._models:
            from vagen.env.svg.dreamsim import DreamSimScoreCalculator
            self._models["dreamsim"] = DreamSimScoreCalculator(
                device=self.devices["dreamsim"]
            )
        return self._models["dreamsim"]
    
    def _config_to_env_config(self, config):
        env_config_dict = config.get('env_config', {})
        env_config = SvgEnvConfig(**env_config_dict)
        data_dir = os.path.join(self.script_dir, self.config.get("data_dir", ""))
        dataset_name = env_config.dataset_name
        split = env_config.get("split", "train")
        return {
            "dataset_id":"-".join([str(data_dir), str(dataset_name), str(split)]),
            "config": env_config,
        }
    
    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        id_to_env_config = {}
        for env_id, config in ids2configs.items():
            rst = self._config_to_env_config(config)
            dataset_id = rst["dataset_id"]
            env_config = rst["config"]
            if dataset_id not in self.dataset:
                self.dataset[dataset_id] = load_svg_dataset(
                    data_dir=os.path.join(self.script_dir, env_config.get("data_dir", "")), 
                    dataset_name=env_config.dataset_name,
                    split=env_config.get("split", "train")
                )
            id_to_env_config[env_id] = (env_config, dataset_id)
                
        def create_single_env(env_id, env_config, dataset):
            env = SVGEnv(env_config, dataset)
            return env_id, (env, env_config), None
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(create_single_env, k, v[0], self.dataset[v[1]]): env_id 
                for k, v in id_to_env_config.items()
            }
            
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if not error:
                    env, env_config = result
                    self.environments[env_id] = env
                    self.env_configs[env_id] = env_config
                    self.cache[env_id] = {}
    
    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        results = {}
        
        def reset_single_env(env_id, seed):
            if env_id not in self.environments:
                return env_id, None, f"Environment {env_id} not found"
            
            env = self.environments[env_id]
            observation, info = env.reset(seed=seed)
            
            if env_id in self.cache:
                self.cache[env_id] = {
                    'gt_image': env.gt_image, 
                    'gt_svg_code': env.gt_svg_code,
                    'gen_image': None,
                    'gen_svg_code': None,
                    'scores': None
                }
            
            serialized_observation = serialize_observation(observation)
            return env_id, (serialized_observation, info), None
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(reset_single_env, env_id, seed): env_id 
                for env_id, seed in ids2seeds.items()
            }
            
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
        Optimized step_batch method that maximizes RAM and GPU utilization
        """
        results = {}
        
        # Process SVG actions in batch
        env_processing_results, error_results = self._process_svg_actions_batch(ids2actions)
        results.update(error_results)
        
        # Collect valid SVGs for batch processing
        valid_env_ids = []
        gt_images = []
        gen_images = []
        gt_codes = []
        gen_codes = []
        score_configs = []
        
        for env_id, result in env_processing_results.items():
            if result["valid"] and result["gen_image"] is not None and result["metrics"]["turn_metrics"]["svg_is_valid"]:
                valid_env_ids.append(env_id)
                gt_images.append(result["env"].gt_image)
                gen_images.append(result["gen_image"])
                gt_codes.append(result["env"].gt_svg_code)
                gen_codes.append(result["gen_svg_code"])
                score_configs.append(result["env"].config.get_score_config())
        
        if valid_env_ids:
            # Get models from service
            dino_model = self.get_dino_model()
            dreamsim_model = self.get_dreamsim_model()

            # Calculate all scores at once
            batch_results = calculate_total_score_batch(
                gt_images, gen_images, gt_codes, gen_codes, score_configs,
                dino_model=dino_model, dreamsim_model=dreamsim_model
            )
            
            # Process results and update environments
            for i, env_id in enumerate(valid_env_ids):
                result = env_processing_results[env_id]
                env = result["env"]
                scores = batch_results[i]
                
                # Update reward
                env.reward += scores["total_score"]
                env.total_reward += env.reward
                
                # Determine effectiveness based on improvement
                previous_score = 0.0
                is_first_generation = True
                
                if env_id in self.cache and self.cache[env_id].get('scores') is not None:
                    previous_score = self.cache[env_id]['scores'].get('total_score', 0.0)
                    is_first_generation = False
                
                # Check if first generation or improved
                if is_first_generation:
                    result["metrics"]["turn_metrics"]["action_is_effective"] = scores["total_score"] > 0
                else:
                    result["metrics"]["turn_metrics"]["action_is_effective"] = scores["total_score"] > previous_score
                
                # Update metrics
                result["metrics"]["turn_metrics"]["dino_score"] = scores["dino_score"]
                result["metrics"]["turn_metrics"]["dreamsim_score"] = scores["dreamsim_score"]
                info = result["rst"].copy()
                info["scores"] = scores
                info["metrics"] = result["metrics"]
                
                # Update cache
                if env_id in self.cache:
                    self.cache[env_id]['gen_image'] = env.gen_image
                    self.cache[env_id]['gen_svg_code'] = env.gen_svg_code
                    self.cache[env_id]['scores'] = scores
                
                # Create observation
                observation = env._render(init_obs=False)
                results[env_id] = serialize_step_result((observation, env.reward, False, info))
        
        # Handle invalid cases
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
                    
                # Set invalid metrics
                info["metrics"]["turn_metrics"]["action_is_valid"] = False
                info["metrics"]["turn_metrics"]["action_is_effective"] = False
                
                # Zero scores for invalid SVGs
                info["scores"] = {
                    "dino_score": 0.0,
                    "structural_score": 0.0,
                    "dreamsim_score": 0.0,
                    "total_score": 0.0
                }
                
                # Apply penalty
                reward = 0.0
                if hasattr(env.config, "format_penalty"):
                    reward = env.config.format_penalty
                    
                env.reward = reward
                env.total_reward += reward
                env.gen_svg_code = None
                env.gen_image = None
                
                # Create observation
                observation = env._render(init_obs=False)
                
                # Update cache
                if env_id in self.cache:
                    self.cache[env_id]['gen_image'] = None
                    self.cache[env_id]['gen_svg_code'] = None
                    self.cache[env_id]['scores'] = info["scores"]
                
                results[env_id] = serialize_step_result((observation, reward, False, info))
        
        return results
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[Any, float]:
        results = {}
        
        def compute_reward_single_env(env_id):
            if env_id not in self.environments:
                return env_id, None, f"Environment {env_id} not found"
            
            env = self.environments[env_id]
            return env_id, env.compute_reward(), None
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(compute_reward_single_env, env_id): env_id 
                for env_id in env_ids
            }
            
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    results[env_id] = 0.0
                else:
                    results[env_id] = result
        
        return results
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[Any, str]:
        results = {}
        
        def get_system_prompt_single_env(env_id):
            if env_id not in self.environments:
                return env_id, None, f"Environment {env_id} not found"
            
            env = self.environments[env_id]
            return env_id, env.system_prompt(), None
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(get_system_prompt_single_env, env_id): env_id 
                for env_id in env_ids
            }
            
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    results[env_id] = ""
                else:
                    results[env_id] = result
        
        return results
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        if env_ids is None:
            env_ids = list(self.environments.keys())
        
        def close_single_env(env_id):
            if env_id not in self.environments:
                return f"Environment {env_id} not found"
            
            env = self.environments[env_id]
            env.close()
            return None
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(close_single_env, env_id) for env_id in env_ids]
            
            for future in as_completed(futures):
                future.result()
        
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)
            self.cache.pop(env_id, None)
    
    def _process_svg_actions_batch(self, ids2actions):
        env_processing_results = {}
        error_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            def process_action(env_id, action):
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                rst = parse_llm_raw_response(
                    response=action,
                    special_token_list=env.config.get('special_token_list', None),
                    action_sep=env.config.get("action_sep", ","),
                    max_actions=env.config.get("max_actions_per_step", 1)
                )
                
                svg_code = None
                svg_is_valid = False
                
                if not rst['actions']:
                    svg_code = env._extract_svg_code(action)
                    if svg_code:
                        svg_is_valid = is_valid_svg(svg_code)
                        rst['actions'] = [svg_code]
                else:
                    svg_code = env._extract_svg_code(rst['actions'][0])
                    if svg_code:
                        svg_is_valid = is_valid_svg(svg_code)
                        rst['actions'] = [svg_code]
                    else:
                        rst['actions'] = []
                
                metrics = {
                    "turn_metrics": {
                        "action_is_valid": rst['actions'] != [],
                        "svg_is_valid": svg_is_valid,
                        "action_is_effective": False,
                        "dino_score": 0.0,
                        "dreamsim_score": 0.0,
                    },
                    "traj_metrics": {
                        "success": False,
                    }
                }
                
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
                
                env.reward = env.config.format_reward if svg_is_valid else env.config.format_penalty
                env.total_reward += env.reward
                env.gen_svg_code = rst['actions'][0]
                env.valid_actions = rst['actions']
                
                try:
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
                except Exception as e:
                    env.gen_image = None
                    env.valid_actions = []
                    
                    return env_id, {
                        "env": env,
                        "gen_image": None,
                        "gen_svg_code": env.gen_svg_code,
                        "rst": rst,
                        "metrics": metrics,
                        "valid": False,
                        "done": False
                    }, None
            
            futures = {
                executor.submit(process_action, env_id, action): env_id 
                for env_id, action in ids2actions.items()
            }
            
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    error_results[env_id] = ({}, 0.0, False, {"error": error})
                else:
                    env_processing_results[env_id] = result
        
        return env_processing_results, error_results