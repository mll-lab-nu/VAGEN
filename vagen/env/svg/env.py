from vagen.env.base_env import BaseEnv
from vagen.env.svg.svg_utils import (process_and_rasterize_svg, is_valid_svg, load_svg_dataset)
from vagen.env.svg.score import calculate_total_score
from vagen.env.utils.context_utils import parse_llm_raw_response, convert_numpy_to_PIL
from .config import SVGConfig
from .prompt import system_prompt, init_observation_template, action_template

import os
import re
import json
import logging
import random
from PIL import Image
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datasets import Dataset

class SVGEnv(BaseEnv):
    def __init__(self, config: SVGConfig):
        BaseEnv.__init__(self)
        self.config = config
        
        # Load the actual SVG dataset
        self.dataset = load_svg_dataset(data_dir = self.config.get("data_dir", ""), 
                                        dataset_name = self.config.dataset_name,
                                        split = self.config.get("split", "train"))
        
        # Initialize state variables
        self.total_reward = 0
        self.reward = 0
        self.valid_actions = []
        self.current_sample = None
        self.img_id = None
        self.gt_svg_code = None
        self.gt_image = None
        self.gen_svg_code = None
        self.gen_image = None
        
        # Initialize random number generator
        self.rng = random.Random()
        if hasattr(self.config, "seed") and self.config.seed is not None:
            self.rng.seed(self.config.seed)
        
        # Set up analysis logging if enabled
        if self.config.analysis_mode:
            self._setup_analysis_logging()
    

    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        """Reset the environment with an optional seed"""
        # Update seed if provided
        if seed is not None:
            self.rng.seed(seed)
            
        # Deterministically select a sample from the dataset
        dataset_length = len(self.dataset)
        index = self.rng.randint(0, dataset_length - 1)
        self.current_sample = self.dataset[index]
        
        # Extract SVG code and filename
        # Field names may vary depending on the actual dataset structure
        self.gt_svg_code = self.current_sample.get('Svg', self.current_sample.get('svg', ''))
        self.img_id = self.current_sample.get('Filename', self.current_sample.get('filename', f'image_{index}'))
        
        if not self.gt_svg_code:
            raise ValueError(f"Ground truth SVG code not found in sample at index {index}")
            
        # Process ground truth SVG to get the image
        _, self.gt_image = process_and_rasterize_svg(self.gt_svg_code)
        
        # Reset tracking variables
        self.total_reward = 0
        self.reward = 0
        self.gen_svg_code = ""
        self.gen_image = None
        
        return self._render(init_obs=True), {}

    def step(self, action_str: str) -> Tuple[Dict, float, bool, Dict]:
        """Execute a step in the environment"""
        # Parse LLM response
        rst = parse_llm_raw_response(
            response=action_str,
            special_token_list=self.config.get('special_token_list', None),
            action_sep=self.config.get("action_sep", ","),
            max_actions=self.config.get("max_actions_per_step", 1)
        )
        
        # Extract SVG code if not found in parsed response
        if not rst['actions']:
            svg_code = self._extract_svg_code(action_str)
            if svg_code and is_valid_svg(svg_code):
                rst['actions'] = [svg_code]
        else:
            # Check if the extracted action is a valid SVG
            svg_code = self._extract_svg_code(rst['actions'][0])
            if svg_code and is_valid_svg(svg_code):
                rst['actions'] = [svg_code]
            else:
                rst['actions'] = []
        
        metrics = {
            "turn_metrics": {
                "action_is_valid": rst['actions'] != [],
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            }
        }
        
        self.reward = 0
        self.valid_actions = []
        done = False
        info = {}
        info.update(rst)
        
        if not rst['actions']:
            # Invalid format - apply penalty
            self.reward += self.config.format_penalty
            
            # Log failure if analysis mode is enabled
            if hasattr(self, 'failure_logger'):
                failure_info = {
                    'img_id': self.img_id,
                    'gt_svg_code': self.gt_svg_code,
                    'gen_svg_code': action_str,
                    'failure_reason': 'invalid_svg'
                }
                self.failure_logger.info(json.dumps(failure_info))
                
                done = True
                info["metrics"] = metrics
                self.total_reward += self.reward
                self.gen_svg_code = None
                return self._render(init_obs=False), self.reward, done, info
        else:
            # Valid SVG code - apply format reward and process it
            self.reward += self.config.format_reward
            self.gen_svg_code = rst['actions'][0]
            self.valid_actions = rst['actions']
            
            try:
                # Process the generated SVG code
                _, gen_image = process_and_rasterize_svg(self.gen_svg_code)
                self.gen_image = gen_image
                
                # Calculate score
                score_config = {
                    "model_size": self.config.model_size,
                    "dino_only": self.config.dino_only,
                }
                
                # Add optional weights if set
                if self.config.dino_weight is not None:
                    score_config["dino_weight"] = self.config.dino_weight
                if self.config.structural_weight is not None:
                    score_config["structural_weight"] = self.config.structural_weight
                if self.config.color_weight is not None:
                    score_config["color_weight"] = self.config.color_weight
                if self.config.code_weight is not None:
                    score_config["code_weight"] = self.config.code_weight
                
                try:
                    scores = calculate_total_score(
                        gt_im=self.gt_image,
                        gen_im=gen_image,
                        gt_code=self.gt_svg_code,
                        gen_code=self.gen_svg_code,
                        score_config=score_config
                    )                
                except Exception as e:
                    print(f"Score calculation failed: {e}")
                
                # Set metrics and update reward
                self.reward += scores["total_score"]
                info["scores"] = scores
                
                # SVG generation is considered effective if score is above threshold
                metrics["turn_metrics"]["action_is_effective"] = scores["total_score"] > 0
                
                # Log success if analysis mode is enabled
                if hasattr(self, 'success_logger'):
                    success_info = {
                        'img_id': self.img_id,
                        'gt_svg_code': self.gt_svg_code,
                        'gen_svg_code': self.gen_svg_code,
                        'scores': scores
                    }
                    self.success_logger.info(json.dumps(success_info))
                    
            except Exception as e:
                # Error processing SVG - log failure
                if hasattr(self, 'failure_logger'):
                    failure_info = {
                        'img_id': self.img_id,
                        'gt_svg_code': self.gt_svg_code,
                        'gen_svg_code': self.gen_svg_code,
                        'failure_reason': str(e)
                    }
                    self.failure_logger.info(json.dumps(failure_info))
                
                # Reset actions and update metrics
                self.valid_actions = []
                metrics["turn_metrics"]["action_is_valid"] = False
        
        # Update information and total reward
        info["metrics"] = metrics
        self.total_reward += self.reward
        
        return self._render(init_obs=False), self.reward, done, info
    
    def _extract_svg_code(self, text: str) -> str:
        """Extract SVG code from text"""
        svg_match = re.search(r'<svg.*?</svg>', text, re.DOTALL)
        if svg_match:
            return svg_match.group(0)

        if '<svg' in text and '</svg>' in text:
            start_idx = text.find('<svg')
            end_idx = text.rfind('</svg>') + 6  # 6 is the length of '</svg>'
            if start_idx < end_idx:
                return text[start_idx:end_idx]

        return ""
        
    def system_prompt(self) -> str:
        """Return the system prompt"""
        return system_prompt
        
    def compute_reward(self) -> float:
        """Return the total reward collected so far"""
        return self.total_reward
        
    def close(self):
        """Close the environment and clean up resources"""
        if hasattr(self, 'failure_logger'):
            for handler in self.failure_logger.handlers:
                handler.close()
                self.failure_logger.removeHandler(handler)
                
        if hasattr(self, 'success_logger'):
            for handler in self.success_logger.handlers:
                handler.close()
                self.success_logger.removeHandler(handler)
    
    def _render(self, init_obs=False):
        """Render the current state of the environment"""
        multi_modal_data = None
        
        # Determine which image to show
        if init_obs:
            img = self.gt_image
        elif self.gen_svg_code:
            img = self.gen_image
        else:
            img = Image.new('RGB', (256, 256), color='white')
            
        # Set up multi-modal data with the image
        img_placeholder = self.config.get("image_placeholder", "image")
        multi_modal_data = {
            img_placeholder: [img]
        }
        img_str = img_placeholder
        
        # Prepare observation string based on whether this is initial observation
        if init_obs:
            obs_str = init_observation_template.format(observation=img_str)
        else:
            obs_str = action_template.format(
                valid_action=self.valid_actions,
                observation=img_str,
                reward=self.reward,
                done=False,  # SVG task doesn't have a "done" state
            )
        
        # Return observation with multi-modal data
        return {
            "obs_str": obs_str,
            "multi_modal_data": multi_modal_data,
        }
    
    def _setup_analysis_logging(self):
        """Set up logging for analysis mode"""
        log_dir = Path(self.config.data_dir) / 'analysis_logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Failure logger
        self.failure_logger = logging.getLogger(f'svg_failure_{id(self)}')
        self.failure_logger.setLevel(logging.INFO)
        
        if not self.failure_logger.handlers:
            failure_handler = logging.FileHandler(log_dir / 'failure_cases.log')
            failure_handler.setFormatter(logging.Formatter('%(message)s'))
            self.failure_logger.addHandler(failure_handler)
        
        # Success logger
        self.success_logger = logging.getLogger(f'svg_success_{id(self)}')
        self.success_logger.setLevel(logging.INFO)
        
        if not self.success_logger.handlers:
            success_handler = logging.FileHandler(log_dir / 'success_cases.log')
            success_handler.setFormatter(logging.Formatter('%(message)s'))
            self.success_logger.addHandler(success_handler)

if __name__ == "__main__":
    config = SVGConfig(
        dataset_name="starvector/svg-emoji-simple",
        data_dir="vagen/env/svg/data",
        split="test",
        model_size="small"
    )
    
    try:
        env = SVGEnv(config)
        print(f"Successfully loaded dataset")
        
        # Test with seed
        seed = 42
        obs, info = env.reset(seed=seed)
        print(f"Testing with seed {seed}")
        
        # Example SVG action
        action = """<think>
        The image appears to be a simple emoji face with two eyes and a smile.
        I'll create an SVG with:
        1. A circle for the face
        2. Two circles for the eyes
        3. A path for the smile
        </think>
        <answer>
        <svg width="100" height="100" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="40" fill="yellow"/>
          <circle cx="35" cy="40" r="5" fill="black"/>
          <circle cx="65" cy="40" r="5" fill="black"/>
          <path d="M30 60 Q50 75 70 60" stroke="black" stroke-width="3" fill="none"/>
        </svg>
        </answer>"""
        
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"obs:{obs}")
        print(f"Score components: {info.get('scores', {})}")
        
        # Test with another seed to verify determinism
        seed = 123
        obs, info = env.reset(seed=seed)
        print(f"\nTesting with seed {seed}")
        
        env.close()
    except Exception as e:
        print(f"Error: {e}")