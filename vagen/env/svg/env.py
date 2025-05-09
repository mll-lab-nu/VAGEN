from vagen.env.base.base_env import BaseEnv
from vagen.env.svg.svg_utils import (process_and_rasterize_svg, is_valid_svg, load_svg_dataset)
from vagen.env.svg.score import calculate_total_score
from vagen.env.utils.context_utils import parse_llm_raw_response, convert_numpy_to_PIL
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from .env_config import SvgEnvConfig
from .prompt import (
    system_prompt,
    init_observation_template,
    action_template,
    format_prompt
)

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
    """SVG environment for training and evaluating language models to generate SVG code.
    
    This environment provides an image and asks the LLM to generate SVG code that
    reproduces the image as accurately as possible.
    """
    
    def __init__(self, config: SvgEnvConfig,dataset):
        """Initialize the SVG environment.
        
        Args:
            config: Configuration for the environment
        """
        BaseEnv.__init__(self)
        self.config = config

        # Load the actual SVG dataset
        
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
        self.dino_model = None
        self.dataset = dataset
        # Store the format prompt function for later use
        self.prompt_format=self.config.get('prompt_format', 'free_think')
        self.format_prompt_func = format_prompt[self.prompt_format]
        
        # Get the parse function based on the prompt format
        self.parse_func = PARSE_FUNC_MAP[self.prompt_format]
        
        # Initialize random number generator
        self.rng = random.Random()
        if hasattr(self.config, "seed") and self.config.seed is not None:
            self.rng.seed(self.config.seed)
    
    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        """Reset the environment with an optional seed.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Observation dict, info dict
        """
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
        self.valid_actions = []
        
        return self._render(init_obs=True), {}

    def step(self, action_str: str, dino_model=None, dreamsim_model=None) -> Tuple[Dict, float, bool, Dict]:
        """Execute a step in the environment."""
        # Process the LLM response to extract actions
        rst = self.parse_func(
            response=action_str,
            special_token_list=self.config.get('special_token_list', None),
            action_sep=self.config.get('action_sep', ','),
            max_actions=self.config.get('max_actions_per_step', 1)
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
                "action_is_valid": len(rst['actions']) > 0,
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
            
            done = True
            info["metrics"] = metrics
            self.total_reward += self.reward
            self.gen_svg_code = None
            return self._render(init_obs=False), self.reward, done, info
        else:
            # Valid SVG code - apply format reward and process it
            if rst.get("format_correct", True):
                self.reward += self.config.format_reward
                
            self.gen_svg_code = rst['actions'][0]
            self.valid_actions = rst['actions']
            
            try:
                # Process the generated SVG code
                _, gen_image = process_and_rasterize_svg(self.gen_svg_code)
                self.gen_image = gen_image
                
                # Calculate score using service models if provided
                score_config = self.config.get_score_config()
                scores = calculate_total_score(
                    gt_im=self.gt_image,
                    gen_im=gen_image,
                    gt_code=self.gt_svg_code,
                    gen_code=self.gen_svg_code,
                    score_config=score_config,
                    dino_model=dino_model,
                    dreamsim_model=dreamsim_model
                ) 
                
                # Set metrics and update reward
                self.reward += scores["total_score"]
                info["scores"] = scores
                
                # SVG generation is considered effective if score is above threshold
                metrics["turn_metrics"]["action_is_effective"] = scores["total_score"] > 0
                    
            except Exception as e:
                # Reset actions and update metrics
                self.valid_actions = []
                metrics["turn_metrics"]["action_is_valid"] = False
        
        # Update information and total reward
        info["metrics"] = metrics
        self.total_reward += self.reward
        
        return self._render(init_obs=False), self.reward, done, info
    
    def _extract_svg_code(self, text: str) -> str:
        """Extract SVG code from text.
        
        Args:
            text: Text containing SVG code
            
        Returns:
            Extracted SVG code or empty string
        """
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
        """Return the system prompt for the environment.
        
        Returns a prompt explaining the environment to the LLM agent,
        along with the format prompt.
        
        Returns:
            System prompt string
        """
        # Get format prompt with examples for system prompt
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.get('max_actions_per_step', 1),
            action_sep=self.config.get('action_sep', ','),
            add_example=True  # Always true for system prompt
        )
        
        return system_prompt(format=self.prompt_format) + '\n' + format_prompt_text
        
    def close(self):
        """Close the environment and clean up resources."""
        pass
    
    def _render(self, init_obs=False):
        """Render the current state of the environment.
        
        This method creates an observation with the current image and appropriate
        text prompt.
        
        Args:
            init_obs: Whether this is the initial observation
            
        Returns:
            Observation dict
        """
        # Determine which image to show
        if init_obs:
            img = self.gt_image
        elif self.gen_svg_code:
            img = self.gen_image
        else:
            img = Image.new('RGB', (256, 256), color='white')
            
        # Set up multi-modal data with the image
        img_placeholder = self.config.get("image_placeholder", "<image>")
        multi_modal_data = {
            img_placeholder: [img]
        }
        
        # Get format prompt without examples for action/init templates
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.get('max_actions_per_step', 1),
            action_sep=self.config.get('action_sep', ','),
            add_example=False  # No examples for action and init obs
        )
        
        # Prepare observation string based on whether this is initial observation
        if init_obs:
            obs_str = init_observation_template(
                observation=img_placeholder
            ) + "\n" + format_prompt_text
        else:
            valid_action_str = self.valid_actions[0] if self.valid_actions else ""
            obs_str = action_template(
                valid_action=valid_action_str,
                observation=img_placeholder,
                reward=self.reward,
                done=False,  # SVG task doesn't have a "done" state
            ) + "\n" + format_prompt_text
        
        # Return observation with multi-modal data
        return {
            "obs_str": obs_str,
            "multi_modal_data": multi_modal_data,
        }