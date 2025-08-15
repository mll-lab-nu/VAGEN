from vagen.env.base.base_env import BaseEnv
from vagen.env.alfworld.alfworld_utils import load_alfworld_dataset
from vagen.env.utils.context_utils import parse_llm_raw_response, convert_numpy_to_PIL
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from .env_config import AlfEnvConfig
from .prompt import (
    system_prompt,
    init_observation_template,
    action_template,
    format_prompt
)

import os
import json
import logging
import random
import alfworld.agents.environment as environment
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

class ALFWorldEnv(BaseEnv):
    """ALFRED environment for training and evaluating language models as agents.
    
    This environment implements a text-based world where an agent must complete
    household tasks. It is designed specifically for Large Language Models (LLMs)
    as agents, providing structured observations and handling text-based actions.
    """
    
    def __init__(self, config: AlfEnvConfig):
        """Initialize the ALFRED environment.
        
        Args:
            config: Configuration for the environment
        """
        super().__init__()
        self.config = config
        
        # Setup environment
        import yaml
        with open(self.config.alf_config_path) as reader:
            alf_config = yaml.safe_load(reader)
        
        if self.config.render_mode == "vision":
            alf_config['env']['type'] = 'AlfredThorEnv'
            env = alfworld.agents.environment.AlfredThorEnv(alf_config)
        else:
            alf_config['env']['type'] = 'AlfredTWEnv'
            env = alfworld.agents.environment.AlfredTWEnv(alf_config)
        
        self.env = env.init_env(batch_size=1)
        
        # Initialize state variables
        self.total_reward = 0
        self.reward = 0
        self.valid_actions = []
        self.current_task = None
        self.current_observation = None
        self.available_actions = []
        self.is_done = False
        self.task_completed = False
        self.step_count = 0
        
        # Store the format prompt function for later use
        self.format_prompt_func = format_prompt[self.config.get('prompt_format', 'free_think')]
        
        # Get the parse function based on the prompt format
        self.parse_func = PARSE_FUNC_MAP[self.config.get('prompt_format', 'free_think')]
        
        # Initialize the dataset
        self.dataset = self._load_dataset()
        self.num_games = len(self.dataset) if self.dataset else 0
        
        # Initialize random number generator
        self.rng = random.Random()
        if hasattr(self.config, "seed") and self.config.seed is not None:
            self.rng.seed(self.config.seed)
    
    def _load_dataset(self):
        """Load the ALFRED dataset."""
        if hasattr(self.config, "split_id"):
            return self.env.json_file_list[self.config.split_id]
        return self.env.json_file_list
    
    def reset(self, seed=None):
        """Reset the environment with an optional seed.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Observation dict, info dict
        """
        # Update seed if provided
        if seed is not None:
            self.rng.seed(seed)
            
        # Determine game index
        if hasattr(self.config, "gamefiles") and self.config.gamefiles:
            self.env.game_files = self.config.gamefiles
            self.env.random_start = False
        else:
            game_index = self.rng.randint(0, self.num_games - 1) if seed is None else seed % self.num_games
            self.env.game_files = [self.dataset[game_index]]
            self.env.random_start = False
            
        # Reset the environment
        self.current_observation, game_info = self.env.reset()
        self.current_task = game_info["task_desc"] if "task_desc" in game_info else None
        self.available_actions = self.env.admissible_commands
        
        # Reset state variables
        self.total_reward = 0
        self.reward = 0
        self.valid_actions = []
        self.is_done = False
        self.task_completed = False
        self.step_count = 0
        
        return self._render(init_obs=True), {}
    
    def step(self, action_str: str):
        """Execute an action in the environment.
        
        This method:
        1. Parses the raw LLM response to extract actions
        2. Executes each valid action in sequence
        3. Calculates rewards and metrics
        4. Generates the next observation
        
        Args:
            action_str: Raw text response from LLM
            
        Returns:
            Observation, reward, done, info
        """
        # Process the LLM response to extract actions
        rst = self.parse_func(
            response=action_str,
            special_token_list=self.config.get('special_token_list', None),
            action_sep=self.config.get('action_sep', ','),
            max_actions=self.config.get('max_actions_per_step', 1)
        )
        
        action_list = rst['actions']
        
        metrics = {
            "turn_metrics": {
                "action_is_valid": len(action_list) > 0,
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            }
        }
        
        self.reward = 0
        self.valid_actions = []
        done = self.is_done
        info = {}
        info.update(rst)
        
        # Execute valid actions
        if metrics["turn_metrics"]["action_is_valid"]:
            # Add format reward if actions were valid and format is correct
            if rst.get("format_correct", True):
                self.reward += self.config.format_reward
            
            for action in action_list:
                # Check if action is admissible
                if action in self.available_actions:
                    # Execute the action
                    observation, reward, done, _ = self.env.step(action)
                    self.current_observation = observation
                    self.available_actions = self.env.admissible_commands
                    self.is_done = done
                    
                    # Update state and metrics
                    self.reward += reward
                    self.valid_actions.append(action)
                    metrics["turn_metrics"]["action_is_effective"] = True
                    
                    if reward > 1.0:  # Task completion reward
                        metrics["traj_metrics"]["success"] = True
                        self.task_completed = True
                else:
                    # Invalid action penalty
                    self.reward += self.config.invalid_action_penalty
                    metrics["turn_metrics"]["action_is_valid"] = False
                    break
                
                self.step_count += 1
                if self.step_count >= self.config.max_steps or done:
                    break
        
        # Update info dict and total reward
        info["metrics"] = metrics
        info["task"] = self.current_task
        info["step_count"] = self.step_count
        info["task_completed"] = self.task_completed
        
        self.total_reward += self.reward
        
        return self._render(init_obs=False), self.reward, done, info
    
    def system_prompt(self):
        """Get the system prompt for the environment.
        
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
        
        if self.config.get("use_vision", False):
            return self.config.vision_system_prompt.format(
                max_actions_per_step=self.config.max_actions_per_step,
                action_sep=self.config.action_sep
            ) + '\n' + format_prompt_text
        else:
            return system_prompt() + '\n' + format_prompt_text
    
    def compute_reward(self):
        """Return the total reward for the episode.
        
        Returns:
            Total reward
        """
        return self.total_reward
    
    def close(self):
        """Close the environment."""
        if hasattr(self, "env"):
            self.env.close()
    
    def _render(self, init_obs=False):
        """Render the environment observation.
        
        This method creates a text representation of the environment state,
        formatting the observation string based on whether this is the
        initial observation or a subsequent one.
        
        Args:
            init_obs: Whether this is the initial observation
            
        Returns:
            Observation dict
        """
        # Get format prompt without examples for action/init templates
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.get('max_actions_per_step', 1),
            action_sep=self.config.get('action_sep', ','),
            add_example=False  # No examples for action and init obs
        )
        
        # Format the commands as a string
        commands_str = ", ".join(self.available_actions)
        
        # Get vision observation if enabled
        multi_modal_data = None
        if self.config.get("use_vision", False) and hasattr(self.env, "get_frame"):
            frame = self.env.get_frame()
            if frame is not None:
                img_placeholder = self.config.get("image_placeholder", "<image>")
                multi_modal_data = {
                    img_placeholder: [convert_numpy_to_PIL(frame)]
                }
                # Use the image placeholder in the observation if frame is available
                observation = img_placeholder
            else:
                observation = self.current_observation
        else:
            observation = self.current_observation
        
        # Format the template
        if init_obs:
            obs_str = init_observation_template(
                observation=observation,
                commands=commands_str,
                instruction=self.current_task
            ) + "\n" + format_prompt_text
        else:
            valid_action_str = self.valid_actions[-1] if self.valid_actions else "No action"
            obs_str = action_template(
                valid_action=valid_action_str,
                observation=observation,
                commands=commands_str,
                reward=self.total_reward,
                done=self.is_done,
                instruction=self.current_task
            ) + "\n" + format_prompt_text
        
        return {
            "obs_str": obs_str,
            "multi_modal_data": multi_modal_data
        }