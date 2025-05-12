from vagen.env.base.base_env import BaseEnv
import numpy as np
import copy
from typing import Dict, List, Optional, Tuple, Any
from gymnasium.utils import seeding
from vagen.env.utils.context_utils import convert_numpy_to_PIL
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from .env_config import PrimitiveSkillEnvConfig
from .maniskill.utils import build_env, handle_info, get_workspace_limits
from .prompt import system_prompt, init_observation_template, action_template, format_prompt
import vagen.env.primitive_skill.maniskill.env
import random
from vagen.env.utils.state_reward_text_utils import env_state_reward_wrapper
class PrimitiveSkillEnv(BaseEnv):
    def __init__(self, config: PrimitiveSkillEnvConfig):
        """
        Initialize the PrimitiveSkill environment.
        
        Args:
            config (PrimitiveSkillEnvConfig): Configuration parameters for the environment
        """
        BaseEnv.__init__(self)
        self.config = config
        if self.config.record_video:
            record_dir = self.config.video_record_dir
        else:
            record_dir = None
        self.env = build_env(config.env_id, record_dir=record_dir)
        
        # Store the format prompt function for later use based on the configuration
        self.format_prompt_func = format_prompt[self.config.prompt_format]
        self.parse_func = PARSE_FUNC_MAP[self.config.prompt_format]
        # Define the state keys for the environment
        self.state_keys = self.env.state_keys
        self.last_info = None
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed (Optional[int]): Random seed for environment generation
                                  If None, a random seed is used
        
        Returns:
            Tuple[Dict, Dict]: 
                - obs: Dictionary containing observation string and optional image data
                - info: Empty dictionary for initial state
        """
        _, info = self.env.reset(seed=seed)
        self.last_info = info
        obs = self._render(init_obs=True)
        self.initial_reward = self._compute_reward()
        self.total_reward = 0
        self.steps = 0
        return obs, {}
    
    @env_state_reward_wrapper
    def step(self, action_str):
        """
        Take a step in the environment based on the agent's action.
        
        Args:
            action_str (str): Raw string from LLM containing actions
        
        Returns:
            Tuple[Dict, float, bool, Dict]:
                - obs: Dictionary with observation string and optional image data
                - reward: Numeric reward for the step
                - done: Boolean indicating if episode is complete
                - info: Dictionary containing metrics and parsed action data
        """
        reward = 0
        rst = self.parse_func(response=action_str,
            special_token_list=self.config.special_token_list,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step)
        
        output_info = {}
        output_info.update(rst)
        valid_actions = []
        metrics = {
            "turn_metrics": {
                "action_is_valid": False,  # True if at least one valid action was parsed
            },
            "traj_metrics": {
                "success": False,  # Will be set to True if agent reaches goal
            },
        }
        
        info = self.last_info
        terminated, truncated = False, False
        
        
        # Execute each action in the list
        for action in rst['actions']:
            parsed_action = self._parse_action(action)
            if parsed_action is not None:
                _, _, terminated, truncated, info = self.env.step(parsed_action)
                valid_actions.append(action)
                self.last_info = info
                self.steps += 1
            else:
                info = self.last_info
                terminated, truncated = False, False
                break
            if truncated or terminated:
                break
        
        # Check if actions were valid and format was correct
        metrics["turn_metrics"]['action_is_valid'] = len(valid_actions) > 0 and len(valid_actions) == len(rst['actions'])
        if metrics["turn_metrics"]['action_is_valid'] and rst["format_correct"]:
            reward += self.config.format_reward
            output_info["is_format_rewarded"] = True
        else:
            output_info["is_format_rewarded"] = False
        
        if info.get('is_success', False):
            metrics["traj_metrics"]['success'] = True
        
        done = terminated or truncated
        
        obs = self._render(init_obs=False, valid_actions=valid_actions)
        output_info["metrics"] = metrics
        
        self.total_reward += reward
        if isinstance(done, np.ndarray):
            done = done.item()
            
        return obs, reward, done, output_info
    
    def system_prompt(self):
        """
        Get the system prompt for the environment.
        
        Returns:
            str: System prompt string with environment description and instructions
        """
        # Get format prompt with examples for system prompt
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            state_keys=self.state_keys,
            add_example=True  # Always true for system prompt
        )
        
        return system_prompt() + '\n' + format_prompt_text
    
    def close(self):
        """
        Close the environment and clean up resources.
        """
        self.env.close()
    
    def _compute_reward(self):
        """
        Calculate the reward based on environment state.
        
        Returns:
            float: Computed reward value
        """
        if self.last_info.get("success", False):
            return 10
        
        # Find the highest successful stage
        max_stage = -1
        for key in self.last_info.keys():
            if key.startswith("stage_") and key.endswith("_success"):
                try:
                    # Extract the stage number
                    stage_num = int(key.split("_")[1])
                    # Check if this stage is successful
                    if self.last_info[key]:
                        max_stage = max(max_stage, stage_num)
                except (ValueError, IndexError):
                    # Skip keys that don't follow the expected format
                    continue
        return (max_stage + 1) * 2
    
    def compute_reward(self):
        """
        Get the cumulative reward for the episode.
        
        Returns:
            float: Total reward accumulated during the current episode
        """
        return self._compute_reward() - self.initial_reward - self.steps * 0.1

    
    def _render(self, init_obs=True, valid_actions=None,seed=42):
        """
        Render the environment as an observation.
        
        Args:
            init_obs (bool): If True, create initial observation
            valid_actions (list): List of valid actions executed (for step observations)
        
        Returns:
            Dict: Observation dictionary containing observation string and optional image data
        """
        info = self.last_info.copy()
        new_info = handle_info(info, state_keys=self.state_keys,mask_success=self.config.mask_success, env=self.env)
        positions_list = list(new_info['obj_positions'].values())

        object_positions = str(positions_list)
        # object_names=str([key.removesuffix("_position") for key in new_info['obj_positions'].keys()])
        other_information = str(new_info['other_info'])
        instruction = self.env.instruction()
        img_placeholder = self.config.image_placeholder
        x_workspace, y_workspace, z_workspace = get_workspace_limits(self.env)
        
        # Get format prompt without examples for action/init templates
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            state_keys=self.state_keys,
            add_example=False  # No examples for action and init obs
        )
        
        if init_obs:
            # Initial observation
            obs_str = init_observation_template(
                observation=img_placeholder,
                instruction=instruction,
                x_workspace=x_workspace,
                y_workspace=y_workspace,
                z_workspace=z_workspace,
                object_positions=object_positions,
                other_information=other_information,
                #object_names=object_names
            ) + "\n" + format_prompt_text
        else:
            # Subsequent observations include action results
            obs_str = action_template(
                valid_actions=valid_actions,
                observation=img_placeholder,
                instruction=instruction,
                x_workspace=x_workspace,
                y_workspace=y_workspace,
                z_workspace=z_workspace,
                object_positions=object_positions,
                other_information=other_information,
                #object_names=object_names
            ) + "\n" + format_prompt_text
        
        multi_modal_data = None
        if self.config.render_mode == "vision":
            img = self.env.render()
            multi_modal_data = {
                img_placeholder: [convert_numpy_to_PIL(img)]
            }
        
        # Return observation dictionary with appropriate fields
        if multi_modal_data is not None:
            return {
                "obs_str": obs_str,
                "multi_modal_data": multi_modal_data,
            }
        else:
            return {
                "obs_str": obs_str,
            }
    
    
    def get_env_state(self):
        """
        Get the current state of the environment.
        
        Returns:
            dict: Dictionary representation of the environment state
        """
        rst=handle_info(self.last_info, state_keys=self.state_keys,mask_success=self.config.mask_success, env=self.env)
        return rst["obj_positions"]
    
    
    def _parse_action(self, action_str):
        """
        Parse a single action string into an action array.
        
        Args:
            action_str (str): Action string to parse
            
        Returns:
            np.array: Parsed action array or None if invalid
        """
        # Initialize empty 9-dim array (3 for action type, 6 for coordinates)
        action_array = np.zeros(9)
        
        # Workspace boundaries
        workspace_x, workspace_y, workspace_z = get_workspace_limits(self.env)
        
        # Check if the string is empty or None
        if not action_str:
            return None
        
        try:
            # Extract action name and parameters
            action_name = action_str.split('(')[0].strip().lower()
            
            # Set the action type
            if action_name == "pick":
                action_array[0] = 1
            elif action_name == "place":
                action_array[1] = 1
            elif action_name == "push":
                action_array[2] = 1
            else:
                # Invalid action name
                return None
            
            # Extract parameters
            params_str = action_str.split('(')[1].split(')')[0]
            params = [float(p.strip()) for p in params_str.split(',')]
            
            # Check if we have the correct number of parameters
            if action_name in ["pick", "place"] and len(params) != 3:
                return None
            elif action_name == "push" and len(params) != 6:
                return None
            
            # Apply workspace constraints and scale
            # First point (x,y,z)
            params[0] = np.clip(params[0], workspace_x[0], workspace_x[1])
            params[1] = np.clip(params[1], workspace_y[0], workspace_y[1])
            params[2] = np.clip(params[2], workspace_z[0], workspace_z[1])
            
            # Second point (x1,y1,z1) if it exists (for push)
            if action_name == "push":
                params[3] = np.clip(params[3], workspace_x[0], workspace_x[1])
                params[4] = np.clip(params[4], workspace_y[0], workspace_y[1])
                params[5] = np.clip(params[5], workspace_z[0], workspace_z[1])
            
            # Fill the coordinate dimensions (after dividing by 1000 as in your modified function)
            for i in range(len(params)):
                action_array[i+3] = params[i]/1000.0
            
            return action_array
        
        except (IndexError, ValueError):
            # If any parsing error occurs, return None
            return None
        
if __name__ == "__main__":
    """
    Example usage of the manipulation environment.
    
    This code demonstrates how to create an instance of the environment,
    reset it, and interact with it using manual input actions.
    """
    # AlignTwoCube,PlaceTwoCube,PutAppleInDrawer,StackThreeCube
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Navigation Environment")
    parser.add_argument(
        "--env_id",
        type=str,
        default="AlignTwoCube",
    )
    args = parser.parse_args()
    
    config = PrimitiveSkillEnvConfig(record_video=True, video_record_dir="./test_manipulation_video",env_id=args.env_id, render_mode="vision")
    env = PrimitiveSkillEnv(config)
    
    print(env.system_prompt())
    obs, info = env.reset()
    print(obs["obs_str"])
    
    i = 0
    import os
    if config.render_mode == 'vision':
        os.makedirs("./test_manipulation", exist_ok=True)
        img = obs["multi_modal_data"][config.image_placeholder][0]
        img.save(f"./test_manipulation/manipulation_{i}.png")
    
    while True:
        i += 1
        action = input("Enter action:")
        #action = f"<think>Let me try this direction.</think><answer>{action}</answer>"
        obs, reward, done, info = env.step(action)
        print(obs["obs_str"])
        
        if config.render_mode == 'vision':
            img = obs["multi_modal_data"][config.image_placeholder][0]
            img.save(f"./test_manipulation/manipulation_{i}.png")
        
        if done:
            break
    
    print(f"Total reward: {env.compute_reward()}")
    print(info)
    env.close()