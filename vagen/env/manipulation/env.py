from vagen.env.base.base_env import BaseEnv
import numpy as np
import copy
from typing import Dict, List, Optional, Tuple, Any
from gymnasium.utils import seeding
from gymnasium.envs.toy_text.frozen_lake import manipulationEnv as GymmanipulationEnv
from vagen.env.utils.env_utils import NoLoggerWarnings, set_seed
from vagen.env.utils.context_utils import parse_llm_raw_response, convert_numpy_to_PIL
from .env_config import ManipulationEnvConfig
from .maniskill.utils import build_env, handel_info
from .prompts import system_prompt, init_observation_template, action_template

class ManipulationEnv(BaseEnv):
    def __init__(self, config: ManipulationEnvConfig):
        self.config = config
        self.env=build_env(config.env_id,record_dir=None)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.total_reward = 0
        _, info=self.env.reset(seed=seed)
        obs=self._render(info,init_obs=True)
        self.last_info=info
        return obs, info
    
    def step(self,action_str):
        reward=0
        rst = parse_llm_raw_response(
            response=action_str,
            special_token_list=self.config.special_token_list,
            action_sep=self.config.action_sep,
            max_actions=1
        )
        output_info={}
        output_info.update(rst)
        action =self._parse_action(rst['actions'][0]) if len(rst['actions']) > 0 else None
        valid_action=rst['actions'][0] if len(rst['actions']) > 0 else ""
        metrics = {
            "turn_metrics": {
                "action_is_valid": action is not None,  # True if at least one valid action was parsed
            },
            "traj_metrics": {
                "success": False,  # Will be set to True if agent reaches goal
            },
        }
        if metrics["turn_metrics"]['action_is_valid']:
            reward += self.config.format_reward
        new_obs, rews, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if info['is_success']:
            metrics["traj_metrics"]['success'] = True
        obs=self._render(info,init_obs=False,valid_action=valid_action)
        output_info["metrics"] = metrics
        self.total_reward += reward
        self.last_info = info
        return obs,reward,done,output_info
    
    def system_prompt(self):
        return system_prompt
    
    def close(self):
        self.env.close()
    
    def compute_reward(self):
        if self.last_info.get("success", False):
            return 10+self.total_reward
        
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
        return (max_stage + 1) * 2+self.total_reward
    
    
    def _render(self,info,init_obs=False,valid_action=None):
        new_info=handel_info(info.copy())
        object_positions=new_info['obj_positions']
        other_information=new_info['other_info']
        instruction=self.env.instruction
        img_placeholder = self.config.image_placeholder
        if init_obs:
            obs_str = init_observation_template.format(observation=img_placeholder, instruction=instruction, object_positions=object_positions, other_information=other_information)
        else:
            obs_str = action_template.format(valiad_action=valid_action,observation=img_placeholder, instruction=instruction, object_positions=object_positions, other_information=other_information)
        multi_modal_data = None
        if self.config.render_mode == "vision":
            img=self.env.render()
            multi_modal_data = {
                    img_placeholder: [convert_numpy_to_PIL(img)]
                }
        if multi_modal_data is not None:
            return {
                "obs_str": obs_str,
                "multi_modal_data": multi_modal_data,
            }
        else:
            return {
                "obs_str": obs_str,
            }
            
    import numpy as np

    def _parse_action(self,action_str):
        # Initialize empty 9-dim array (3 for action type, 6 for coordinates)
        action_array = np.zeros(9)
        
        # Workspace boundaries
        workspace_x = self.env.workspace_x
        workspace_y = self.env.workspace_y
        workspace_z = self.env.workspace_z
        
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
                return np.zeros(9)
            
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
            params[0] = np.clip(params[0], workspace_x[0]*1000, workspace_x[1]*1000)
            params[1] = np.clip(params[1], workspace_y[0]*1000, workspace_y[1]*1000)
            params[2] = np.clip(params[2], workspace_z[0]*1000, workspace_z[1]*1000)
            
            # Second point (x1,y1,z1) if it exists (for push)
            if action_name == "push":
                params[3] = np.clip(params[3], workspace_x[0]*1000, workspace_x[1]*1000)
                params[4] = np.clip(params[4], workspace_y[0]*1000, workspace_y[1]*1000)
                params[5] = np.clip(params[5], workspace_z[0]*1000, workspace_z[1]*1000)
            
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
    config = ManipulationEnvConfig()
    env = ManipulationEnv(config)
    
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
        action = f"<think>Let me try this direction.</think><answer>{action}</answer>"
        
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