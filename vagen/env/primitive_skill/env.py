from vagen.env.base.base_env import BaseEnv
import numpy as np
import copy
from typing import Dict, List, Optional, Tuple, Any
from gymnasium.utils import seeding
from vagen.env.utils.context_utils import parse_llm_raw_response, convert_numpy_to_PIL
from .env_config import PrimitiveSkillEnvConfig
from .maniskill.utils import build_env, handel_info, get_workspace_limits
from .prompts import system_prompt, init_observation_template, action_template
import vagen.env.primitive_skill.maniskill.env

class PrimitiveSkillEnv(BaseEnv):
    def __init__(self, config: PrimitiveSkillEnvConfig):
        self.config = config
        self.env=build_env(config.env_id,record_dir='./test')
    
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
            max_actions=self.config.max_actions_per_step,
        )
        output_info={}
        output_info.update(rst)
        valid_actions = []
        metrics = {
            "turn_metrics": {
                "action_is_valid": True,  # True if at least one valid action was parsed
            },
            "traj_metrics": {
                "success": False,  # Will be set to True if agent reaches goal
            },
        }
        for action in rst['actions']:
            parsed_action = self._parse_action(action)
            if parsed_action is not None:
                _, _, terminated, truncated, info = self.env.step(parsed_action)
                valid_actions.append(action)
                self.last_info = info
            else:
                info=self.last_info
                terminated, truncated = False, False
                metrics["turn_metrics"]['action_is_valid'] = False
                break
            if truncated or terminated:
                break
        if metrics["turn_metrics"]['action_is_valid']:
            reward += self.config.format_reward
        if info['is_success']:
            metrics["traj_metrics"]['success'] = True
        done= terminated or truncated
        info["action_is_valid"] = metrics["turn_metrics"]['action_is_valid']
        obs=self._render(info,init_obs=False,valid_actions=valid_actions)
        output_info["metrics"] = metrics
        self.total_reward += reward
        return obs,reward,done,output_info
    
    def system_prompt(self):
        return system_prompt.format(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
        )
    
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
    
    
    def _render(self,info,init_obs=False,valid_actions=None):
        new_info=handel_info(info.copy())
        object_positions=new_info['obj_positions']
        other_information=new_info['other_info']
        instruction=self.env.instruction()
        img_placeholder = self.config.image_placeholder
        x_workspace, y_workspace, z_workspace = get_workspace_limits(self.env)
        
        if init_obs:
            obs_str = init_observation_template.format(observation=img_placeholder, 
                                                       instruction=instruction, 
                                                       object_positions=object_positions, 
                                                       other_information=other_information,
                                                       x_workspace=x_workspace,
                                                       y_workspace=y_workspace,
                                                       z_workspace=z_workspace,
                                                       max_action=self.config.max_actions_per_step)
        else:
            obs_str = action_template.format(valid_actions=valid_actions,
                                             observation=img_placeholder, 
                                             instruction=instruction, 
                                             object_positions=object_positions, 
                                             other_information=other_information,
                                             x_workspace=x_workspace,
                                             y_workspace=y_workspace,
                                             z_workspace=z_workspace,
                                             max_action=self.config.max_actions_per_step)
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
    config = PrimitiveSkillEnvConfig()
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