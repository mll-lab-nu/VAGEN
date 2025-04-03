import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import numpy as np
import re
import copy
from typing import Tuple, Dict, Optional, List, Any, Union
from PIL import Image

from vagen.utils import NoLoggerWarnings
from vagen.utils import set_seed
from vagen.env.register import register
from vagen.env.sokoban.room_utils import generate_room
from vagen.env_new.base_interface import (
    BaseInterface,
)

from vagen.env_new.utils.env_utils import (
    convert_numpy_to_PIL,
    preprocess,
    postprocess,
)
from .prompt import (
    init_observation_template,
    action_template,
    instruction_template,
)



@register(name="sokoban")
class SokobanInterface(BaseInterface):
    def __init__(
            self,
            config,
        ):
        """
        Args:
            env_config (Dict): environment configuration
            interface_config (Dict): interface configuration
        """
        super().__init__(env_config)

        dim_room = self.env_config['dim_room']
        num_boxes = self.env_config['num_boxes']
        max_steps = self.env_config['max_steps']
        search_depth = self.env_config['search_depth']
        self.env = SokobanEnv(
            dim_room=dim_room,
            num_boxes=num_boxes,
            max_steps=max_steps,
            search_depth=search_depth
        )
        self.visual_env = self.env_config.get('visual_env', True)

        max_action_per_step = interface_config.setdefault('max_action_per_step', 1)
        max_action_penalty = interface_config.setdefault('max_action_penalty', -0.5)
        format_reward = interface_config.setdefault('format_reward', 0.5)
        self.interface_config = {
            'max_action_per_step': max_action_per_step,
            'max_action_penalty': max_action_penalty,
            'format_reward': format_reward,
        }
        
    @classmethod
    def _extract_one_action(cls, text):
        """
        Extract single action from text, the input text should ensure only one action contained
        - 0: Still (Invalid Action)
        - 1: Up
        - 2: Down
        - 3: Left
        - 4: Right
        """
        DIRECTION_MAP = {"Up": 1, "Down": 2, "Left": 3, "Right": 4}
        # TODO: originally, we parse either number (key of direction_map) or direction (value of direction_map).
        # here we remove numbers and preserve directions only, but regex has not been removed. please remove them later.
        pattern = r'^\s*(([1-4])\s*\((up|down|left|right)\)|(up|down|left|right)|([1-4]))\s*$'
        match = re.fullmatch(pattern, text.strip(), flags=re.IGNORECASE | re.X)
        
        if not match:
            return cls.INVALID_ACTION
        
        if match.group(2):   
            return int(match.group(2))
        elif match.group(4): 
            return DIRECTION_MAP[match.group(4).capitalize()]
        elif match.group(5): 
            return int(match.group(5))
        
        return cls.INVALID_ACTION
    

    def _step(self, raw_text: str) -> Tuple[Any, float, bool, Dict]:
        """Step the environment with llm raw response
        - Multiple actions are allowed, execute until the first invalid action or environment terminates
        - The observation is the last step observation
        
        Args:
            raw_text: raw text from LLM

        Returns:
            Tuple[Any, float, bool, Dict]: observation, reward, done, info
            - observation (dict): observation of the environment
            - reward (float): reward of the environment for the raw_text (multiple actions, including format reward and env reward)
            - done (bool): whether the environment is done
            - info (dict): extra info
        """

        assert not self.env.finished(), "Environment finished before step"
        reward, done, final_info = 0, False, {}


        # preprocess_result = self._preprocess(raw_text)
        preprocess_result = preprocess(raw_text, self._extract_one_action, self.INVALID_ACTION)
        think = preprocess_result.think
        action_list = preprocess_result.action_list
        answer = preprocess_result.answer
        final_info['llm_raw_response'] = preprocess_result.llm_raw_response


        # parse format and action list
        if action_list:
            reward += self.interface_config['format_reward']
        if len(action_list) > self.interface_config['max_action_per_step']:
            reward += self.interface_config['max_action_penalty']
            action_list = action_list[:self.interface_config['max_action_per_step']]
            preprocess_result.action_list = action_list
            

        info = {}
        for action in action_list:
            if done or self.env.finished():
                break
            _, env_reward, done, info = self.env.step(action)
            # if env_reward == -0.1:
            #     env_reward = -0.01 # NOTE hard coded here to set step reward to 0
            reward += env_reward
        self.traj_reward += reward
        final_info.update(info) # NOTE currently only use the last step info

        env_state = self.env._render(mode='text' if not self.visual_env else 'rgb_array') # NOTE currently called after step

        return postprocess(
            env_state=env_state,
            reward=reward,
            done=done,
            info=final_info,
            preprocess_result=preprocess_result,
            action_lookup=self.ACTION_LOOKUP,
            action_template=action_template,
        )
    
    def _reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment and return the observation at the first step.
        """
        self.env._reset(seed=seed)
        self.traj_reward = 0
        env_state = self.env._render(mode='text' if not self.visual_env else 'rgb_array') # NOTE currently called after reset
        if isinstance(env_state, np.ndarray):
            env_state = convert_numpy_to_PIL(env_state)
        observation = IMAGE_PLACEHOLDER if not isinstance(env_state, str) else env_state
        text_template = init_observation_template.format(
            observation=observation,
        )
        if isinstance(env_state, str):
            obs = {'text_template': text_template}
        else:
            obs = {
                'text_template': text_template,
                'multi_modal_data': {
                    IMAGE_PLACEHOLDER: [env_state],
                },
            }
        return obs, {}

    def close(self):
        self.env.close()

    @classmethod
    def config_repr(cls, env_config: Dict, interface_config: Dict) -> str:
        """
        Create a string representation of the configuration.
        
        Args:
            env_config: Dictionary containing environment configuration
            interface_config: Dictionary containing interface configuration
            
        Returns:
            String representation of the configuration
            
        Raises:
            ValueError: If required keys are missing from the configuration
        """

        required_keys = ['dim_room', 'num_boxes', 'max_steps', 'search_depth']
        
        # Check for required keys
        if not all(key in env_config for key in required_keys):
            missing_keys = [key for key in required_keys if key not in env_config]
            raise ValueError(f"Missing required keys in config: {missing_keys}")
            
        env_config_str = (
            f"SokobanGame(dim_room={env_config['dim_room']}, "
            f"num_boxes={env_config['num_boxes']}, "
            f"max_steps={env_config['max_steps']}, "
            f"search_depth={env_config['search_depth']})"
        )
        interface_config_str = (
            f"SokobanInterface(max_action_per_step={interface_config.get('max_action_per_step', 1)}, "
            f"max_action_penalty={interface_config.get('max_action_penalty', -0.1)}, "
            f"format_reward={interface_config.get('format_reward', 0.5)})"
        )
        return f"{env_config_str}, {interface_config_str}"
    def get_task_instruction(self) -> str:
        return instruction_template.format(
            max_action_per_step=self.interface_config['max_action_per_step']
        )
    
    def get_traj_reward(self):
        return self.traj_reward