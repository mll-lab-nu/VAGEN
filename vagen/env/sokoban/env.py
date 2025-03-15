import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import numpy as np
import re
import copy
from typing import Tuple, Dict, Optional, List, Any, Union
from PIL import Image
from dataclasses import dataclass

from vagen.utils import NoLoggerWarnings
from vagen.utils import set_seed
from vagen.env.register import register
from vagen.env.sokoban.room_utils import generate_room
from vagen.env.base import (
    BaseEnv,
    BaseInterface,
    IMAGE_PLACEHOLDER
)

from vagen.env.utils import preprocess_text, convert_numpy_to_PIL

system_prompt = """
You are a helpful assistant. You first think about the reasoning process in the mind and then provides the user with the answer.
"""

instruction_template = """You are a Sokoban solver.

Sokoban Quick Guide
Goal: Push all boxes (X) onto targets (O).

Symbols:
# Wall | _ Floor | O Target | X Box | P You | √ = Box on Target | S = You on Target
The observation is a 2D grid of the current state of the Sokoban game.

Rules:
1. Push boxes (can't pull).
2. Avoid walls (#).

Actions you can take: Up, Down, Left, Right. You can only take one action at a time.
Up: move up to the cell above (to the above row).
Down: move down to the cell below (to the below row).
Left: move left to the cell to the left (to the left column).
Right: move right to the cell to the right (to the right column).

Rewards:
Move: -0.1
Box on target: +1.0
All boxes placed: +10.0

Include your thought in <think> </think> tags and your final answer in <answer> </answer> tags.
Your response should be like: <think> [Your thought] </think> <answer> [Your answer] </answer>
"""




init_observation_template = """
[Initial Observation]:
{observation}
Decide your next action.
"""

action_template = """After you answer {answer}, the extracted valid action is {valid_action}.\
After that, the observation is:
{observation}
reward: {reward}
done: {done}
"""


class SokobanEnv(BaseEnv, GymSokobanEnv):

    GRID_LOOKUP = {
        0: " # \t",  # wall
        1: " _ \t",  # floor
        2: " O \t",  # target
        3: " √ \t",  # box on target
        4: " X \t",  # box
        5: " P \t",  # player
        6: " S \t",  # player on target
        # Use tab separator to separate columns and \n\n to separate rows.
    }

    ACTION_LOOKUP = {
        0: "None",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }

    def __init__(self, **kwargs):
        BaseEnv.__init__(self)
        self.search_depth = kwargs.pop('search_depth', 300)
        GymSokobanEnv.__init__(
            self,
            dim_room=kwargs.pop('dim_room', (6, 6)), 
            max_steps=kwargs.pop('max_steps', 100),
            num_boxes=kwargs.pop('num_boxes', 3),
            **kwargs
        )
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)


    def _reset(self, seed: int):
        with NoLoggerWarnings():
            try:
                with set_seed(seed):
                    self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                        dim=self.dim_room,
                        num_steps=self.num_gen_steps,
                        num_boxes=self.num_boxes,
                        search_depth=self.search_depth
                    )
            except (RuntimeError, RuntimeWarning) as e:
                print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
                print("[SOKOBAN] Retry . . .")
                next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
                return self._reset(next_seed)
            
            # self.action_sequence = self._reverse_action_sequence(action_sequence)
            self.player_position = np.argwhere(self.room_state == 5)[0]
            self.num_env_steps = self.reward_last = self.boxes_on_target = 0
        
        return self._render(mode='text'), {}
    
    def _step(self, action: int):
        """
        - Step the environment with the given action.
        - Check if the action is effective (whether player moves in the env).

        TODO modify here after definition of RolloutManager
        """
        assert not self.success()
        result = {
            'step_reward': 0,
            'done': False,
            'info': {},
        }
        
        prev_player_position = self.player_position
        obs, step_reward, done, info = GymSokobanEnv.step(self, action, observation_mode='tiny_rgb_array')
        
        info['action_is_effective'] = not np.array_equal(prev_player_position, self.player_position)
        return obs, step_reward, done, info
     

    def _render(self, mode='text'):
        assert mode in ['text', 'list', 'state', 'rgb_array']

        if mode == 'rgb_array':
            img = self.get_image(mode, scale=1) # numpy array
            return img


        if mode == 'state':
            return np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
        
        room_state = self._render(mode='state').tolist()

        if mode == 'list':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?").strip("\t").strip()
            return [" ".join(lookup(cell) for cell in row) for row in room_state]
        
        if mode == 'text':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
            return "\n".join("".join(lookup(cell) for cell in row) for row in room_state)

    def close(self):
        GymSokobanEnv.close(self)

    def finished(self):
        return self.num_env_steps >= self.max_steps or self.success()

    def success(self):
        return self.boxes_on_target == self.num_boxes




@dataclass
class PreprocessResult:
    action_list: List[int]
    answer_list: List[str] # string of extracted answer (may be invalid action)
    valid_list: List[bool]
    think: str
    answer: str
    llm_raw_response: str

    def to_dict(self):
        return {
            'action_list': self.action_list,
            'answer_list': self.answer_list,
            'valid_list': self.valid_list,
            'think': self.think,
            'answer': self.answer,
            'llm_raw_response': self.llm_raw_response,
        }


@register(name="sokoban")
class SokobanInterface(BaseInterface):

    INVALID_ACTION = 0
    FORMAT_REWARD = 1
    ACTION_LOOKUP = {
        0: "None",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }

    def __init__(
            self,
            **env_config,
        ):
        super().__init__(**env_config)

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
    
    @classmethod
    def _preprocess(cls, text: str) -> PreprocessResult:
        """Preprocess the raw text from LLM into a list of actions.
        Ensure at least one action (may be invalid).

        Args:
            text: raw text from LLM

        Returns:
            PreprocessResult containing parsed actions and validity flags
        """
        first_step_preprocess = preprocess_text(text)
        preprocess_result = PreprocessResult(
            action_list=[],
            valid_list=[],
            answer_list=first_step_preprocess['answer_list'],
            think=first_step_preprocess['think'],
            answer=first_step_preprocess['answer'],
            llm_raw_response=text,
        )
        
        for answer in preprocess_result.answer_list:
            action = cls._extract_one_action(answer)
            if action != cls.INVALID_ACTION:
                preprocess_result.action_list.append(action)
                preprocess_result.valid_list.append(True)
            else:
                preprocess_result.action_list.append(cls.INVALID_ACTION)
                preprocess_result.valid_list.append(False)
        
        return preprocess_result
        
    @classmethod
    def _postprocess(
        cls, 
        env_state: Union[str, np.ndarray], 
        reward: float,
        done: bool,
        info: Dict,
        preprocess_result: PreprocessResult,
    ) -> Tuple[Dict, float, bool, Dict]:
        """Postprocess the environment feedback
        NOTE now assume there's only one image in the observation

        Args:
            env_state: environment state (text or numpy array (image))
            reward: reward of the environment
            done: whether the environment is done
            info: extra info
            preprocess_result: preprocess result

        Returns:
            Tuple[Dict, float, bool, Dict]: observation, reward, done, info
        """

        if isinstance(env_state, np.ndarray):
            env_state = convert_numpy_to_PIL(env_state)

        answer = preprocess_result.answer
        valid_action = []
        for action, valid in zip(preprocess_result.action_list, preprocess_result.valid_list):
            if valid:
                valid_action.append(cls.ACTION_LOOKUP[action])
            else:
                break

        observation = IMAGE_PLACEHOLDER if not isinstance(env_state, str) else env_state
        text_template = action_template.format(
            answer=answer,
            valid_action=valid_action,
            observation=observation,
            reward=reward,
            done=done,
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
        return obs, reward, done, info

    

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


        preprocess_result = self._preprocess(raw_text)
        think = preprocess_result.think
        action_list = preprocess_result.action_list
        valid_list = preprocess_result.valid_list
        answer = preprocess_result.answer
        final_info['llm_raw_response'] = preprocess_result.llm_raw_response

        # deal with format
        if think and answer: # format is correct
            reward += self.FORMAT_REWARD
        else:
            reward -= self.FORMAT_REWARD*0.1

        info = {}
        for action, valid in zip(action_list, valid_list):
            if done or self.env.finished():
                break
            if valid:
                _, env_reward, done, info = self.env.step(action)
                reward += env_reward
            else: # termiante at the first invalid action
                break
        self.traj_reward += reward
        final_info.update(info) # NOTE currently only use the last step info

        env_state = self.env._render(mode='text' if not self.visual_env else 'rgb_array') # NOTE currently called after step

        return self._postprocess(
            env_state=env_state,
            reward=reward,
            done=done,
            info=final_info,
            preprocess_result=preprocess_result,
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
    def config_repr(cls, config: Dict) -> str:
        """
        Create a string representation of the configuration.
        
        Args:
            config: Dictionary containing configuration
            
        Returns:
            String representation of the configuration
            
        Raises:
            ValueError: If required keys are missing from the configuration
        """
        required_keys = ['dim_room', 'num_boxes', 'max_steps', 'search_depth']
        
        # Check for required keys
        if not all(key in config for key in required_keys):
            missing_keys = [key for key in required_keys if key not in config]
            raise ValueError(f"Missing required keys in config: {missing_keys}")
            
        # Format the configuration string
        return (f"SokobanGame(dim_room={config['dim_room']}, "
                f"num_boxes={config['num_boxes']}, "
                f"max_steps={config['max_steps']}, "
                f"search_depth={config['search_depth']})")
    
    def get_task_instruction(self) -> str:
        return instruction_template
    
    def get_traj_reward(self):
        return self.traj_reward
