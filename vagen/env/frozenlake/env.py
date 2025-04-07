from vagen.env.base_env import BaseEnv
import numpy as np
import copy
from typing import Dict, List, Optional, Tuple, Any
from gymnasium.utils import seeding
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
from vagen.env.utils.env_utils import NoLoggerWarnings, set_seed
from vagen.env.utils.context_utils import parse_llm_raw_response, convert_numpy_to_PIL
from .prompt import system_prompt_text, system_prompt_vision, init_observation_template, action_template
from .config import FrozenLakeConfig
from .utils import generate_random_map, is_valid

class FrozenLakeEnv(BaseEnv):
    # Map gym state in integer
    MAP_LOOKUP = {
        b"P": 0,  # player
        b"F": 1,  # frozen
        b"H": 2,  # hole
        b"G": 3,  # goal
    }

    # Define rules to transform to rendered text observation of the environment
    GRID_LOOKUP = {
        0: " P \t",  # player
        1: " _ \t",  # frozen
        2: " O \t",  # hole
        3: " G \t",  # goal
        4: " X \t",  # player fall into hole
        5: " √ \t",  # player on goal
    }

    ACTION_LOOKUP = {
        "Left": 0,
        "Down": 1,
        "Right": 2,
        "Up": 3,
    }

    def __init__(self, config: FrozenLakeConfig):
        BaseEnv.__init__(self)
        self.config = config
       
        
        if self.config.desc is None:
            random_map = generate_random_map(size=self.config.size, p=self.config.p)
        else:
            random_map = np.asarray(copy.deepcopy(self.config.desc), dtype="c")
            
        self.gym_env = GymFrozenLakeEnv(
            desc=random_map,
            is_slippery=self.config.is_slippery
        )
        
        self.total_reward = 0
        self.valid_actions = []
        self.reward = 0

    def reset(self, seed=None):
        """Reset the environment with seed"""
        with NoLoggerWarnings():
            with set_seed(seed):
                self.gym_env.reset(seed=seed)
        self.total_reward = 0
        return self._render(init_obs=True), {}

    def step(self, action_str: str):
        rst = parse_llm_raw_response(
            response=action_str,
            special_token_list=self.config.special_token_list,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step
        )
        
        action_list = rst['actions']
        prev_player_position = self._get_player_position()
        
        metrics = {
            "turn_metrics": {
                "action_is_valid": True,
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            },
        }
        
        self.reward = 0
        self.valid_actions = []
        done = False
        info = {}
        info.update(rst)
        
        for action in action_list:
            if action in self.ACTION_LOOKUP:
                action_int = self.ACTION_LOOKUP[action]
                _, step_reward, terminated, _, _ = self.gym_env.step(action_int)
                self.reward += step_reward
                self.valid_actions.append(action)
                done=self._finished()
                assert terminated == done
                if done:
                    if self._success():
                        metrics["traj_metrics"]['success'] = True
                    break
            else:
                metrics["turn_metrics"]['action_is_valid'] = False
                break
        
        if metrics["turn_metrics"]['action_is_valid']:
            self.reward += self.config.format_reward
        
        info["metrics"] = metrics
        metrics["turn_metrics"]['action_is_effective'] = not np.array_equal(prev_player_position, self._get_player_position())
        self.total_reward += self.reward
        
        return self._render(init_obs=False), self.reward, done, info

    def system_prompt(self):
        """Return the system prompt based on render mode"""
        if self.config.render_mode == 'vision':
            return system_prompt_vision.format(max_actions_per_step=self.config.max_actions_per_step)
        else:
            return system_prompt_text.format(max_actions_per_step=self.config.max_actions_per_step)

    def compute_reward(self):
        """Return the total reward collected so far"""
        return self.total_reward

    def close(self):
        """Close the environment"""
        self.gym_env.close()


    def _get_player_position(self):
        """Get the current player position"""
        return (self.gym_env.s // self.gym_env.ncol, self.gym_env.s % self.gym_env.ncol)  # (row, col)

    def _render(self, init_obs=False):
        """Render the environment"""
        multi_modal_inputs = None
        
        if self.config.render_mode == 'vision':
            img_placeholder = self.config.image_placeholder
            multi_modal_inputs = {
                img_placeholder: [convert_numpy_to_PIL(self.gym_env._render_gui(mode='rgb_array'))]
            }
            img_str = img_placeholder
        else:
            room_state = self._get_text_representation()
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
            img_str = "\n".join("".join(lookup(cell) for cell in row) for row in room_state)
        
        if init_obs:
            obs_str = init_observation_template.format(observation=img_str)
        else:
            obs_str = action_template.format(
                valid_action=self.valid_actions,
                observation=img_str,
                reward=self.reward,
                done=self._finished(),
            )
        
        if multi_modal_inputs is not None:
            return {
                "obs_str": obs_str,
                "multi_modal_inputs": multi_modal_inputs,
            }
        else:
            return {
                "obs_str": obs_str,
            }

    def _get_text_representation(self):
        """Get the text representation of the environment"""
        room_state = copy.deepcopy(self.gym_env.desc)
        
        # Replace the position of start 'S' with 'F'
        position_S = np.where(room_state == b'S')
        room_state[position_S] = b'F'
        
        # Convert characters to internal representation
        room_state = np.vectorize(lambda x: self.MAP_LOOKUP.get(x, 0))(room_state)
        
        # Add player position
        position_P = self._get_player_position()
        player_cell = room_state[position_P]
        
        # Handle special cases: player on goal or hole
        if self.gym_env.desc[position_P] == b'H':
            room_state[position_P] = 4  # player in hole
        elif self.gym_env.desc[position_P] == b'G':
            room_state[position_P] = 5  # player on goal
        else:
            room_state[position_P] = 0  # normal player
            
        return room_state

    def _success(self):
        """Check if the agent has reached the goal"""
        player_pos = self._get_player_position()
        return self.gym_env.desc[player_pos] == b'G'
    
    def _finished(self):
        """Check if the episode is done"""
        player_pos = self._get_player_position()
        return self.gym_env.desc[player_pos] in [b'G', b'H']


if __name__ == "__main__":
    config = FrozenLakeConfig()
    env = FrozenLakeEnv(config)
    print(env.system_prompt())
    obs, info = env.reset()
    print(obs["obs_str"])
    i=0
    import os
    if config.render_mode == 'vision':
        os.makedirs("./test_frozenlake", exist_ok=True)
        img = obs["multi_modal_inputs"][config.image_placeholder][0]
        img.save(f"./test_frozenlake/frozenlake_{i}.png")
    while True:
        i += 1
        action = input("Enter action (Left, Down, Right, Up): ")
        action = f"<think>Let me try this direction.</think><answer>{action}</answer>"
        obs, reward, done, info = env.step(action)
        print(obs["obs_str"])
        if config.render_mode == 'vision':
            # save the image
            img = obs["multi_modal_inputs"][config.image_placeholder][0]
            img.save(f"./test_frozenlake/frozenlake_{i}.png")
        if done:
            break
        
    
    print(f"Total reward: {env.compute_reward()}")
    print(info)
    env.close()