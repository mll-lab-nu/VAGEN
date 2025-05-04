from vagen.env.base.base_env import BaseEnv
import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
from .utils import generate_room
from typing import Dict
from vagen.env.utils.env_utils import NoLoggerWarnings, set_seed
from vagen.env.utils.context_utils import convert_numpy_to_PIL
import numpy as np
from vagen.env.utils.parse_utils_3 import parse_function_map
from .prompt_3 import (
    system_prompt, 
    init_observation_template, 
    action_template,
    format_prompt
)
from .env_config import SokobanEnvConfig

class SokobanEnv(BaseEnv):
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
        "Up":1,
        "Down":2,
        "Left":3,
        "Right":4,
    }

    def __init__(self, config:SokobanEnvConfig):
        BaseEnv.__init__(self)
        self.config=config
        self.env=GymSokobanEnv(
            dim_room=self.config.get('dim_room', (6, 6)), 
            max_steps=self.config.get('max_steps', 100),
            num_boxes=self.config.get('num_boxes', 3),
        )
        
        # Get the appropriate format prompt function
        self.format_prompt_func = format_prompt[self.config.prompt_format]
        
        # Call the function with add_example=True for system prompt
    
        
        self.parse_func = parse_function_map[self.config.prompt_format.rstrip("_symbol")]
        
    def reset(self, seed=None):
        with NoLoggerWarnings():
            try:
                with set_seed(seed):
                    self.env.room_fixed, self.env.room_state, self.env.box_mapping, action_sequence = generate_room(
                        dim=self.env.dim_room,
                        num_steps=self.env.num_gen_steps,
                        num_boxes=self.env.num_boxes,
                        search_depth=self.config.get('search_depth', 100),
                    )
            except (RuntimeError, RuntimeWarning) as e:
                print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
                print("[SOKOBAN] Retry . . .")
                next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
                return self.reset(next_seed)
    
            self.env.player_position = np.argwhere(self.env.room_state == 5)[0]
            self.env.num_env_steps = self.env.reward_last = self.env.boxes_on_target = 0
        self.total_reward = 0
        return self._render(init_obs=True), {}
    
    def step(self, action_str: str):
        rst=self.parse_func(
            response=action_str,
            special_token_list=self.config.get('special_token_list', None),
            action_sep=self.config.get('action_sep', ','),
            max_actions=self.config.get('max_actions_per_step', 3)
        )
        #print("rst:", rst)
        action_list=rst['actions']
        prev_player_position = self.env.player_position
        
        metrics={
            "turn_metrics":{
                "action_is_valid": len(action_list) != 0,
                "action_is_effective": False,},
            "traj_metrics": {
                "success": False,
            }
        }
        
        self.reward=0
        self.valid_actions=[]
        done=False
        info={}
        info.update(rst)
        
        for action in action_list:
            if action in self.ACTION_LOOKUP:
                action_int=self.ACTION_LOOKUP[action]
                _,step_reward, _, _=self.env.step(action_int)
                done=self._success()
                self.reward+=step_reward
                self.valid_actions.append(action)
                if done:
                    metrics['traj_metrics']['success'] = True
                    break
            else:
                metrics['turn_metrics']['action_is_valid'] = False
                break
        if metrics['turn_metrics']['action_is_valid'] and rst["format_correct"]:
            self.reward += self.config.format_reward
        info["metrics"] = metrics
        metrics['turn_metrics']['action_is_effective'] = not np.array_equal(prev_player_position, self.env.player_position)
        self.total_reward += self.reward

        return self._render(init_obs=False), self.reward, done, info
    
    def system_prompt(self):
        format_prompt=self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=True  # Always true for system prompt
        )
        return system_prompt() + "\n" + format_prompt
    
    def compute_reward(self):
        return self.total_reward
    
    def close(self):
        self.env.close()
    
    def _render(self, init_obs=False):
        assert self.config.render_mode in ['text', 'vision']
        multi_modal_data = None
        
        # Get the appropriate format prompt function for action/init templates (with add_example=False)
        
        format_prompt = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False  # No examples for action and init obs
        )
        
        if self.config.render_mode == 'vision':
            img_placeholder=self.config.get("image_placeholder", "<image>")
            multi_modal_data={
                img_placeholder: [convert_numpy_to_PIL(self.env.render(mode='rgb_array'))],
                } 
            img_str=img_placeholder
        else:
            room_state = np.where((self.env.room_state == 5) & (self.env.room_fixed == 2), 6, self.env.room_state).tolist()
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
            img_str = "\n".join("".join(lookup(cell) for cell in row) for row in room_state)
        
        if init_obs:
            obs_str = init_observation_template(img_str) + "\n" + format_prompt
        else:
            obs_str = action_template(
                valid_action=self.valid_actions,
                observation=img_str,
            ) + "\n" + format_prompt
        
        if multi_modal_data is not None:
            return {
                "obs_str": obs_str,
                "multi_modal_data": multi_modal_data,
            }
        else:   
            return {
                "obs_str": obs_str,
            }
    
    def _success(self):
        return self.env.boxes_on_target == self.env.num_boxes
    
    
if __name__ == "__main__":
    kwargs = {
        'render_mode': 'vision',
    }
    config = SokobanEnvConfig(**kwargs)
    env = SokobanEnv(config)
    print(env.system_prompt())
    obs, info = env.reset()
    print(obs["obs_str"])
    i=0
    import os
    if config.render_mode == 'vision':
        os.makedirs("./test_sokoban", exist_ok=True)
        img = obs["multi_modal_data"][config.image_placeholder][0]
        img.save(f"./test_sokoban/sokoban_{i}.png")
    while True:
        i += 1
        action = input("Enter action (Left, Down, Right, Up): ")
        action = f"<think>Let me try this direction.</think><answer>{action}</answer>"
        obs, reward, done, info = env.step(action)
        print(obs["obs_str"])
        if config.render_mode == 'vision':
            # save the image
            img = obs["multi_modal_data"][config.image_placeholder][0]
            img.save(f"./test_sokoban/sokoban_{i}.png")
        if done:
            break
        
    print(f"Total reward: {env.compute_reward()}")
    print(info)
    env.close()