from vagen.env_new.base_env import BaseEnv
import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
from vagen.env.sokoban.room_utils import generate_room
from typing import Dict


class SokobanVisionEnv(BaseEnv, GymSokobanEnv):

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

    def __init__(self, config: Dict):
        BaseEnv.__init__(self)
        self.config=config
        GymSokobanEnv.__init__(
            self,
            dim_room=kwargs.pop('dim_room', (6, 6)), 
            max_steps=kwargs.pop('max_steps', 100),
            num_boxes=kwargs.pop('num_boxes', 3),
            **kwargs
        )
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)


    def reset(self, seed: int):
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
    
    def step(self, action: int):
        """
        - Step the environment with the given action.
        - Check if the action is effective (whether player moves in the env).

        TODO modify here after definition of RolloutManager
        """
        assert not self._success()
        result = {
            'step_reward': 0,
            'done': False,
            'info': {},
        }
        
        prev_player_position = self.player_position
        obs, step_reward, done, info = GymSokobanEnv.step(self, action, observation_mode='tiny_rgb_array')
        
        info['action_is_effective'] = not np.array_equal(prev_player_position, self.player_position)
        return obs, step_reward, done, info



    def close(self):
        GymSokobanEnv.close(self)

    def _finished(self):
        return self.num_env_steps >= self.max_steps or self.success()

    def _success(self):
        return self.boxes_on_target == self.num_boxes