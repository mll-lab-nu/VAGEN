from vagen.env.base.base_env import BaseEnv
import numpy as np
import copy
from typing import Dict, List, Optional, Tuple, Any
from gymnasium.utils import seeding
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
from vagen.env.utils.env_utils import NoLoggerWarnings, set_seed
from vagen.env.utils.context_utils import convert_numpy_to_PIL
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from .prompt import system_prompt, init_observation_template, action_template, format_prompt
from .env_config import FrozenLakeEnvConfig
from .utils import generate_random_map, is_valid
from vagen.env.utils.state_reward_text_utils import env_state_reward_wrapper
from .utils import state_to_sentences, convert_frozenlake_state_to_relative_list
class FrozenLakeEnv(BaseEnv):
    """
    FrozenLake Environment for training and evaluating language models as agents.
    
    This environment implements a grid-world where an agent must navigate from a starting
    position to a goal while avoiding holes in the ice. It is designed specifically for
    Large Language Models (LLMs) as agents, providing structured observations and handling
    text-based actions.
    """
    
    # Map gym state characters to integer representation
    MAP_LOOKUP = {
        b"P": 0,  # player
        b"F": 1,  # frozen
        b"H": 2,  # hole
        b"G": 3,  # goal
    }

    # Define text representations for rendering the environment
    GRID_LOOKUP = {
        0: " P \t",  # player
        1: " _ \t",  # frozen
        2: " O \t",  # hole
        3: " G \t",  # goal
        4: " X \t",  # player fall into hole
        5: " √ \t",  # player on goal
    }

    # Map action strings to Gymnasium action integers
    ACTION_LOOKUP = {
        "Left": 0,
        "Down": 1,
        "Right": 2,
        "Up": 3,
    }

    def __init__(self, config: FrozenLakeEnvConfig):
        """
        Initialize the FrozenLake environment.
        
        Args:
            config (FrozenLakeEnvConfig): Configuration parameters for the environment
                including map size, slipperiness, rendering mode, etc.
        """
        BaseEnv.__init__(self)
        self.config = config
       
        # Generate a random map if none is provided, otherwise use the provided map
        if self.config.desc is None:
            random_map = generate_random_map(size=self.config.size, p=self.config.p)
        else:
            random_map = np.asarray(copy.deepcopy(self.config.desc), dtype="c")
            
        # Initialize the underlying Gymnasium environment
        self.gym_env = GymFrozenLakeEnv(
            desc=random_map,
            is_slippery=self.config.is_slippery
        )
        
        # Initialize episode state
        self.total_reward = 0
        self.valid_actions = []
        self.reward = 0
        
        # Store the format prompt function for later use
        self.format_prompt_func = format_prompt[self.config.prompt_format]
        
        self.parse_func = PARSE_FUNC_MAP[self.config.prompt_format]

    def reset(self, seed=None):
        """
        Reset the environment to an initial state.
        
        This method resets the underlying Gymnasium environment and initializes
        tracking variables. If a seed is provided, it ensures deterministic
        environment generation.
        
        Args:
            seed (Optional[int]): Random seed for environment generation
                                  If None, a random seed is used
        
        Returns:
            Tuple[Dict, Dict]: 
                - obs: Dictionary containing observation string and optional image data
                - info: Empty dictionary for initial state
        """
        with NoLoggerWarnings():
            with set_seed(seed):
                self.gym_env.reset(seed=seed)
        self.total_reward = 0
        return self._render(init_obs=True), {}

    @env_state_reward_wrapper
    def step(self, action_str: str):
        """
        Take a step in the environment based on the agent's action.
        
        This method:
        1. Parses the raw LLM response to extract actions
        2. Executes each valid action in sequence
        3. Calculates rewards and metrics
        4. Generates the next observation
        
        The action string is expected to be the raw output from an LLM, which 
        may contain special tokens for thought processes or other structured content.
        
        Args:
            action_str (str): Raw string from LLM containing actions
        
        Returns:
            Tuple[Dict, float, bool, Dict]:
                - obs: Dictionary with observation string and optional image data
                - reward: Numeric reward for the step
                - done: Boolean indicating if episode is complete
                - info: Dictionary containing metrics and parsed action data
        """
        # Parse the LLM's raw response to extract actions
        rst = self.parse_func(
            response=action_str,
            special_token_list=self.config.special_token_list,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step
        )
        
        action_list = rst['actions']
        prev_player_position = self._get_player_position()
        
        # Initialize metrics for this step
        metrics = {
            "turn_metrics": {
                "action_is_valid": len(action_list) != 0,  # True if at least one valid action was parsed
                "action_is_effective": False,  # Will be updated after actions are executed
            },
            "traj_metrics": {
                "success": False,  # Will be set to True if agent reaches goal
            },
        }
        
        # Reset step-specific state
        self.reward = 0
        self.valid_actions = []
        done = False
        info = {}
        info.update(rst)  # Include parsed action data in info
        
        # Execute each action in the list until done or all actions processed
        for action in action_list:
            if action in self.ACTION_LOOKUP:
                # Convert string action to integer and execute in gym environment
                action_int = self.ACTION_LOOKUP[action]
                _, step_reward, terminated, _, _ = self.gym_env.step(action_int)
                self.reward += step_reward
                self.valid_actions.append(action)
                done = self._finished()
                assert terminated == done
                if done:
                    # If episode is done and successful, add bonus reward
                    if self._success():
                        metrics["traj_metrics"]['success'] = True
                        self.reward += 9  # Bonus reward for reaching goal
                    break
            else:
                # If an invalid action is encountered, mark actions as invalid and stop
                metrics["turn_metrics"]['action_is_valid'] = False
                break
        
        # Add format reward if actions were valid
        if metrics["turn_metrics"]['action_is_valid'] and rst["format_correct"]:
            self.reward += self.config.format_reward
            info["is_format_rewarded"] = True
        else:
            info["is_format_rewarded"] = False
        
        
        # Check if position changed to determine if action was effective
        metrics["turn_metrics"]['action_is_effective'] = not np.array_equal(prev_player_position, self._get_player_position())
        info["metrics"] = metrics
        # Update total reward for the episode
        self.total_reward += self.reward
        
        # Generate observation, return result tuple
        return self._render(init_obs=False), self.reward, done, info

    def system_prompt(self):
        """
        Get the system prompt for the environment.
        
        Returns a prompt explaining the environment to the LLM agent,
        with different prompts for text and vision modes.
        
        Returns:
            str: System prompt string with environment description and instructions
        """
        # Get format prompt with examples for system prompt
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=True  # Always true for system prompt
        )
        
        return system_prompt() + '\n' + format_prompt_text

    def close(self):
        self.gym_env.close()

    def _get_player_position(self):
        return (self.gym_env.s // self.gym_env.ncol, self.gym_env.s % self.gym_env.ncol)  # (row, col)

    def _render(self, init_obs=False):
        """
        Render the environment as an observation.
        
        This method creates either a text representation or an image of the environment
        state, depending on the configured render mode. It formats the observation string
        based on whether this is the initial observation or a subsequent one.
        
        Args:
            init_obs (bool): If True, create initial observation; otherwise create a
                            step observation that includes action results
        
        Returns:
            Dict: Observation dictionary containing:
                - "obs_str": String observation for the LLM
                - "multi_modal_data": Optional dictionary with image data for vision mode
        """
        multi_modal_data = None
        
        # Get format prompt without examples for action/init templates
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False  # No examples for action and init obs
        )
        
        # Generate either vision or text representation
        if self.config.render_mode == 'vision':
            # For vision mode, generate an image of the environment
            img_placeholder = self.config.image_placeholder
            multi_modal_data = {
                img_placeholder: [convert_numpy_to_PIL(self.gym_env._render_gui(mode='rgb_array'))]
            }
            observation = img_placeholder  # In the text, just use the placeholder
        else:
            # For text mode, generate a text grid representation
            room_state = self._get_text_representation()
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
            observation = "\n".join("".join(lookup(cell) for cell in row) for row in room_state)
        
        # Format the observation string using the appropriate template
        if init_obs:
            # Initial observation doesn't include action results
            obs_str = init_observation_template(observation=observation) + "\n" + format_prompt_text
        else:
            # Subsequent observations include action results
            obs_str = action_template(
                valid_action=self.valid_actions,
                observation=observation,
            ) + "\n" + format_prompt_text
        
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

    def _get_text_representation(self):
        room_state = copy.deepcopy(self.gym_env.desc)
        
        position_S = np.where(room_state == b'S')
        room_state[position_S] = b'F'
        
        room_state = np.vectorize(lambda x: self.MAP_LOOKUP.get(x, 0))(room_state)
        
        position_P = self._get_player_position()
        player_cell = room_state[position_P]
        
        if self.gym_env.desc[position_P] == b'H':
            room_state[position_P] = 4  # player in hole
        elif self.gym_env.desc[position_P] == b'G':
            room_state[position_P] = 5  # player on goal
        else:
            room_state[position_P] = 0  # normal player on frozen tile
            
        return room_state

    def _success(self):
        player_pos = self._get_player_position()
        return self.gym_env.desc[player_pos] == b'G'
    
    def _finished(self):
        player_pos = self._get_player_position()
        return self.gym_env.desc[player_pos] in [b'G', b'H']

    def get_env_state(self):
        """
        Get the current state of the environment as a dictionary.
        
        Returns:
            Dict: Contains player position, target position, and hole positions
                as coordinate tuples (row, col)
        """
        # Get dimensions of the grid
        nrow, ncol = self.gym_env.desc.shape
        

        player_position = player_position = tuple(map(int, self._get_player_position()))
        
        target_position = tuple(map(int, np.argwhere(self.gym_env.desc == b'G')[0]))
        
        hole_positions = [tuple(map(int, pos)) for pos in np.argwhere(self.gym_env.desc == b'H')]
        state_dict={
            "player_position": player_position,
            "target_position": target_position,
            "hole_positions": hole_positions,
            "grid_size": (nrow, ncol),
        }
        return convert_frozenlake_state_to_relative_list(state_dict)

if __name__ == "__main__":
    """
    Example usage of the FrozenLake environment.
    
    This code demonstrates how to create an instance of the environment,
    reset it, and interact with it using manual input actions.
    """
    config = FrozenLakeEnvConfig()
    env = FrozenLakeEnv(config)
    
    print(env.system_prompt())
    obs, info = env.reset()
    print(obs["obs_str"])
    
    i = 0
    import os
    if config.render_mode == 'vision':
        os.makedirs("./test_frozenlake", exist_ok=True)
        img = obs["multi_modal_data"][config.image_placeholder][0]
        img.save(f"./test_frozenlake/frozenlake_{i}.png")
    
    while True:
        i += 1
        action = input("Enter action (Left, Down, Right, Up): ")
        action = f"<think>Let me try this direction.</think><answer>{action}</answer>"
        
        obs, reward, done, info = env.step(action)
        print(obs["obs_str"])
        
        if config.render_mode == 'vision':
            img = obs["multi_modal_data"][config.image_placeholder][0]
            img.save(f"./test_frozenlake/frozenlake_{i}.png")
        
        if done:
            break
    
    print(f"Total reward: {env.compute_reward()}")
    print(info)
    env.close()