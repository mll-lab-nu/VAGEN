import numpy as np
import copy
from PIL import Image
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv

from vagen.envs.frozenlake.utils.prompt import (
    action_template,
    format_prompt,
    init_observation_template,
    system_prompt,
)
from vagen.envs.frozenlake.utils.utils import parse_response, numpy_to_pil, generate_random_map

from vagen.envs.gym_image_env import GymImageEnv

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional


@dataclass
class FrozenLakeEnvConfig:
    """Configuration for FrozenLake environment"""
    size: int = 4                      # Grid size (size x size)
    desc: Optional[List[str]] = None   # Custom map description
    is_slippery: bool = False          # Whether the ice is slippery (gym default 1/3 slip)
    slip_prob: float = 0.0             # Custom slip probability (0.02 = 2% chance to slip)
    p: float = 0.8                     # Probability of frozen tile when generating random map
    render_mode: str = "text"          # "text" or "vision"
    max_actions_per_step: int = 5      # Max actions per step
    action_sep: str = ","              # Separator between actions
    image_placeholder: str = "<image>" # Placeholder for vision mode
    use_example_in_sys_prompt: bool = True  # Whether to add example system prompt
    prompt_format: str = "free_think"  # "free_think" or "wm"
    format_reward: float = 0.02         # Reward for following the format correctly
    success_reward: float = 1.0       # Reward for reaching the goal


class FrozenLake(GymImageEnv):
    """
    FrozenLake environment that implements the GymImageEnv async interface.
    Uses asyncio.to_thread(...) to offload blocking gym calls (reset/step/render/close)
    to a thread pool so the event loop is not blocked.
    """

    # Map gym state characters to integer representation
    MAP_LOOKUP = {
        b"P": 0,  # player
        b"F": 1,  # frozen
        b"H": 2,  # hole
        b"G": 3,  # goal
        b"S": 1,  # start (treat as frozen)
    }

    # Text rendering lookup
    GRID_LOOKUP = {
        0: " P ",  # player
        1: " _ ",  # frozen
        2: " O ",  # hole
        3: " G ",  # goal
        4: " X ",  # player fell into hole
        5: " V ",  # player on goal (victory)
    }

    # Action mapping
    ACTION_LOOKUP = {
        "left": 0,
        "down": 1,
        "right": 2,
        "up": 3,
    }

    # Perpendicular actions for slipping: action -> [perpendicular1, perpendicular2]
    PERPENDICULAR_ACTIONS = {
        0: [1, 3],  # left -> down, up
        1: [0, 2],  # down -> left, right
        2: [1, 3],  # right -> down, up
        3: [0, 2],  # up -> left, right
    }

    def __init__(self, env_config: Dict[str, Any]):
        """
        :param env_config: a Dict with keys mapped to FrozenLakeEnvConfig
        """
        super().__init__(env_config)
        self.config = FrozenLakeEnvConfig(**env_config)
        self.gym_env: Optional[GymFrozenLakeEnv] = None
        self.total_reward: float = 0.0
        self.valid_actions: List[str] = []

    def _create_gym_env(self, seed: Optional[int] = None) -> GymFrozenLakeEnv:
        """Create the underlying gym environment with a new map."""
        if self.config.desc is None:
            random_map = generate_random_map(
                size=self.config.size,
                p=self.config.p,
                seed=seed
            )
        else:
            random_map = copy.deepcopy(self.config.desc)

        return GymFrozenLakeEnv(
            desc=random_map,
            is_slippery=self.config.is_slippery,
            render_mode="rgb_array"  # Always use rgb_array for potential vision mode
        )

    # ------------------------------
    # GymImageEnv abstract methods
    # ------------------------------
    async def close(self) -> None:
        """Non-blocking close via to_thread to avoid blocking the loop."""
        if self.gym_env is not None:
            await asyncio.to_thread(self.gym_env.close)

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Non-blocking reset:
        - Creates a new gym environment with a randomly generated map
        - Offloads env.reset() to a thread pool to avoid blocking the event loop.
        """
        # Create new environment with the seed
        self.gym_env = self._create_gym_env(seed=seed)
        await asyncio.to_thread(self.gym_env.reset, seed=seed)

        self.total_reward = 0.0
        self.valid_actions = []
        obs = await self._render_async(init_obs=True)
        info: Dict[str, Any] = {}
        return obs, info

    async def system_prompt(self) -> Dict[str, Any]:
        """
        Non-blocking system prompt:
        - Returns the system prompt observation.
        """
        return {
            "obs_str": self.get_system_prompt(),
        }

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Non-blocking step:
        - Parses action_str
        - Offloads env.step(...) to thread pool for each primitive action
        - Computes metrics, reward shaping, success, etc.
        """
        parsed = parse_response(
            response=action_str,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step,
            prompt_format=self.config.prompt_format,
        )
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        self.valid_actions = []
        info.update(parsed)
        action_list: List[str] = parsed.get("actions", [])

        # Copy current player position
        prev_player_pos = self._get_player_position()

        metrics = {
            "turn_metrics": {
                "action_is_valid": len(action_list) > 0 and parsed.get("format_correct", False),
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            },
        }

        for action in action_list:
            if action in self.ACTION_LOOKUP:
                action_int = self.ACTION_LOOKUP[action]
                # Apply custom slip probability if configured
                action_int = self._apply_slip(action_int)
                # Offload the blocking gym step to a thread
                _obs, step_reward, terminated, truncated, _ = await asyncio.to_thread(
                    self.gym_env.step, action_int
                )
                self.valid_actions.append(action)

                # Check if episode is done
                done = self._finished()

                if done:
                    if self._is_success():
                        reward += self.config.success_reward
                        metrics["traj_metrics"]["success"] = True
                    break
            else:
                metrics["turn_metrics"]["action_is_valid"] = False
                break

        # Add format reward if actions were valid
        if self.valid_actions and parsed.get("format_correct", False):
            reward += self.config.format_reward

        # Effective action: detect player position change
        metrics["turn_metrics"]["action_is_effective"] = not np.array_equal(
            prev_player_pos, self._get_player_position()
        )

        info["metrics"] = metrics
        info["success"] = metrics["traj_metrics"]["success"]
        self.total_reward += reward

        obs = await self._render_async(init_obs=False)
        return obs, reward, done, info

    # ------------------------------
    # Public helpers
    # ------------------------------
    def get_system_prompt(self) -> str:
        """Generate the system prompt for the environment."""
        format_prompt_str = format_prompt(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=self.config.use_example_in_sys_prompt,
            prompt_format=self.config.prompt_format,
        )
        return system_prompt() + "\n" + format_prompt_str

    # ------------------------------
    # Internal helpers
    # ------------------------------
    def _get_player_position(self) -> Tuple[int, int]:
        """Get current player position as (row, col)."""
        return (self.gym_env.s // self.gym_env.ncol, self.gym_env.s % self.gym_env.ncol)

    def _apply_slip(self, action_int: int) -> int:
        """Apply custom slip probability to action.

        With probability slip_prob, the agent slips to a random perpendicular direction.
        """
        if self.config.slip_prob > 0 and np.random.random() < self.config.slip_prob:
            # Slip to a random perpendicular direction
            perpendicular = self.PERPENDICULAR_ACTIONS[action_int]
            return np.random.choice(perpendicular)
        return action_int

    async def _render_async(self, init_obs: bool) -> Dict[str, Any]:
        """
        Async wrapper of render to avoid blocking:
        - For vision mode, offloads env.render() to thread pool.
        - For text mode, uses current state to format grid text.
        """
        multi_modal_input: Optional[Dict[str, List[Image.Image]]] = None

        # Build format prompt (without example in obs)
        format_prompt_str = format_prompt(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False,
            prompt_format=self.config.prompt_format,
        )

        if self.config.render_mode == "vision":
            # Offload blocking render to a thread pool
            rgb_array = await asyncio.to_thread(self.gym_env.render)
            img_str = self.config.image_placeholder
            multi_modal_input = {
                self.config.image_placeholder: [numpy_to_pil(rgb_array)]
            }
        else:
            img_str = self._grid_to_text()

        if init_obs:
            obs_str = init_observation_template(img_str) + "\n"
        else:
            obs_str = action_template(self.valid_actions, img_str) + "\n"

        obs: Dict[str, Any] = {"obs_str": obs_str}
        if multi_modal_input is not None:
            obs["multi_modal_input"] = multi_modal_input
        return obs

    def _grid_to_text(self) -> str:
        """Convert current state to a human-readable text grid."""
        room_state = self._get_text_representation()
        text_rows = []
        for row in room_state:
            text_row = "".join(self.GRID_LOOKUP.get(int(cell), "?") for cell in row)
            text_rows.append(text_row)
        return "\n".join(text_rows)

    def _get_text_representation(self) -> np.ndarray:
        """Get the text representation of the current state."""
        room_state = copy.deepcopy(self.gym_env.desc)

        # Convert bytes to integer representation
        room_state = np.vectorize(lambda x: self.MAP_LOOKUP.get(x, 1))(room_state)

        # Get player position and update cell
        position_P = self._get_player_position()

        # Check what tile the player is on
        original_tile = self.gym_env.desc[position_P]

        if original_tile == b'H':
            room_state[position_P] = 4  # player fell into hole
        elif original_tile == b'G':
            room_state[position_P] = 5  # player on goal (victory)
        else:
            room_state[position_P] = 0  # normal player on frozen tile

        return room_state

    def _is_success(self) -> bool:
        """Check if player reached the goal."""
        player_pos = self._get_player_position()
        return self.gym_env.desc[player_pos] == b'G'

    def _finished(self) -> bool:
        """Check if episode is finished (player on goal or fell into hole)."""
        player_pos = self._get_player_position()
        return self.gym_env.desc[player_pos] in [b'G', b'H']


# ------------------------------
# Local async test (optional)
# ------------------------------
if __name__ == "__main__":
    import fire
    import os
    import logging

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )

    async def main_async(
        render_mode: str = "vision",
        size: int = 4,
        is_slippery: bool = False,
        max_actions_per_step: int = 4,
        save_path: str = "./test_frozenlake",
        prompt_format: str = "free_think",
    ):
        cfg = {
            "render_mode": render_mode,
            "size": size,
            "is_slippery": is_slippery,
            "max_actions_per_step": max_actions_per_step,
            "prompt_format": prompt_format,
        }
        env = FrozenLake(cfg)

        print("System Prompt:")
        sys_prompt = await env.system_prompt()
        print(sys_prompt["obs_str"])
        print("\n" + "=" * 50 + "\n")

        obs, info = await env.reset(seed=0)
        print("Initial Observation:")
        print(obs["obs_str"])
        step = 0
        os.makedirs(save_path, exist_ok=True)
        if "multi_modal_input" in obs:
            img = obs["multi_modal_input"][env.config.image_placeholder][0]
            img.save(os.path.join(save_path, f"step_{step}.png"))

        while True:
            step += 1
            print(f"\nStep {step}:")
            try:
                action_input = input("Enter action string (Left, Down, Right, Up) or 'quit': ")
            except EOFError:
                action_input = "quit"

            if action_input.lower() == "quit":
                break

            if not action_input.startswith("<think>"):
                action_input = f"<think>Moving towards the goal.</think><answer>{action_input}</answer>"

            obs, reward, done, info = await env.step(action_input)
            if "multi_modal_input" in obs:
                img = obs["multi_modal_input"][env.config.image_placeholder][0]
                img.save(os.path.join(save_path, f"step_{step}.png"))
            print(f"Reward: {reward}, Done: {done}")
            print(f"Observation:\n{obs['obs_str']}")
            if done:
                if info.get("success", False):
                    print("Goal reached!")
                else:
                    print("Fell into a hole!")
                break

        print(f"\nTotal reward: {env.total_reward}")
        await env.close()

    def main(**kwargs):
        asyncio.run(main_async(**kwargs))

    fire.Fire(main)