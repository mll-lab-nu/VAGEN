import numpy as np
from PIL import Image

# from gym_sokoban.envs.sokoban_env import SokobanEnv
from vagen.envs.sokoban.patch_sokoban_env import PatchedSokobanEnv as SokobanEnv
from vagen.envs.sokoban.utils.prompt import (
    action_template,
    format_prompt,
    init_observation_template,
    system_prompt,
)
from .utils.utils import parse_response, numpy_to_pil


from vagen.envs import GymImageEnv
from vagen.envs import HasStateReward
from vagen.envs.sokoban.state_reward_spec import SPEC as SOKOBAN_STATE_REWARD_SPEC

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional


from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SokobanEnvConfig:
    """Configuration for Sokoban environment"""
    dim_room: Tuple[int, int] = (6, 6)  # Room dimensions (height, width)
    max_steps: int = 100      # Maximum steps per episode
    num_boxes: int = 1        # Number of boxes in the room
    render_mode: str = "text" # "text" or "vision"
    max_actions_per_step: int = 3  # Max actions per step
    action_sep: str = ","     # Separator between actions
    image_placeholder: str = "<image>"  # Placeholder for vision mode
    use_example_in_sys_prompt: bool = True  # Whether to add example system prompt

    # Map generation constraints
    min_solution_steps: Optional[Tuple[int, int]] = None  # (min, max) range for solution steps
    reset_seed_max_tries: int = 10000  # Max tries to find a valid seed
    min_solution_bfs_max_depth: int = 200  # Max BFS depth for solution
    prompt_format: str = "wm"  # "free_think" or "wm"
    format_reward: float = 0.1  # Reward for following the format correctly
    success_reward: float = 1.0
    # ★ Whether the world-modeling format is enforced. One switch, governing both halves,
    # so that "the environment before/after the 2026-08-10 fix" is a single knob rather
    # than two that can be set inconsistently:
    #
    #   True  (default) -- a response failing `format_correct` has its salvaged action
    #                      discarded AND earns no format reward.
    #   False           -- the pre-fix behaviour exactly: the salvaged action runs, and
    #                      the format reward is paid for any runnable action.
    #
    # Reproducing a pre-2026-08-10 sokoban run therefore needs `strict_format: false`
    # *and* `format_reward: 0.1` (the dataclass default those runs used, since the train
    # yaml did not override it until 95ad6da).
    #
    # The salvage half in detail: asking for the world-modeling format
    # and then executing a bare `<answer>` teaches the policy that the other three
    # sections are optional, which is exactly what happened -- sokoban rollouts collapsed
    # to `<answer>Left, Left</answer>`. frozenlake has always been strict this way.
    #
    # Set False to restore the salvage path. The argument for it is real and measured: on
    # the base model, half of all episodes reached zero usable actions when a malformed
    # turn was dropped, which discards half the batch over punctuation. The argument
    # against is that it makes the format unenforceable, since the format reward is the
    # only remaining pressure and it is worth 0.02 against a 1.0 success reward.
    strict_format: bool = True

    # State reward is intentionally not a dataclass field. The shared environment factory
    # removes config.state_reward before constructing this class, then wraps the instance
    # using STATE_REWARD_SPEC below. That keeps the generic switch out of every individual
    # environment dataclass while still locating it in envs[].config for train and eval.

class Sokoban(GymImageEnv, HasStateReward):
    """
    Sokoban environment that implements the EnvImageBase async interface.
    Uses asyncio.to_thread(...) to offload blocking gym calls (reset/step/render/close)
    to a thread pool so the event loop is not blocked.
    """

    # Text rendering lookup
    GRID_LOOKUP = {
        0: " # ",  # wall
        1: " _ ",  # floor
        2: " O ",  # target
        3: " √ ",  # box on target
        4: " X ",  # box
        5: " P ",  # player
        6: " S ",  # player on target
    }

    # What config.state_reward needs to score this environment's descriptions. Declared
    # here rather than in a table in the agent loop, so the capability is a property of
    # the environment and cannot disagree with the registry name.
    STATE_REWARD_SPEC = SOKOBAN_STATE_REWARD_SPEC

    # Action mapping
    ACTION_LOOKUP = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
    }

    def __init__(self, env_config: Dict[str, Any]):
        """
        :param env_config: a Dict with keys mapped to SokobanEnvConfig
        """
        super().__init__(env_config)
        self.config = SokobanEnvConfig(**env_config)
        # Create the underlying (blocking) gym env
        self.env = SokobanEnv(
            dim_room=self.config.dim_room,
            max_steps=self.config.max_steps,
            num_boxes=self.config.num_boxes,
        )
        self.total_reward: float = 0.0
        self.valid_actions: List[str] = []

    # ------------------------------
    # EnvImageBase abstract methods
    # ------------------------------
    async def close(self) -> None:
        """Non-blocking close via to_thread to avoid blocking the loop."""
        await asyncio.to_thread(self.env.close)

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Non-blocking reset:
        - Offloads env.reset() to a thread pool to avoid blocking the event loop.
        - Uses seed for deterministic reset
        - Generates map according to min_solution_steps requirement
        """
        # If seeding is needed, set it before reset in to_thread, or call a seeded reset API.
        
        await asyncio.to_thread(self.env.reset, seed=seed,
                                min_solution_steps=self.config.min_solution_steps,
                                reset_seed_max_tries=self.config.reset_seed_max_tries,
                                min_solution_bfs_max_depth=self.config.min_solution_bfs_max_depth)
        self.total_reward = 0.0
        self.valid_actions = []
        obs = await self._render_async(init_obs=True)
        info: Dict[str, Any] = {}
        return obs, info

    async def system_prompt(self) -> Dict[str, Any]:
        """
        Non-blocking system prompt:
        - Offloads system prompt to a thread pool to avoid blocking the event loop.
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
        if getattr(self.config, "strict_format", True) and not parsed.get("format_correct", False):
            # The turn still happens and still costs a step; it simply does nothing. That
            # is the point -- a malformed turn has to be worse than a well-formed one, and
            # under `wm` "malformed" means the observation/think/prediction sections the
            # prompt asked for are missing, not merely that the action was unparseable.
            action_list = []
        # Copy current player position (read-only)
        prev_player_pos = np.array(self.env.player_position, copy=True)

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
                # Offload the blocking gym step to a thread
                _obs, step_reward, step_done, _ = await asyncio.to_thread(self.env.step, action_int)
                # reward += float(step_reward) # ignore sokoban reward
                self.valid_actions.append(action)
                # Early success check
                if self._is_success():
                    done = True
                    reward += self.config.success_reward
                    metrics["traj_metrics"]["success"] = True
                    break
            else:
                metrics["turn_metrics"]["action_is_valid"] = False
                break

        # ★ The format reward is for the FORMAT, and must be gated on `format_correct` --
        # not on having produced a runnable action. This paid out for any response the
        # forgiving extractor could salvage an action from, which is the one thing it must
        # not do: `parse_wm` is deliberately two-tier, strict for `format_correct` and
        # lenient for extraction, so that a malformed turn still keeps its data. Paying the
        # bonus on the lenient test collapses the two tiers and removes every incentive to
        # write the tags at all.
        #
        # Under `prompt_format=wm` the policy is asked for
        # <observation>/<think>/<answer>/<prediction>; writing them costs ~100 tokens and,
        # with this bug, earned exactly what emitting `<answer>Left, Left</answer>` alone
        # earned. Sokoban rollouts duly collapsed to that. frozenlake, navigation and
        # primitive_skill all gate on `format_correct`; sokoban was the only one that did
        # not, and it computes the flag ~25 lines above for `action_is_valid`.
        if self.valid_actions and (
            parsed.get("format_correct", False) or not getattr(self.config, "strict_format", True)
        ):
            reward += self.config.format_reward

        # Effective action: detect player position change
        metrics["turn_metrics"]["action_is_effective"] = not np.array_equal(
            prev_player_pos, self.env.player_position
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
        """Keep original system prompt composition."""
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
    async def _render_async(self, init_obs: bool) -> Dict[str, Any]:
        """
        Async wrapper of render to avoid blocking:
        - For vision mode, offloads env.render(mode="rgb_array") to thread pool.
        - For text mode, uses current room_state to format grid text.
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
            rgb_array = await asyncio.to_thread(self.env.render, "rgb_array")
            img_str = self.config.image_placeholder
            multi_modal_input = {
                self.config.image_placeholder: [numpy_to_pil(rgb_array)]
            }
        else:
            img_str = self._grid_to_text()

        if init_obs:
            obs_str = init_observation_template(img_str) + "\n" #+ format_prompt_str
        else:
            obs_str = action_template(self.valid_actions, img_str) + "\n" #+ format_prompt_str

        obs: Dict[str, Any] = {"obs_str": obs_str}
        if multi_modal_input is not None:
            obs["multi_modal_input"] = multi_modal_input
        return obs

    def _grid_to_text(self) -> str:
        """Convert current room_state to a human-readable text grid."""
        room_state = np.where(
            (self.env.room_state == 5) & (self.env.room_fixed == 2),
            6,
            self.env.room_state,
        )
        text_rows = []
        for row in room_state:
            text_row = "".join(self.GRID_LOOKUP.get(int(cell), "?") for cell in row)
            text_rows.append(text_row)
        return "\n".join(text_rows)

    def _is_success(self) -> bool:
        """Check if all boxes are on targets."""
        return self.env.boxes_on_target == self.env.num_boxes


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

    async def main_async(render_mode: str = "vision",
                         num_boxes: int = 1,
                         dim_room: Tuple[int, int] = (6, 6),
                         max_actions_per_step: int = 2,
                         save_path: str = "./test",
                         min_solution_steps: Tuple[int, int] = (1, 5),
                         reset_seed_max_tries: int = 10000,
                         min_solution_bfs_max_depth: int = 100
                        ):
        cfg = {
            "render_mode": render_mode,
            "num_boxes": num_boxes,
            "dim_room": dim_room,
            "max_actions_per_step": max_actions_per_step,
            "min_solution_steps": min_solution_steps,
            "reset_seed_max_tries": reset_seed_max_tries,
            "min_solution_bfs_max_depth": min_solution_bfs_max_depth,
            "prompt_format": "free_think",
            
        }
        env = Sokoban(cfg)

        print("System Prompt:")
        print(env.get_system_prompt())
        print("\n" + "=" * 50 + "\n")

        obs, info = await env.reset(seed=0)
        print("Initial Observation:")
        print(obs["obs_str"])
        step = 0
        os.makedirs(save_path, exist_ok=True)
        if "multi_modal_input" in obs:
            # save the image to target folder
            img = obs["multi_modal_input"][env.config.image_placeholder][0]
            img.save(os.path.join(save_path, f"step_{step}.png"))

        while True:
            step += 1
            print(f"\nStep {step}:")
            try:
                action_input = input("Enter action string (or 'quit'): ")
            except EOFError:
                action_input = "quit"

            if action_input.lower() == "quit":
                break

            if not action_input.startswith("<think>"):
                action_input = f"<think>Moving towards the goal.</think><answer>{action_input}</answer>"

            obs, reward, done, info = await env.step(action_input)
            if "multi_modal_input" in obs:
                # save the image to target folder
                img = obs["multi_modal_input"][env.config.image_placeholder][0]
                img.save(os.path.join(save_path, f"step_{step}.png"))
            print(f"Reward: {reward}, Done: {done}")
            print(f"Observation:\n{obs['obs_str']}")
            if done:
                print("Puzzle solved!")
                break

        print(f"\nTotal reward: {env.total_reward}")
        await env.close()

    def main(**kwargs):
        asyncio.run(main_async(
            **kwargs
        ))

    fire.Fire(main)
