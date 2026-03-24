"""
ManiSkill primitive-skill environment for VAGEN.

The LLM controls a Franka Emika robot arm via pick/place/push commands.
Wraps ManiSkill environments with a skill abstraction layer.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import numpy as np
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from vagen.envs.gym_image_env import GymImageEnv
from .maniskill_config import ManiSkillEnvConfig
from .maniskill.utils import build_env, handle_info, get_workspace_limits
from .prompt import (
    system_prompt as maniskill_system_prompt,
    format_prompt,
    init_observation_template,
    action_template,
)

LOGGER = logging.getLogger(__name__)


def _parse_response(response: str, prompt_format: str, action_sep: str, max_actions: int) -> Dict:
    """Use VAGEN-style parsers."""
    from vagen.envs.sokoban.utils.utils import parse_response
    return parse_response(
        response=response,
        prompt_format=prompt_format,
        action_sep=action_sep,
        max_actions=max_actions,
    )


def _numpy_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.shape[-1] == 3:
        return Image.fromarray(arr.astype(np.uint8), mode="RGB")
    elif arr.shape[-1] == 4:
        return Image.fromarray(arr.astype(np.uint8), mode="RGBA").convert("RGB")
    raise ValueError(f"Unsupported channels: {arr.shape[-1]}")


class ManiSkillEnv(GymImageEnv):
    """Async ManiSkill environment implementing the GymImageEnv interface."""

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.config = ManiSkillEnvConfig(**env_config)

        # Build the underlying ManiSkill env (blocking, done once)
        record_dir = self.config.video_record_dir if self.config.record_video else None
        self.env = build_env(self.config.env_id, record_dir=record_dir)

        self.state_keys = self.env.state_keys
        self.last_info: Dict[str, Any] = {}
        self.total_reward: float = 0.0
        self.initial_reward: float = 0.0
        self.steps: int = 0

    # ------------------------------------------------------------------
    # GymImageEnv interface
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await asyncio.to_thread(self.env.close)

    async def system_prompt(self) -> Dict[str, Any]:
        prompt_text = maniskill_system_prompt()
        fmt_text = format_prompt(
            prompt_format=self.config.prompt_format,
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=True,
        )
        return {"obs_str": prompt_text + "\n" + fmt_text}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        _, info = await asyncio.to_thread(self.env.reset, seed=seed)
        self.last_info = info
        self.initial_reward = self._compute_reward()
        self.total_reward = 0.0
        self.steps = 0
        obs = await self._render_async(init_obs=True)
        return obs, {}

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        parsed = _parse_response(
            response=action_str,
            prompt_format=self.config.prompt_format,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step,
        )

        reward = 0.0
        info: Dict[str, Any] = {}
        info.update(parsed)
        valid_actions: List[str] = []
        terminated, truncated = False, False

        metrics = {
            "turn_metrics": {
                "action_is_valid": False,
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            },
        }

        # Execute each action
        for action in parsed['actions']:
            parsed_action = self._parse_action(action)
            if parsed_action is not None:
                _, _, terminated, truncated, step_info = await asyncio.to_thread(
                    self.env.step, parsed_action
                )
                valid_actions.append(action)
                self.last_info = step_info
                self.steps += 1
            else:
                break
            if terminated or truncated:
                break

        # Metrics
        metrics["turn_metrics"]["action_is_valid"] = (
            len(valid_actions) > 0 and len(valid_actions) == len(parsed['actions'])
        )
        metrics["turn_metrics"]["action_is_effective"] = len(valid_actions) > 0

        if metrics["turn_metrics"]["action_is_valid"] and parsed.get("format_correct", False):
            reward += self.config.format_reward

        if self.last_info.get('is_success', False):
            metrics["traj_metrics"]["success"] = True
            reward += self.config.success_reward

        done = bool(terminated) or bool(truncated)
        if isinstance(done, np.bool_):
            done = bool(done)

        info["metrics"] = metrics
        info["success"] = metrics["traj_metrics"]["success"]
        self.total_reward += reward

        obs = await self._render_async(init_obs=False, valid_actions=valid_actions)
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self) -> float:
        if self.last_info.get("success", False):
            return 10.0
        max_stage = -1
        for key in self.last_info.keys():
            if key.startswith("stage_") and key.endswith("_success"):
                try:
                    stage_num = int(key.split("_")[1])
                    if self.last_info[key]:
                        max_stage = max(max_stage, stage_num)
                except (ValueError, IndexError):
                    continue
        return (max_stage + 1) * 2.0

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    def _parse_action(self, action_str: str) -> Optional[np.ndarray]:
        """Parse 'pick(x,y,z)', 'place(x,y,z)', 'push(x1,y1,z1,x2,y2,z2)' to action array."""
        if not action_str:
            return None

        action_array = np.zeros(9)
        workspace_x, workspace_y, workspace_z = get_workspace_limits(self.env)

        try:
            action_name = action_str.split('(')[0].strip().lower()

            if action_name == "pick":
                action_array[0] = 1
            elif action_name == "place":
                action_array[1] = 1
            elif action_name == "push":
                action_array[2] = 1
            else:
                return None

            params_str = action_str.split('(')[1].split(')')[0]
            params = [float(p.strip()) for p in params_str.split(',')]

            if action_name in ["pick", "place"] and len(params) != 3:
                return None
            elif action_name == "push" and len(params) != 6:
                return None

            # Clip to workspace
            params[0] = np.clip(params[0], workspace_x[0], workspace_x[1])
            params[1] = np.clip(params[1], workspace_y[0], workspace_y[1])
            params[2] = np.clip(params[2], workspace_z[0], workspace_z[1])

            if action_name == "push":
                params[3] = np.clip(params[3], workspace_x[0], workspace_x[1])
                params[4] = np.clip(params[4], workspace_y[0], workspace_y[1])
                params[5] = np.clip(params[5], workspace_z[0], workspace_z[1])

            for i in range(len(params)):
                action_array[i + 3] = params[i] / 1000.0

            return action_array

        except (IndexError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    async def _render_async(self, init_obs: bool, valid_actions=None) -> Dict[str, Any]:
        info = copy.copy(self.last_info)
        new_info = handle_info(
            info, state_keys=self.state_keys,
            mask_success=self.config.mask_success, env=self.env,
        )
        object_positions = str(list(new_info['obj_positions'].values()))
        other_information = str(new_info['other_info'])
        instruction = self.env.instruction()
        x_ws, y_ws, z_ws = get_workspace_limits(self.env)

        fmt_text = format_prompt(
            prompt_format=self.config.prompt_format,
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False,
        )

        if init_obs:
            obs_str = init_observation_template(
                observation=self.config.image_placeholder,
                instruction=instruction,
                x_workspace=x_ws, y_workspace=y_ws, z_workspace=z_ws,
                object_positions=object_positions,
                other_information=other_information,
            ) + "\n" + fmt_text
        else:
            obs_str = action_template(
                valid_actions=valid_actions or [],
                observation=self.config.image_placeholder,
                instruction=instruction,
                x_workspace=x_ws, y_workspace=y_ws, z_workspace=z_ws,
                object_positions=object_positions,
                other_information=other_information,
            ) + "\n" + fmt_text

        multi_modal_input = None
        if self.config.render_mode == "vision":
            rgb = await asyncio.to_thread(self.env.render)
            multi_modal_input = {
                self.config.image_placeholder: [_numpy_to_pil(rgb)]
            }

        obs: Dict[str, Any] = {"obs_str": obs_str}
        if multi_modal_input is not None:
            obs["multi_modal_input"] = multi_modal_input
        return obs
