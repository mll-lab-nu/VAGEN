"""
ManiSkill primitive-skill environment (GymImageEnv async interface).

The LLM controls a Franka Emika robot arm via pick/place/push commands.
Wraps ManiSkill environments with a skill abstraction layer.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import time
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from vagen.envs.gym_image_env import GymImageEnv
from vagen.envs.primitive_skill.utils.prompt import (
    system_prompt,
    init_observation_template,
    action_template,
    get_format_prompt,
    VALID_FORMATS,
)
from vagen.envs.primitive_skill.utils.parse import parse_response, compute_reward

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PrimitiveSkillEnvConfig:
    env_name: str = "primitive_skill"
    env_id: str = "AlignTwoCube"       # AlignTwoCube, PlaceTwoCube, PutAppleInDrawer, StackThreeCube
    render_mode: str = "vision"        # vision | text
    max_actions_per_step: int = 2
    action_sep: str = "|"
    max_steps: int = 10
    record_video: bool = False
    video_record_dir: str = "./test"
    mask_success: bool = True
    prompt_format: str = "wm"
    # Reward config
    format_reward: float = 0.1
    success_reward: float = 10.0
    # Process reward for grounding / world modeling
    use_state_reward: bool = False
    grounding_reward_weight: float = 0.5
    worldmodeling_reward_weight: float = 0.5
    # GPU device (set by handler)
    gpu_device: int = 0
    image_placeholder: str = "<image>"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PrimitiveSkillEnv(GymImageEnv):
    """Async ManiSkill environment implementing the GymImageEnv interface."""

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        valid = {f.name for f in fields(PrimitiveSkillEnvConfig)}
        self.cfg = PrimitiveSkillEnvConfig(**{k: v for k, v in env_config.items() if k in valid})
        assert self.cfg.prompt_format in VALID_FORMATS, (
            f"Unknown prompt_format: {self.cfg.prompt_format}. Valid: {VALID_FORMATS}"
        )

        self._env = None  # lazy init
        self._state_keys: List[str] = []
        self._last_info: Dict[str, Any] = {}
        self._initial_reward: float = 0.0
        self._total_reward: float = 0.0
        self._step_count: int = 0
        self._t0: float = 0.0

    # ------------------------------------------------------------------
    # Lazy env creation (must happen in thread, uses GPU)
    # ------------------------------------------------------------------

    def _ensure_env(self):
        if self._env is not None:
            return
        # Lazy import: ManiSkill dependencies (gymnasium, mani_skill, etc.)
        # are only available on the GPU server
        from vagen.envs.primitive_skill.maniskill.utils import build_env
        import vagen.envs.primitive_skill.maniskill.env  # noqa: F401
        record_dir = self.cfg.video_record_dir if self.cfg.record_video else None
        self._env = build_env(self.cfg.env_id, record_dir=record_dir)
        self._state_keys = self._env.state_keys

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _compute_stage_reward(self) -> float:
        """Calculate reward based on highest completed stage."""
        if self._last_info.get("success", False):
            return 10.0
        max_stage = -1
        for key in self._last_info.keys():
            if key.startswith("stage_") and key.endswith("_success"):
                try:
                    stage_num = int(key.split("_")[1])
                    if self._last_info[key]:
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

        from vagen.envs.primitive_skill.maniskill.utils import get_workspace_limits
        action_array = np.zeros(9)
        workspace_x, workspace_y, workspace_z = get_workspace_limits(self._env)

        try:
            action_name = action_str.split("(")[0].strip().lower()

            if action_name == "pick":
                action_array[0] = 1
            elif action_name == "place":
                action_array[1] = 1
            elif action_name == "push":
                action_array[2] = 1
            else:
                return None

            params_str = action_str.split("(")[1].split(")")[0]
            params = [float(p.strip()) for p in params_str.split(",")]

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

    def _render_obs(self, init: bool, valid_actions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Render observation dict with text + optional image."""
        from vagen.envs.primitive_skill.maniskill.utils import handle_info, get_workspace_limits
        info = copy.copy(self._last_info)
        new_info = handle_info(
            info, state_keys=self._state_keys,
            mask_success=self.cfg.mask_success, env=self._env,
        )
        object_positions = str(list(new_info["obj_positions"].values()))
        other_information = str(new_info["other_info"])
        instruction = self._env.instruction()
        x_ws, y_ws, z_ws = get_workspace_limits(self._env)
        ph = self.cfg.image_placeholder

        fmt_text = get_format_prompt(
            format_name=self.cfg.prompt_format,
            max_actions_per_step=self.cfg.max_actions_per_step,
            action_sep=self.cfg.action_sep,
            state_keys=self._state_keys,
            add_example=False,
        )

        if init:
            obs_str = init_observation_template(
                observation=ph, instruction=instruction,
                x_workspace=x_ws, y_workspace=y_ws, z_workspace=z_ws,
                object_positions=object_positions,
                other_information=other_information,
            ) + "\n" + fmt_text
        else:
            obs_str = action_template(
                valid_actions=valid_actions or [],
                observation=ph, instruction=instruction,
                x_workspace=x_ws, y_workspace=y_ws, z_workspace=z_ws,
                object_positions=object_positions,
                other_information=other_information,
            ) + "\n" + fmt_text

        obs: Dict[str, Any] = {"obs_str": obs_str}

        if self.cfg.render_mode == "vision":
            rgb = self._env.render()
            img = _numpy_to_pil(rgb)
            obs["multi_modal_input"] = {ph: [img]}

        return obs

    # ------------------------------------------------------------------
    # Sync methods (run in thread)
    # ------------------------------------------------------------------

    def _sync_reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._ensure_env()
        _, info = self._env.reset(seed=seed)
        self._last_info = info
        self._initial_reward = self._compute_stage_reward()
        self._total_reward = 0.0
        self._step_count = 0
        self._t0 = time.time()
        return self._render_obs(init=True), {}

    def _sync_step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        parsed = parse_response(
            action_str,
            prompt_format=self.cfg.prompt_format,
            action_sep=self.cfg.action_sep,
            max_actions=self.cfg.max_actions_per_step,
        )

        valid_actions: List[str] = []
        terminated, truncated = False, False
        info: Dict[str, Any] = {**parsed}

        metrics = {
            "turn_metrics": {
                "action_is_valid": False,
                "action_is_effective": False,
                "format_correct": parsed["format_correct"],
            },
            "traj_metrics": {
                "success": False,
            },
        }

        prev_reward = self._compute_stage_reward()

        # Execute each action
        for action in parsed["actions"]:
            parsed_action = self._parse_action(action)
            if parsed_action is not None:
                _, _, terminated, truncated, step_info = self._env.step(parsed_action)
                valid_actions.append(action)
                self._last_info = step_info
                self._step_count += 1
            else:
                break
            if terminated or truncated:
                break

        cur_reward = self._compute_stage_reward()
        stage_reward = cur_reward - prev_reward

        # Metrics
        metrics["turn_metrics"]["action_is_valid"] = (
            len(valid_actions) > 0 and len(valid_actions) == len(parsed["actions"])
        )
        metrics["turn_metrics"]["action_is_effective"] = len(valid_actions) > 0

        success = bool(self._last_info.get("is_success", False))
        if success:
            metrics["traj_metrics"]["success"] = True

        done = bool(terminated) or bool(truncated)
        if isinstance(done, np.bool_):
            done = bool(done)

        # Compute reward
        reward = compute_reward(
            parsed=parsed,
            valid_actions=valid_actions,
            success=success,
            stage_reward=stage_reward,
            format_reward=self.cfg.format_reward,
            success_reward=self.cfg.success_reward,
        )

        info["metrics"] = metrics
        info["success"] = success
        info["env_step"] = self._step_count
        info["episode_elapsed_seconds"] = time.time() - self._t0
        self._total_reward += reward

        obs = self._render_obs(init=False, valid_actions=valid_actions)
        return obs, reward, done, info

    def _sync_close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    # ------------------------------------------------------------------
    # Async interface (GymImageEnv)
    # ------------------------------------------------------------------

    async def system_prompt(self) -> Dict[str, Any]:
        return {"obs_str": system_prompt(
            format_name=self.cfg.prompt_format,
            max_actions_per_step=self.cfg.max_actions_per_step,
            action_sep=self.cfg.action_sep,
            state_keys=self._state_keys,
            add_example=True,
        )}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return await asyncio.to_thread(self._sync_reset, seed)

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        return await asyncio.to_thread(self._sync_step, action_str)

    async def close(self) -> None:
        await asyncio.to_thread(self._sync_close)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numpy_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return Image.fromarray(arr.astype(np.uint8), mode="RGBA").convert("RGB")
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Interactive test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import fire

    async def _run(
        env_id: str = "AlignTwoCube",
        prompt_format: str = "wm",
        render_mode: str = "vision",
        seed: int = 0,
        save_path: str = "./test_primitive_skill",
    ):
        cfg = {
            "env_id": env_id,
            "prompt_format": prompt_format,
            "render_mode": render_mode,
        }
        env = PrimitiveSkillEnv(cfg)

        print("=== System Prompt ===")
        print((await env.system_prompt())["obs_str"])

        obs, _ = await env.reset(seed)
        os.makedirs(save_path, exist_ok=True)
        step = 0

        def _save_and_show(obs, step):
            print(f"\n--- Observation (step {step}) ---")
            print(obs["obs_str"][:500] + ("..." if len(obs["obs_str"]) > 500 else ""))
            if "multi_modal_input" in obs:
                path = os.path.join(save_path, f"step_{step}.png")
                obs["multi_modal_input"]["<image>"][0].save(path)
                print(f"  [image saved: {path}]")

        _save_and_show(obs, step)

        while True:
            step += 1
            try:
                raw = input(f"Step {step}> ")
            except (EOFError, KeyboardInterrupt):
                break
            if raw.strip().lower() in ("quit", "q", ""):
                break
            # Auto-wrap if user didn't use tags
            if "<answer>" not in raw:
                raw = f"<think>exploring</think><answer>{raw}</answer>"

            obs, reward, done, info = await env.step(raw)
            print(f"  reward={reward:.2f}  done={done}  success={info.get('success')}")
            _save_and_show(obs, step)
            if done:
                print("\n*** Episode finished! ***" +
                      (" Goal reached!" if info.get("success") else " Failed."))
                break

        print(f"\nTotal reward: {env._total_reward:.2f}")
        await env.close()

    fire.Fire(lambda **kw: asyncio.run(_run(**kw)))
