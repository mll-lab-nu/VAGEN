"""
EB-ALFRED environment adapter for VAGEN.

Wraps the EBAlfEnv from EmbodiedBench as a GymImageEnv,
enabling integration with VAGEN's RL training and evaluation pipeline.

The underlying EBAlfEnv uses AI2-THOR for 3D household robot task simulation.
It requires a GPU-accelerated X server for rendering.
"""

import asyncio
import json
import os
import signal
import time
import threading
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, List, Optional

from .utils.prompt import (
    system_prompt,
    format_prompt,
    init_observation_template,
    action_template,
)
from .utils.utils import parse_response, match_action, numpy_to_pil

from vagen.envs.gym_image_env import GymImageEnv

# Thread-local storage for passing x_display to ThorConnector without
# touching the process-global X_DISPLAY module variable.
# Each worker thread sets its own _tl.x_display before creating EBAlfEnv;
# the monkey-patched ThorConnector.__init__ reads it instead of the global.
_tl = threading.local()
_patched = False
_patch_lock = threading.Lock()


def _ensure_thor_patched():
    """One-time monkey-patch: make ThorConnector read display from thread-local."""
    global _patched
    if _patched:
        return
    with _patch_lock:
        if _patched:
            return
        from embodiedbench.envs.eb_alfred.thor_connector import ThorConnector

        _orig_init = ThorConnector.__init__

        def _patched_init(self, x_display=None, **kwargs):
            # Prefer thread-local display (set by EbAlfred.__init__)
            tl_display = getattr(_tl, "x_display", None)
            if tl_display is not None:
                x_display = tl_display
            _orig_init(self, x_display=x_display, **kwargs)

        ThorConnector.__init__ = _patched_init
        _patched = True


@dataclass
class EbAlfredEnvConfig:
    """Configuration for EB-ALFRED environment."""

    # Environment settings
    eval_set: str = "base"
    exp_name: str = "vagen_eval"
    down_sample_ratio: float = 1.0
    resolution: int = 500
    x_display: str = "1"
    selected_indexes: List[int] = field(default_factory=list)
    detection_box: bool = False

    # Interaction settings
    max_turns: int = 30
    max_actions_per_step: int = 20
    max_env_steps: int = 30  # Max total environment actions per episode (matches ERA)
    action_sep: str = "|"
    image_placeholder: str = "<image>"
    prompt_format: str = "free_think"
    use_example_in_sys_prompt: bool = True

    # Observation image settings
    obs_image_size: Optional[int] = None  # Resize obs image to this size (square). None = use original.

    # Reward settings
    format_reward: float = 0.1
    success_reward: float = 1.0


class EbAlfred(GymImageEnv):
    """
    EB-ALFRED environment implementing the GymImageEnv async interface.

    Wraps EBAlfEnv from EmbodiedBench, which uses AI2-THOR for
    3D household robot task simulation (e.g., "Clean a rag, put it away").

    Key features:
    - 162+ discrete actions (find, pick up, put down, open, close, etc.)
    - Dynamic action space per episode (multi-instance objects)
    - Vision-only observations (RGB images from AI2-THOR)
    - Dense reward via task progress + format reward
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)

        # Filter config keys to only those in the dataclass
        valid_keys = EbAlfredEnvConfig.__dataclass_fields__
        filtered = {k: v for k, v in env_config.items() if k in valid_keys}
        self.config = EbAlfredEnvConfig(**filtered)

        # Patch ThorConnector to read x_display from thread-local storage
        # instead of the process-global X_DISPLAY.  This allows fully
        # parallel env creation across GPUs with no locks.
        _ensure_thor_patched()
        from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv

        _tl.x_display = self.config.x_display
        try:
            self.env = EBAlfEnv(
                eval_set=self.config.eval_set,
                exp_name=self.config.exp_name,
                down_sample_ratio=self.config.down_sample_ratio,
                selected_indexes=self.config.selected_indexes,
                detection_box=self.config.detection_box,
                resolution=self.config.resolution,
            )
        finally:
            _tl.x_display = None

        # Replace single-split dataset with ALL splits so that a pooled
        # env can serve any eval_set on reset without recreating Unity.
        # Original EBAlfEnv loads only one eval_set into self.dataset (list).
        # We reload the full splits.json into a dict {eval_set: [episodes]}.
        with open(self.env.data_path) as f:
            all_splits = json.load(f)
        ds_ratio = self.config.down_sample_ratio
        if 0 < ds_ratio < 1:
            every = round(1 / ds_ratio)
            all_splits = {k: v[::every] for k, v in all_splits.items()}
        self.env.dataset = all_splits
        self.env._default_eval_set = self.config.eval_set

        # Adapter state (reset per episode)
        self._total_turns: int = 0
        self._total_env_steps: int = 0
        self._last_action: str = ""
        self._last_feedback: str = ""
        self._last_thinking: str = ""
        self._last_action_id: Optional[int] = None
        self._action_list: List[str] = []
        self._action_map: Dict[str, str] = {}  # lowercase -> original

    # ------------------------------------------------------------------
    # GymImageEnv abstract methods
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close AI2-THOR process.

        Applies a 30-second timeout so that a hung Unity process or
        WSGI server shutdown does not block the event loop forever.
        """
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self.env.close), timeout=30.0
            )
        except asyncio.TimeoutError:
            # Force-kill the Unity process if graceful shutdown hangs
            pid = getattr(self.env.env, "unity_pid", None)
            if pid:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

    async def system_prompt(self) -> Dict[str, Any]:
        """
        Return the system prompt with per-episode task and action list.

        Includes role description, action descriptions, guidelines,
        the current task instruction, available actions, and format
        instructions. This ensures no-concat mode always has access
        to task and action information.
        """
        sys_str = system_prompt(
            task_instruction=self.env.episode_language_instruction,
            action_list=self._action_list,
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            max_turns=self.config.max_turns,
        )
        fmt_str = format_prompt(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=self.config.use_example_in_sys_prompt,
            prompt_format=self.config.prompt_format,
        )
        return {"obs_str": sys_str + "\n" + fmt_str}

    def _reset_sync(self, eval_set: str, episode_idx: int):
        """Synchronous reset that drives the underlying EBAlfEnv.

        We call ``_reset_controller`` directly (instead of ``env.reset()``)
        because the upstream EBAlfEnv.reset() only supports sequential
        iteration over a single eval_set.  This wrapper manages episode
        selection externally via (eval_set, episode_idx).
        """
        task = self.env.dataset[eval_set][episode_idx]
        self.env._reset_controller(task)
        self.env._current_step = 0
        self.env._cur_invalid_actions = 0
        self.env._reset = True
        self.env.episode_log = []
        self.env._episode_start_time = time.time()

    async def reset(self, seed: int, eval_set: str = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset environment for a new episode.

        The seed selects which episode to load from the dataset
        (seed % number_of_episodes_in_eval_set). The eval_set can be
        overridden per-reset so a single pooled env can serve any split.
        """
        es = eval_set or self.config.eval_set
        n_episodes = len(self.env.dataset.get(es, []))
        episode_idx = seed % n_episodes

        await asyncio.wait_for(
            asyncio.to_thread(self._reset_sync, es, episode_idx),
            timeout=300.0,
        )

        # Reset adapter state
        self._total_turns = 0
        self._total_env_steps = 0
        self._last_action = ""
        self._last_feedback = ""
        self._last_thinking = ""
        self._last_action_id = None

        # Build action lookup for this episode (action space is dynamic)
        self._action_list = list(self.env.language_skill_set)
        self._action_map = {a.lower(): a for a in self._action_list}

        # Build observation
        obs = self._build_obs(init=True)
        info = {
            "task_instruction": self.env.episode_language_instruction,
            "num_actions": len(self._action_list),
            "eval_set": es,
            "episode_idx": episode_idx,
        }
        return obs, info

    async def step(
        self, action_str: str
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step given the LLM's response.

        Parses <think>...</think><answer>...</answer> from action_str,
        matches the action against the current action space, and
        executes it in AI2-THOR.
        """
        self._total_turns += 1

        # Parse LLM response
        parsed = parse_response(
            response=action_str,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step,
            prompt_format=self.config.prompt_format,
        )

        reward = 0.0
        done = False
        info = dict(parsed)

        actions = parsed.get("actions", [])
        format_correct = parsed.get("format_correct", False)
        self._last_thinking = parsed.get("think_content", "")

        metrics = {
            "turn_metrics": {
                "action_is_valid": False,
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            },
        }

        if format_correct and actions:
            reward += self.config.format_reward

            # Clip actions to remaining env step budget (ERA-style)
            remaining = self.config.max_env_steps - self._total_env_steps
            actions = actions[:remaining] if remaining > 0 else []

            for action_name in actions:
                matched = match_action(action_name, self._action_list, self._action_map)

                if matched is None:
                    # Action name not recognized
                    self._last_action = action_name
                    self._last_feedback = (
                        f"Action '{action_name}' is not a recognized action."
                    )
                    break

                metrics["turn_metrics"]["action_is_valid"] = True

                # Execute in AI2-THOR
                self._total_env_steps += 1
                obs_raw, step_reward, step_done, step_info = (
                    await asyncio.wait_for(
                        asyncio.to_thread(self.env.step, matched), timeout=60.0
                    )
                )

                self._last_action = matched
                self._last_action_id = self._action_list.index(matched) if matched in self._action_list else None
                self._last_feedback = step_info.get("env_feedback", "")

                action_success = step_info.get("last_action_success", 0.0)
                if action_success:
                    metrics["turn_metrics"]["action_is_effective"] = True

                task_success = step_info.get("task_success", 0.0)
                if task_success:
                    done = True
                    reward += self.config.success_reward
                    metrics["traj_metrics"]["success"] = True
                    break

                if step_done:
                    done = True
                    break

                # ERA-style: break on action failure to replan
                if not action_success:
                    break

                # Check env step limit
                if self._total_env_steps >= self.config.max_env_steps:
                    done = True
                    break
        else:
            # Format error: no valid actions parsed
            self._last_action = parsed.get("action_content", "")
            self._last_feedback = (
                "Could not parse a valid action from your response. "
                "Please use the format: <think>...</think><answer>action name</answer>"
            )

        # Check turn limit and env step limit
        if self._total_turns >= self.config.max_turns:
            done = True
        if self._total_env_steps >= self.config.max_env_steps:
            done = True

        info["metrics"] = metrics
        info["success"] = metrics["traj_metrics"]["success"]

        obs = self._build_obs(init=False)
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self, init: bool) -> Dict[str, Any]:
        """Build observation dict with image and text."""
        frame = self.env.env.last_event.frame
        img = numpy_to_pil(frame)
        if self.config.obs_image_size is not None:
            sz = self.config.obs_image_size
            img = img.resize((sz, sz), Image.LANCZOS)
        img_str = self.config.image_placeholder

        if init:
            obs_str = init_observation_template(
                img_str=img_str,
                task_instruction=self.env.episode_language_instruction,
            )
        else:
            obs_str = action_template(
                last_action=self._last_action,
                env_feedback=self._last_feedback,
                img_str=img_str,
                task_instruction=self.env.episode_language_instruction,
                step_id=self._total_turns - 1,
                thinking=self._last_thinking,
                action_id=self._last_action_id,
            )

        return {
            "obs_str": obs_str + "\n",
            "multi_modal_input": {
                self.config.image_placeholder: [img]
            },
        }


# ------------------------------
# Local async test (optional)
# ------------------------------
if __name__ == "__main__":
    import fire
    import logging

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    async def main_async(
        eval_set: str = "base",
        resolution: int = 500,
        x_display: str = "1",
        save_path: str = "./test_eb_alfred",
        prompt_format: str = "free_think",
    ):
        cfg = {
            "eval_set": eval_set,
            "resolution": resolution,
            "x_display": x_display,
            "prompt_format": prompt_format,
        }
        env = EbAlfred(cfg)

        print("System Prompt:")
        sys_prompt = await env.system_prompt()
        print(sys_prompt["obs_str"])
        print("\n" + "=" * 50 + "\n")

        obs, info = await env.reset(seed=0)
        print(f"Task: {info['task_instruction']}")
        print(f"Available actions: {info['num_actions']}")
        print(f"Observation:\n{obs['obs_str'][:200]}...")

        step = 0
        os.makedirs(save_path, exist_ok=True)
        if "multi_modal_input" in obs:
            img = obs["multi_modal_input"][env.config.image_placeholder][0]
            img.save(os.path.join(save_path, f"step_{step}.png"))

        while True:
            step += 1
            print(f"\nStep {step}:")
            try:
                action_input = input("Enter action (or 'quit'): ")
            except EOFError:
                action_input = "quit"

            if action_input.lower() == "quit":
                break

            if not action_input.startswith("<think>"):
                action_input = (
                    f"<think>Executing the action.</think>"
                    f"<answer>{action_input}</answer>"
                )

            obs, reward, done, info = await env.step(action_input)
            if "multi_modal_input" in obs:
                img = obs["multi_modal_input"][env.config.image_placeholder][0]
                img.save(os.path.join(save_path, f"step_{step}.png"))
            print(f"Reward: {reward}, Done: {done}")
            print(f"Success: {info.get('success', False)}")
            print(f"Observation:\n{obs['obs_str'][:200]}...")

            if done:
                print("Episode finished!")
                break

        await env.close()

    def main(**kwargs):
        asyncio.run(main_async(**kwargs))

    fire.Fire(main)
