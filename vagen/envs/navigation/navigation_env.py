"""AI2-THOR navigation environment (GymImageEnv async interface)."""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from vagen.envs.gym_image_env import GymImageEnv
from vagen.envs.navigation.utils.prompt import system_prompt, init_observation_template, action_template, get_format_instruction
from vagen.envs.navigation.utils.parse import parse_response, compute_reward


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NavigationEnvConfig:
    env_name: str = "navigation"
    resolution: int = 255
    eval_set: str = "base"
    down_sample_ratio: float = 1.0
    fov: int = 100
    multiview: bool = False
    max_actions_per_step: int = 5
    action_sep: str = "|"
    max_steps: int = 30
    # Reward config (ViewSuite-style)
    format_reward: float = 0.0          # end-of-episode bonus if ALL turns had correct format
    per_turn_format_reward: float = 0.01  # per-step bonus if this turn's format is correct
    success_reward: float = 1.0         # reaching the goal
    gpu_device: int = 0
    prompt_format: str = "free_think"   # free_think | wm | no_think | eval_mode
    example_count: int = 0             # number of examples in system prompt (0 = none)
    success_threshold: float = 1.0
    step_length: float = 0.3
    image_placeholder: str = "<image>"


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------

ACTION_LOOKUP = {
    "moveahead": 1, "moveback": 2, "moveright": 3, "moveleft": 4,
    "rotateright": 5, "rotateleft": 6, "lookup": 7, "lookdown": 8,
}

_ACTION_DISPATCH = {
    1: ("MoveAhead",  {"moveMagnitude": None}),
    2: ("MoveBack",   {"moveMagnitude": None}),
    3: ("MoveRight",  {"moveMagnitude": None}),
    4: ("MoveLeft",   {"moveMagnitude": None}),
    5: ("RotateRight", {"degrees": 90}),
    6: ("RotateLeft",  {"degrees": 90}),
    7: ("LookUp",      {"degrees": 30}),
    8: ("LookDown",    {"degrees": 30}),
}

VALID_EVAL_SETS = [
    "base", "common_sense", "complex_instruction", "visual_appearance", "long_horizon",
    "base_train", "common_sense_train",
]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class NavigationEnv(GymImageEnv):
    """AI2-THOR navigation env. Blocking calls run via asyncio.to_thread."""

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        valid = {f.name for f in fields(NavigationEnvConfig)}
        self.cfg = NavigationEnvConfig(**{k: v for k, v in env_config.items() if k in valid})
        assert self.cfg.eval_set in VALID_EVAL_SETS
        self._controller = None
        self._dataset = self._load_dataset()
        self._episode_data: Optional[Dict] = None
        self._instruction = ""
        self._step_count = 0
        self._t0 = 0.0
        self._valid_actions: List[str] = []
        self._reward = 0.0
        self._total_reward = 0.0
        self._is_format_correct = True  # Track across episode for end-of-episode format bonus
        self._info: Dict[str, Any] = {}

    def _load_dataset(self) -> List[Dict]:
        path = os.path.join(os.path.dirname(__file__), "datasets", f"{self.cfg.eval_set}.json")
        with open(path) as f:
            tasks = json.load(f)["tasks"]
        if 0 < self.cfg.down_sample_ratio < 1:
            tasks = tasks[::round(1 / self.cfg.down_sample_ratio)]
        return tasks

    def _ensure_controller(self):
        if self._controller is not None:
            return
        import ai2thor.controller
        from ai2thor.platform import CloudRendering
        self._controller = ai2thor.controller.Controller(
            agentMode="default", gridSize=0.1, visibilityDistance=10,
            renderDepthImage=False, renderInstanceSegmentation=False,
            width=self.cfg.resolution, height=self.cfg.resolution,
            fieldOfView=self.cfg.fov, platform=CloudRendering,
            gpu_device=self.cfg.gpu_device, server_timeout=300, server_start_timeout=300,
        )

    # --- helpers ---

    def _agent_pos(self):
        return self._controller.last_event.metadata["agent"]["position"]

    def _distance_to_target(self) -> float:
        a, t = self._agent_pos(), self._episode_data["target_position"]
        return math.sqrt((a["x"] - t["x"]) ** 2 + (a["z"] - t["z"]) ** 2)

    def _is_success(self) -> bool:
        return self._distance_to_target() <= self.cfg.success_threshold

    def _exec_action(self, idx: int):
        name, kwargs = _ACTION_DISPATCH[idx]
        kwargs = {k: (self.cfg.step_length if v is None else v) for k, v in kwargs.items()}
        self._controller.step(action=name, **kwargs)

    def _render_obs(self, init: bool) -> Dict[str, Any]:
        ph = self.cfg.image_placeholder
        img = Image.fromarray(self._controller.last_event.frame.astype(np.uint8))
        if init:
            fmt = get_format_instruction(self.cfg.prompt_format,
                                         self.cfg.max_actions_per_step, self.cfg.action_sep)
            obs_str = init_observation_template(observation=ph, instruction=self._instruction) + "\n" + fmt
        else:
            obs_str = action_template(
                valid_action=self._valid_actions, observation=ph,
                reward=self._reward, done=self._is_success(),
                env_feedback=self._info.get("env_feedback", ""),
            )
        return {"obs_str": obs_str, "multi_modal_input": {ph: [img]}}

    # --- sync methods (run in thread) ---

    def _sync_reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._ensure_controller()
        traj = self._dataset[seed % len(self._dataset)]
        self._episode_data = traj
        self._instruction = traj["instruction"]

        self._controller.reset(scene=traj["scene"])
        if self.cfg.multiview:
            ev = self._controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
            pose = {**ev.metadata["actionReturn"], "orthographic": True}
            self._controller.step(action="AddThirdPartyCamera", **pose, skyboxColor="white", raise_for_failure=True)

        ap = traj["agentPose"]
        self._controller.step(
            action="Teleport",
            position={k: ap["position"][k] for k in "xyz"},
            rotation={"x": 0, "y": ap["rotation"], "z": 0},
            horizon=ap["horizon"], standing=True,
        )

        self._step_count = 0
        self._t0 = time.time()
        self._valid_actions = []
        self._reward = 0.0
        self._total_reward = 0.0
        self._is_format_correct = True
        self._info = {}
        return self._render_obs(init=True), {}

    def _sync_step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        parsed = parse_response(
            action_str,
            prompt_format=self.cfg.prompt_format,
            action_sep=self.cfg.action_sep,
            max_actions=self.cfg.max_actions_per_step,
        )
        actions = parsed["actions"]
        prev_pos = self._agent_pos()
        self._reward = 0.0
        self._valid_actions = []
        done = False
        success = False
        info: Dict[str, Any] = {**parsed}

        # Track format correctness across episode
        if not parsed["format_correct"]:
            self._is_format_correct = False

        if actions and parsed["format_correct"]:
            for action in actions:
                if action not in ACTION_LOOKUP:
                    break
                self._exec_action(ACTION_LOOKUP[action])
                self._valid_actions.append(action)
                if self._is_success():
                    done = success = True
                    break
                self._step_count += 1
                if self._step_count >= self.cfg.max_steps:
                    done = True
                    break

        # Compute reward
        self._reward = compute_reward(
            parsed=parsed,
            valid_actions=self._valid_actions,
            success=success,
            format_reward=self.cfg.format_reward,
            per_turn_format_reward=self.cfg.per_turn_format_reward,
            success_reward=self.cfg.success_reward,
            is_format_correct_so_far=self._is_format_correct,
        )

        cur_pos = self._agent_pos()
        info.update({
            "metrics": {
                "turn_metrics": {
                    "action_is_valid": bool(self._valid_actions),
                    "action_is_effective": cur_pos["x"] != prev_pos["x"] or cur_pos["z"] != prev_pos["z"],
                    "format_correct": parsed["format_correct"],
                },
                "traj_metrics": {
                    "success": success,
                    "is_format_correct": self._is_format_correct,
                },
            },
            "distance": self._distance_to_target(),
            "instruction": self._instruction,
            "env_step": self._step_count,
            "episode_elapsed_seconds": time.time() - self._t0,
            "task_success": self._is_success(),
            "success": success,
            "last_action_success": self._controller.last_event.metadata["lastActionSuccess"],
        })
        info["env_feedback"] = (
            "Last action is executed successfully." if info["last_action_success"]
            else "Last action is not executed successfully."
        )
        self._info = info
        self._total_reward += self._reward
        return self._render_obs(init=False), self._reward, done, info

    def _sync_close(self):
        if self._controller is not None:
            self._controller.stop()
            self._controller = None

    # --- async interface ---

    async def system_prompt(self) -> Dict[str, Any]:
        return {"obs_str": system_prompt(
            format_name=self.cfg.prompt_format,
            max_actions_per_step=self.cfg.max_actions_per_step,
            action_sep=self.cfg.action_sep,
            example_count=self.cfg.example_count,
        )}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return await asyncio.to_thread(self._sync_reset, seed)

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        return await asyncio.to_thread(self._sync_step, action_str)

    async def close(self) -> None:
        await asyncio.to_thread(self._sync_close)


# ---------------------------------------------------------------------------
# Interactive test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import fire

    async def _run(
        eval_set: str = "base",
        prompt_format: str = "free_think",
        gpu_device: int = 0,
        seed: int = 0,
        save_path: str = "./test_navigation",
    ):
        cfg = {
            "eval_set": eval_set,
            "prompt_format": prompt_format,
            "gpu_device": gpu_device,
        }
        env = NavigationEnv(cfg)

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
        sep = env.cfg.action_sep
        actions = f" {sep} ".join(ACTION_LOOKUP.keys())
        print(f"\nValid actions: {actions}")
        print(f"Wrap in format or just type raw actions (e.g. 'moveahead{sep}rotateright')\n")

        while True:
            step += 1
            try:
                raw = input(f"Step {step}> ")
            except (EOFError, KeyboardInterrupt):
                break
            if raw.strip().lower() in ("quit", "q", ""):
                break
            # Auto-wrap if user didn't use tags
            if "<action>" not in raw:
                raw = f"<think>exploring</think><action>{raw}</action>"

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
