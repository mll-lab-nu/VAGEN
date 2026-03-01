import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple

from vagen.envs.gym_image_env import GymImageEnv
from .env_config import SpatialGymConfig
from .managers.exploration_manager import ExplorationManager
from .managers.cognitive_map_manager import CognitiveMapManager
from .actions.base import BaseAction
from .core.object import Agent
from .core.room import Room
from .managers.agent_proxy import get_agent_proxy
from .prompts import PromptManager
from .utils.action_utils import action_results_to_text
from .utils.room_utils import initialize_room_from_json
from .utils.utils import parse_llm_response
from .utils.image_handler import ImageHandler
from .actions.actions import ForcedTermAction, ActionSequence, configure_actions
from gymnasium.utils import seeding


class SpatialGym(GymImageEnv):
    """
    Spatial Gym Environment (exploration only).
    Adapts ToS to VAGEN's GymImageEnv interface.
    """
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        if isinstance(env_config, dict):
            try:
                valid_fields = SpatialGymConfig.__dataclass_fields__.keys()
                filtered_config = {k: v for k, v in env_config.items() if k in valid_fields}
                self.config = SpatialGymConfig(**filtered_config)
            except Exception as e:
                print(f"Warning: partial config init failed: {e}. Using default config.")
                self.config = SpatialGymConfig()
        else:
            self.config = env_config

        self.prompter: PromptManager = None
        self.action_classes = configure_actions('exploration')

        self.is_exploration_phase = None
        self.remaining_exp_steps = None
        self.render_cache = None
        self.current_turn_number = None
        self.observed_image_paths: List[str] = None
        self.awaiting_cogmap_output = None

        self.initial_room = None
        self.initial_agent = None
        self.exploration_manager = None

    def _generate_initial_observation(self) -> Tuple[Dict[str, Any], Any]:
        """Generate initial observation based on exploration type."""
        exp_history = {}
        images = []
        final_loc = None
        if self.config.exp_type == 'passive':
            proxy = get_agent_proxy(
                self.config.proxy_agent,
                self.initial_room,
                self.agent,
                grid_size=self.config.grid_size if hasattr(self.config, 'grid_size') else None,
            )
            proxy.run()
            if self.config.render_mode == 'vision':
                obs_str = proxy.to_text(self.config.image_placeholder)
                image_paths = []
                for t in proxy.turns:
                    if any('observe' in result.action_type for result in t.actions):
                        image, image_path = self._get_multi_modal_data(proxy.mgr, t.pos, t.ori)
                        images.append(image)
                        image_paths.append(image_path)
                assert images, "No images captured for vision render mode"
                exp_history['multi_modal_data'] = {self.config.image_placeholder: images}
                exp_history['multi_modal_data_paths'] = image_paths
            else:
                obs_str = proxy.to_text()
            exp_history['obs_str'] = obs_str
            self.exploration_manager = proxy.mgr
            final_loc = (list(proxy.turns[-1].pos), list(proxy.turns[-1].ori))

        obs_dict, self.observed_image_paths = self.prompter.get_initial_observation_prompt(
            room=self.initial_room,
            agent=self.agent,
            exp_history=exp_history,
        )

        obs = {'obs_str': obs_dict['obs_str']}
        mm_data = exp_history.get('multi_modal_data') or obs_dict.get('multi_modal_data') or {}
        if mm_data:
            obs['multi_modal_input'] = mm_data

        return obs, final_loc

    async def system_prompt(self) -> Dict[str, Any]:
        return {'obs_str': PromptManager.system_prompt()}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment for a new episode."""
        self.current_seed = seed
        self.np_random, seed = seeding.np_random(seed)

        self.action_classes = configure_actions('exploration')

        self.image_handler = ImageHandler(self.config.data_dir, seed, image_size=self.config.image_size)
        self.json_data = self.image_handler.json_data
        self.prompter = PromptManager(self.config, self.np_random, self.image_handler)

        self.initial_room, self.agent = initialize_room_from_json(self.json_data)
        self.initial_agent = self.agent.copy()

        self.remaining_exp_steps = self.config.max_exp_steps
        self.current_turn_number = 0
        self.observed_image_paths = []
        self.awaiting_cogmap_output = False
        self.is_exploration_phase = True

        BaseAction.set_field_of_view(self.config.field_of_view)
        self.exploration_manager = ExplorationManager(
            self.initial_room, self.agent,
            grid_size=(self.config.grid_size if hasattr(self.config, 'grid_size') else None),
            seed=seed,
        )

        info = {}
        if self.config.exp_type == 'passive':
            info['finish'] = True

        obs, _ = self._generate_initial_observation()
        self.render_cache = obs
        self.observed_image_paths = []
        return obs, info

    def _get_multi_modal_data(self, room: ExplorationManager, pos: np.ndarray, ori: np.ndarray):
        """Get image for current agent position and orientation."""
        assert self.config.render_mode == 'vision', "Cannot get multi-modal data in text mode"
        position_name = None if not np.allclose(room.init_pos, pos) else 'agent'
        if position_name is None:
            for obj in room.exploration_room.all_objects:
                if np.allclose(obj.pos, pos):
                    position_name = obj.name
                    break
        assert position_name is not None, f"Agent position not found for {pos}, sample id: {self.current_seed}"
        direction = BaseAction._ori_to_direction_label(ori)
        img = self.image_handler.get_image(position_name, direction)
        img_path = self.image_handler.get_image_path(position_name, direction)
        return img, img_path

    def _evaluate_cogmap(self, cogmap_str: str) -> float:
        """Score a cogmap JSON string against ground truth.
        Returns a score in [0, 1] (average of pos and facing accuracy).
        """
        try:
            mgr = CognitiveMapManager()
            turn_log = mgr.evaluate_cogmap_type(
                cogmap_str,
                self.exploration_manager.exploration_room,
                self.exploration_manager.agent,
                list(self.exploration_manager.observed_items),
                map_type="global",
            )
            if turn_log and turn_log.metrics.valid:
                pos = float(turn_log.metrics.pos) if turn_log.metrics.pos is not None else 0.0
                facing = float(turn_log.metrics.facing) if turn_log.metrics.facing is not None else 0.0
                score = (pos + facing) / 2.0
                return max(0.0, min(1.0, score))
        except Exception:
            pass
        return 0.0

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Process agent actions in the spatial gym environment."""
        self.current_turn_number += 1

        if self.awaiting_cogmap_output:
            _, cogmap_answer, _ = parse_llm_response(
                action_str, enable_think=bool(self.config.prompt_config.get('enable_think', True))
            )
            cogmap_str = cogmap_answer if cogmap_answer else action_str
            cogmap_score = self._evaluate_cogmap(cogmap_str)
            obs = {'obs_str': self.prompter.task_finished_message()}
            self.render_cache = obs
            self.awaiting_cogmap_output = False
            info = {'cogmap_score': cogmap_score, 'success': cogmap_score > 0.0}
            return obs, cogmap_score, True, info

        _, action, _ = parse_llm_response(
            action_str, enable_think=bool(self.config.prompt_config.get('enable_think', True))
        )

        obs, reward, done, info, exp_log = self._execute_action(action)

        self.observed_image_paths = []

        return obs, reward, done, info

    def _execute_action(self, action: str):
        """Execute action and return results."""
        obs_str, reward, done, info = "", -0.1, False, {'is_valid_action': True}
        obs: Dict[str, Any] = {}
        exp_log = None
        self.remaining_exp_steps -= 1

        if self.remaining_exp_steps < 0:
            action_sequence = ActionSequence(motion_actions=[], final_action=ForcedTermAction())
            is_valid = True
        else:
            action_sequence = ActionSequence.parse(action, action_classes=self.action_classes)
            is_valid = bool(action) and bool(action_sequence)

        if not is_valid:
            obs_str += self.prompter.invalid_action_message() + "\n"
            info["is_valid_action"] = False
            reward -= 0.5
        else:
            action_results = self.exploration_manager.execute_action_sequence(action_sequence)
            for res in action_results:
                if res.data and 'reported_changes' in res.data:
                    info['reported_changes'] = res.data['reported_changes']
            obs_str += action_results_to_text(
                action_results,
                self.config.image_placeholder if self.config.render_mode == 'vision' else None,
            )
            exp_log = self.exploration_manager.turn_logs[-1]
            if exp_log:
                # Clear large state snapshots to save memory (not needed for VAGEN rollout)
                exp_log.room_state = None
                exp_log.agent_state = None
            if action_sequence.final_action and action_sequence.final_action.is_term():
                self.awaiting_cogmap_output = True
                obs = {'obs_str': self.prompter.get_cogmap_output_prompt()}
            else:
                obs_str += "\n" + self.prompter.steps_left_message(self.remaining_exp_steps)
                if self.config.render_mode == 'vision':
                    image, image_path = self._get_multi_modal_data(
                        self.exploration_manager,
                        self.exploration_manager.agent.pos,
                        self.exploration_manager.agent.ori,
                    )
                    obs = {'multi_modal_input': {self.config.image_placeholder: [image]}, 'obs_str': obs_str}
                    self.observed_image_paths.append(image_path)

        if not obs:
            obs = {'obs_str': obs_str}

        if not done and not self.awaiting_cogmap_output:
            obs['obs_str'] += '\n' + self.prompter.get_format_footer(True)

        self.render_cache = obs
        return obs, reward, done, info, exp_log

    def render(self):
        return self.render_cache

    async def close(self):
        return

    def get_exp_summary(self):
        """Get exploration efficiency metrics."""
        return self.exploration_manager.get_exp_summary() if self.exploration_manager else ExplorationManager.DEFAULT_EXP_SUMMARY

    def _get_env_info(self):
        """Get environment state information."""
        return {
            "config": self.config.to_dict(),
            "initial_room": self.initial_room.to_dict(),
            "initial_agent": self.initial_agent.to_dict(),
        }
