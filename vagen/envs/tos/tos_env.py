import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from vagen.envs.gym_image_env import GymImageEnv
from .env_config import SpatialGymConfig
from .managers.exploration_manager import ExplorationManager
from .managers.cognitive_map_manager import CognitiveMapManager
from .actions.base import BaseAction
from .managers.agent_proxy import get_agent_proxy
from .prompts import PromptManager
from .utils.room_utils import initialize_room_from_json
from .utils.utils import parse_llm_response, execute_exploration_action, get_agent_view
from .utils.image_handler import ImageHandler
from .actions.actions import configure_actions
from gymnasium.utils import seeding

# Reward constants
COGMAP_REWARD_SCALE = 10.0   # cogmap scores in [0,1] scaled to [0,10]
FORCED_TERM_PENALTY = 2.0    # penalty for not calling Term() explicitly


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

        self.prompter: PromptManager = PromptManager(self.config)
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
        self.forced_term_occurred = False

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
                        image, image_path = get_agent_view(
                            proxy.mgr, t.pos, t.ori, self.image_handler, seed=self.current_seed
                        )
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
        return {'obs_str': self.prompter.system_prompt()}

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
        self.forced_term_occurred = False

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

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Process agent actions in the spatial gym environment."""
        self.current_turn_number += 1

        if self.awaiting_cogmap_output:
            _, cogmap_answer, _ = parse_llm_response(
                action_str, enable_think=bool(self.config.prompt_config.get('enable_think', True))
            )
            cogmap_str = cogmap_answer if cogmap_answer else action_str
            cogmap_scores = CognitiveMapManager.score_global_cogmap(
                cogmap_str,
                self.exploration_manager.exploration_room,
                self.exploration_manager.agent,
                list(self.exploration_manager.observed_items),
            )
            cogmap_score = cogmap_scores['overall']
            reward = cogmap_score * COGMAP_REWARD_SCALE
            if self.forced_term_occurred:
                reward -= FORCED_TERM_PENALTY
            obs = {'obs_str': self.prompter.task_finished_message()}
            self.render_cache = obs
            self.awaiting_cogmap_output = False
            info = {
                'cogmap_score': cogmap_score,
                'cogmap_dir': cogmap_scores['dir'],
                'cogmap_facing': cogmap_scores['facing'],
                'cogmap_pos': cogmap_scores['pos'],
                'success': cogmap_score > 0.0,
                'forced_term': self.forced_term_occurred,
            }
            return obs, reward, True, info

        _, action, _ = parse_llm_response(
            action_str, enable_think=bool(self.config.prompt_config.get('enable_think', True))
        )

        obs, reward, done, info, exp_log, self.remaining_exp_steps, self.awaiting_cogmap_output, image_path = (
            execute_exploration_action(
                action,
                self.exploration_manager,
                self.action_classes,
                self.remaining_exp_steps,
                self.config,
                self.prompter,
                image_handler=self.image_handler if self.config.render_mode == 'vision' else None,
                seed=self.current_seed,
            )
        )

        if info.get('forced_term'):
            self.forced_term_occurred = True

        if image_path:
            self.observed_image_paths.append(image_path)
        else:
            self.observed_image_paths = []

        self.render_cache = obs
        return obs, reward, done, info

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
