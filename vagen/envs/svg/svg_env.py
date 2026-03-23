"""
SVG environment for VAGEN.

The LLM sees a target image and generates SVG code to reproduce it.
Scoring (DINO/DreamSim) models are injected by the handler so they
are shared across all env instances on the same server.
"""

from __future__ import annotations

import asyncio
import re
import random
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from vagen.envs.gym_image_env import GymImageEnv
from .svg_config import SvgEnvConfig
from .svg_utils import process_and_rasterize_svg, is_valid_svg, load_svg_dataset
from .score import calculate_total_score
from .prompt import (
    system_prompt as svg_system_prompt,
    format_prompt,
    init_observation_template,
    action_template,
)

LOGGER = logging.getLogger(__name__)


def _parse_response(response: str, prompt_format: str, action_sep: str, max_actions: int) -> Dict:
    """Thin wrapper: use VAGEN-style parsers then extract SVG from the answer."""
    from vagen.envs.sokoban.utils.utils import parse_response
    return parse_response(
        response=response,
        prompt_format=prompt_format,
        action_sep=action_sep,
        max_actions=max_actions,
    )


def _extract_svg(text: str) -> str:
    """Extract <svg>...</svg> from arbitrary text."""
    m = re.search(r'<svg.*?</svg>', text, re.DOTALL)
    if m:
        return m.group(0)
    if '<svg' in text and '</svg>' in text:
        start = text.find('<svg')
        end = text.rfind('</svg>') + 6
        if start < end:
            return text[start:end]
    return ""


class SVGEnv(GymImageEnv):
    """Async SVG environment implementing the GymImageEnv interface."""

    def __init__(
        self,
        env_config: Dict[str, Any],
        dataset=None,
        dino_model=None,
        dreamsim_model=None,
    ):
        super().__init__(env_config)
        self.config = SvgEnvConfig(**env_config)

        # Shared scoring models (injected by handler)
        self._dino = dino_model
        self._dreamsim = dreamsim_model

        # Dataset (shared across envs, loaded once in handler)
        self.dataset = dataset

        # Episode state
        self.total_reward: float = 0.0
        self.reward: float = 0.0
        self.valid_actions: List[str] = []
        self.gt_svg_code: str = ""
        self.gt_image: Optional[Image.Image] = None
        self.gen_svg_code: Optional[str] = None
        self.gen_image: Optional[Image.Image] = None
        self.rng = random.Random(self.config.seed)

    # ------------------------------------------------------------------
    # GymImageEnv interface
    # ------------------------------------------------------------------

    async def close(self) -> None:
        pass  # nothing to release per-instance

    async def system_prompt(self) -> Dict[str, Any]:
        prompt_text = svg_system_prompt(prompt_format=self.config.prompt_format)
        fmt_text = format_prompt(
            prompt_format=self.config.prompt_format,
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=True,
        )
        return {"obs_str": prompt_text + "\n" + fmt_text}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.rng.seed(seed)

        idx = self.rng.randint(0, len(self.dataset) - 1)
        sample = self.dataset[idx]

        self.gt_svg_code = sample.get('Svg', sample.get('svg', ''))
        if not self.gt_svg_code:
            raise ValueError(f"No SVG code in sample at index {idx}")

        _, self.gt_image = await asyncio.to_thread(
            process_and_rasterize_svg, self.gt_svg_code
        )

        # Reset episode state
        self.total_reward = 0.0
        self.reward = 0.0
        self.gen_svg_code = None
        self.gen_image = None
        self.valid_actions = []

        obs = self._render(init_obs=True)
        return obs, {}

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        # Parse LLM response
        parsed = _parse_response(
            response=action_str,
            prompt_format=self.config.prompt_format,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step,
        )

        # Try to extract SVG code from parsed actions or raw text
        svg_code = ""
        if parsed['actions']:
            svg_code = _extract_svg(parsed['actions'][0])
        if not svg_code:
            svg_code = _extract_svg(action_str)

        if svg_code and is_valid_svg(svg_code):
            parsed['actions'] = [svg_code]
        else:
            parsed['actions'] = []

        metrics = {
            "turn_metrics": {
                "action_is_valid": len(parsed['actions']) > 0,
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            },
        }

        self.reward = 0.0
        self.valid_actions = []
        done = False
        info: Dict[str, Any] = {}
        info.update(parsed)

        if not parsed['actions']:
            # Invalid / no SVG
            self.reward += self.config.format_penalty
            done = True
            info["metrics"] = metrics
            info["success"] = False
            self.total_reward += self.reward
            self.gen_svg_code = None
            return self._render(init_obs=False), self.reward, done, info

        # Valid SVG
        if parsed.get("format_correct", True):
            self.reward += self.config.format_reward

        self.gen_svg_code = parsed['actions'][0]
        self.valid_actions = parsed['actions']

        try:
            _, gen_image = await asyncio.to_thread(
                process_and_rasterize_svg, self.gen_svg_code
            )
            self.gen_image = gen_image

            score_config = self.config.get_score_config()
            scores = await asyncio.to_thread(
                calculate_total_score,
                self.gt_image,
                gen_image,
                self.gt_svg_code,
                self.gen_svg_code,
                score_config,
                self._dino,
                self._dreamsim,
            )

            self.reward += scores["total_score"]
            info["scores"] = scores
            metrics["turn_metrics"]["action_is_effective"] = scores["total_score"] > 0

        except Exception as e:
            LOGGER.warning(f"Scoring failed: {e}")
            self.valid_actions = []
            metrics["turn_metrics"]["action_is_valid"] = False

        info["metrics"] = metrics
        info["success"] = metrics["traj_metrics"]["success"]
        self.total_reward += self.reward

        return self._render(init_obs=False), self.reward, done, info

    # ------------------------------------------------------------------
    # Render helper
    # ------------------------------------------------------------------

    def _render(self, init_obs: bool) -> Dict[str, Any]:
        if init_obs:
            img = self.gt_image
        elif self.gen_svg_code and self.gen_image is not None:
            img = self.gen_image
        else:
            img = Image.new('RGB', (256, 256), color='white')

        placeholder = self.config.image_placeholder
        multi_modal_input = {placeholder: [img]}

        fmt_text = format_prompt(
            prompt_format=self.config.prompt_format,
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False,
        )

        if init_obs:
            obs_str = init_observation_template(placeholder) + "\n" + fmt_text
        else:
            valid_action_str = self.valid_actions[0] if self.valid_actions else ""
            obs_str = action_template(
                valid_action=valid_action_str,
                observation=placeholder,
                reward=self.reward,
                done=False,
            ) + "\n" + fmt_text

        return {
            "obs_str": obs_str,
            "multi_modal_input": multi_modal_input,
        }
