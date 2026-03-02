import numpy as np
from typing import Optional
from ..core.room import Room
from ..core.object import Agent
from ..actions.actions import ActionSequence

from ..utils.room_utils import get_room_description
from ..core.relationship import (
    PairwiseRelationship,
    PairwiseRelationshipDiscrete,
    ProximityRelationship,
    DegreeRel, OrientationRel
)
from .prompts import (
    SHARED_INTRO_TEXT, SHARED_INTRO_VISION,
    SHARED_MULTIROOM_RULES, SHARED_RULES_COMMON, ACTIVE_RULES_EXTRA,
    VISION_EXAMPLE,
)
from .cogmap_prompts import (
    BASE_COGMAP_PROMPT, COGMAP_INSTRUCTION_GLOBAL_ONLY,
)
from ..utils.utils import THINK_LABEL, ANSWER_LABEL


# ---------------------------------------------------------------------------
# System-level prompt template (assembled once per config, not per sample)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEXT = """\
# Spatial {task_title}

{intro}

{goal_section}## Environment Rules

{multiroom_rules}
{extra_rules}{common_rules}
## Observation Format

{observation_instructions}

## Available Actions

{action_instructions}

## Cognitive Map Output

After you call `Term()`, you will be asked to output a global cognitive map as JSON.

{cogmap_schema}
## Response Format

{format_instructions}"""

_SYSTEM_PROMPT_VISION = """\
# Spatial {task_title}

{intro}

{goal_section}## Environment Rules

{multiroom_rules}
{extra_rules}{common_rules}
## Observation

{observation_instructions}

## Available Actions

{action_instructions}

## Cognitive Map Output

After you call `Term()`, you will be asked to output a global cognitive map as JSON.

{cogmap_schema}
## Response Format

{format_instructions}"""


class PromptManager:
    def __init__(self, config, np_random: np.random.RandomState = None, image_handler=None):
        self.config = config
        self.image_handler = image_handler
        self.np_random = np_random
        self.enable_think = bool(self.config.prompt_config.get('enable_think', True))

    # ------------------------------------------------------------------
    # System prompt — contains ALL static instructions (no sample data)
    # ------------------------------------------------------------------

    def system_prompt(self) -> str:
        """Return the full static system prompt for this config."""
        is_vision = self.config.render_mode == 'vision'
        is_active = self.config.exp_type == 'active'

        intro = SHARED_INTRO_VISION if is_vision else SHARED_INTRO_TEXT

        goal_section = (
            "**Goal**: Minimize total COST while building a complete and accurate map of the environment.\n\n"
            if is_active else ""
        )

        extra_rules = ACTIVE_RULES_EXTRA if is_active else ""

        if is_vision:
            observation_instructions = (
                "Use the rendered image as the primary observation signal. "
                "Do not assume hidden objects; only use what has been observed."
            )
        else:
            observation_instructions = (
                PairwiseRelationship.prompt()
                + f"\n{DegreeRel.prompt()}"
                + f"\n{OrientationRel.prompt()}"
                + f"\n{PairwiseRelationshipDiscrete.prompt()}"
                + f"\n{ProximityRelationship.prompt()}"
            )

        action_instructions = ActionSequence.get_usage_instructions(is_vision)

        cogmap_schema = BASE_COGMAP_PROMPT + "\n" + COGMAP_INSTRUCTION_GLOBAL_ONLY

        if self.enable_think:
            format_instructions = (
                f"Always output:\n"
                f"{THINK_LABEL}\n[Your reasoning]\n"
                f"{ANSWER_LABEL}\n[Your answer]\n\n"
                "During exploration: `FINAL ANSWER` must be `Actions: [ ... ]`.\n"
                "During cognitive map output: `FINAL ANSWER` must be the JSON map only."
            )
        else:
            format_instructions = (
                f"Always output:\n"
                f"{ANSWER_LABEL}\n[Your answer]\n\n"
                "During exploration: `FINAL ANSWER` must be `Actions: [ ... ]`.\n"
                "During cognitive map output: `FINAL ANSWER` must be the JSON map only."
            )

        template = _SYSTEM_PROMPT_VISION if is_vision else _SYSTEM_PROMPT_TEXT
        return template.format(
            task_title="Exploration Task" if is_active else "Reasoning Task",
            intro=intro,
            goal_section=goal_section,
            multiroom_rules=SHARED_MULTIROOM_RULES,
            extra_rules=extra_rules,
            common_rules=SHARED_RULES_COMMON,
            observation_instructions=observation_instructions,
            action_instructions=action_instructions,
            cogmap_schema=cogmap_schema,
            format_instructions=format_instructions,
        )

    # ------------------------------------------------------------------
    # Initial observation — only sample-specific content
    # ------------------------------------------------------------------

    def get_initial_observation_prompt(
            self,
            room: Room,
            agent: Agent,
            exp_history=None,
    ) -> tuple:
        """Return the initial user message with only sample-specific info."""
        obs = {}
        is_vision = self.config.render_mode == 'vision'
        is_active = self.config.exp_type == 'active'

        room_desc = get_room_description(room, agent)

        images_path = []
        if is_vision:
            images = [self.image_handler.get_image('instruction'), self.image_handler.get_image('label')]
            images_path = [
                self.image_handler.get_image_path('instruction'),
                self.image_handler.get_image_path('label'),
            ]
            if not is_active:
                images.extend(exp_history['multi_modal_data'][self.config.image_placeholder])
                images_path.extend(exp_history['multi_modal_data_paths'])
            obs['multi_modal_data'] = {self.config.image_placeholder: images}

        lines = ["## Room Layout and Initial State", room_desc]

        if is_active:
            lines.append(f"\nYou have a maximum of {self.config.max_exp_steps} exploration steps.")

        if not is_active and exp_history:
            lines.append(f"\n## Exploration History\n{exp_history['obs_str']}")

        if is_vision and not is_active:
            lines.append(
                "\n" + VISION_EXAMPLE.format(image_placeholder=self.config.image_placeholder)
            )

        obs_str = "\n".join(lines)

        if is_active:
            obs_str = obs_str + "\n\n" + self.get_format_footer(True)

        obs['obs_str'] = obs_str
        return obs, images_path

    # ------------------------------------------------------------------
    # Per-step helpers
    # ------------------------------------------------------------------

    def invalid_action_message(self) -> str:
        return (
            "Invalid action. Each sequence must end with one final action: "
            "Observe() or Term(). "
            "E.g. [Rotate(90), Observe()] or [JumpTo(obj), Observe()]."
        )

    def steps_left_message(self, remaining_steps: int) -> str:
        base = f"You have a maximum of {remaining_steps} exploration steps left."
        if remaining_steps <= 1:
            base += (
                f"\n⚠️ DEADLINE: Only {remaining_steps} step(s) left! "
                "You MUST call Term() Now."
            )
        elif remaining_steps <= 5:
            base += (
                f"\n⚠️ DEADLINE: Only {remaining_steps} step(s) left! "
                "You MUST call Term() soon or exploration ends."
            )
        return base

    def task_finished_message(self) -> str:
        return "Task finished"

    def get_format_footer(self, is_exploration: bool) -> str:
        if is_exploration:
            answer_hint = "Actions: [ ... ]"
        else:
            if self._is_internvl_model():
                answer_hint = "[ONLY the letter (A, B, C, ...)]"
            else:
                answer_hint = "[your answer (only required answer, no extra text, notes, formatting or anything else)]"

        cogmap_hint = "  (or Actions: [Term()] to finish)" if is_exploration else ""
        if self.enable_think:
            think = "[Your thoughts on next step actions]" if is_exploration else "[Your thoughts on the question]"
            return f"Strictly follow this format:\n{THINK_LABEL}\n{think}\n{ANSWER_LABEL}\n{answer_hint}{cogmap_hint}"
        else:
            return f"Strictly follow this format:\n{ANSWER_LABEL}\n{answer_hint}{cogmap_hint}"

    def get_cogmap_output_prompt(self) -> str:
        """Short trigger message — full schema is already in the system prompt."""
        prompt = (
            "Exploration complete. Now output your **global cognitive map** as JSON.\n"
            "Rules: integer positions only (never \"unknown\"); start=(0,0) north; include all observed objects.\n"
            "Facing rule: ONLY use \"north|south|east|west\" for every entry (including agent). "
            "Do NOT output diagonal facings (northeast/southeast/southwest/northwest); "
            "project to the nearest cardinal direction.\n"
        )
        if self.enable_think:
            prompt += (
                f"\nStrictly follow this format:\n{THINK_LABEL}\n"
                "[Your thoughts on the global cognitive map]\n"
                f"{ANSWER_LABEL}\n[JSON cognitive map only]"
            )
        else:
            prompt += f"\nStrictly follow this format:\n{ANSWER_LABEL}\n[JSON cognitive map only]"
        return prompt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_internvl_model(self) -> bool:
        try:
            model_cfg = self.config.get_model_config()
            model_name = str((model_cfg or {}).get('model_name', '')).lower()
            return 'internvl' in model_name
        except Exception:
            return False
