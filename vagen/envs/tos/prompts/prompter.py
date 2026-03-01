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
from .prompts import *
from .cogmap_prompts import BASE_COGMAP_PROMPT, COGMAP_INSTRUCTION_GLOBAL_ONLY
from ..utils.utils import THINK_LABEL, ANSWER_LABEL

COGMAP_TASK_INSTRUCTION = """\
## Cognitive Map Output Requirement

After you terminate exploration, output your global cognitive map in JSON format.
Do not output action sequences in this stage.

THINK:
[Your spatial reasoning]
FINAL ANSWER:
{{"agent": {{"position": [0, 0], "facing": "north"}}, "chair": {{"position": [0, 2], "facing": "south"}}, "green door": {{"position": [0, 5], "facing": "north"}}}}

Rules: integer positions only (never "unknown"); start=(0,0) north; include all observed objects.
Facing rule for final global cogmap: ONLY use "north|south|east|west" for every entry (including agent and objects).
Even if exploration used 8 headings, do NOT output diagonal facings (northeast/southeast/southwest/northwest); project to the nearest cardinal direction.

Schema and rules:
{schema}
"""

class PromptManager:
    @staticmethod
    def system_prompt() -> str:
        return  f"You answer should strictly follow this format:\n{THINK_LABEL} Your thoughts\n{ANSWER_LABEL} Your final answer"

    # Simple env message helpers
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
        # Decide answer hint
        if is_exploration:
            answer_hint = "Actions: [ ... ]"
        else:
            # Special stricter format for InternVL during evaluation
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
        schema = BASE_COGMAP_PROMPT + "\n" + COGMAP_INSTRUCTION_GLOBAL_ONLY
        prompt = COGMAP_TASK_INSTRUCTION.format(schema=schema)
        if self.enable_think:
            prompt += f"\n\nStrictly follow this format:\n{THINK_LABEL}\n[Your thoughts on the global cognitive map]\n{ANSWER_LABEL}\n[JSON cognitive map only]"
        else:
            prompt += f"\n\nStrictly follow this format:\n{ANSWER_LABEL}\n[JSON cognitive map only]"
        return prompt

    def __init__(self, config, np_random: np.random.RandomState, image_handler = None):
        self.config = config
        self.image_handler = image_handler
        self.np_random = np_random
        self.enable_think = bool(self.config.prompt_config.get('enable_think', True))

    def _is_internvl_model(self) -> bool:
        """Return True if current model is InternVL."""
        try:
            model_cfg = self.config.get_model_config()
            model_name = str((model_cfg or {}).get('model_name', '')).lower()
            return 'internvl' in model_name
        except Exception:
            return False

    def get_initial_observation_prompt(
            self,
            room: Room,
            agent: Agent,
            exp_history = None
        ) -> tuple:
        """
        Generates the initial observation prompt based on the exploration type.
        """
        obs = {}
        is_vision, is_active = self.config.render_mode == 'vision', self.config.exp_type == 'active'

        room_desc = get_room_description(room, agent)

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

        action_instructions = f"Action Instructions:\n{ActionSequence.get_usage_instructions(is_vision)}"
        exp_history_str = ""
        if not is_active:
            exp_history_str = f"## Exploration History\n{exp_history['obs_str']}"
        images_path = []
        if is_vision:
            images = [self.image_handler.get_image('instruction'), self.image_handler.get_image('label')]
            images_path = [self.image_handler.get_image_path('instruction'), self.image_handler.get_image_path('label')]
            if not is_active:
                images.extend(exp_history['multi_modal_data'][self.config.image_placeholder])
                images_path.extend(exp_history['multi_modal_data_paths'])
            obs['multi_modal_data'] = {self.config.image_placeholder: images}

        
        template = INSTRUCTION_TEMPLATE_VISION if is_vision else INSTRUCTION_TEMPLATE_TEXT

        fmt_kwargs = {
            'title': 'Spatial Exploration Task' if is_active else 'Spatial Reasoning Task',
            'intro': SHARED_INTRO_TEXT if not is_vision else SHARED_INTRO_VISION,
            'goal_lines': (
                'Goal: **Minimize total COST** while building a complete and accurate map of the environment.'
                if is_active else ''
            ),
            'observation_instructions': observation_instructions,
            'action_instructions': action_instructions,
            'room_info': room_desc,
            'steps_left': (
                f"\n\nYou have a maximum of {self.config.max_exp_steps} exploration steps."
                if is_active else ''
            ),
            'multiroom_rules': SHARED_MULTIROOM_RULES,
            'active_rules_extra': ACTIVE_RULES_EXTRA if is_active else '',
            'rules_common': SHARED_RULES_COMMON,
            'exp_history': exp_history_str,
            'vision_example': (VISION_EXAMPLE.format(image_placeholder=self.config.image_placeholder) if is_vision else ''),
        }

        obs_str = template.format(**fmt_kwargs)
        if is_active:
            obs_str = obs_str + "\n" + self.get_format_footer(is_active)

        obs['obs_str'] = obs_str
        return obs, images_path
