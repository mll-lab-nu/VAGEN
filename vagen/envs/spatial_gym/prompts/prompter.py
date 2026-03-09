import numpy as np
from typing import Optional
from ..core.room import Room
from ..core.object import Agent
from ..actions.actions import ActionSequence

from ..utils.room_utils import get_room_description
from .cogmap_prompts import GLOBAL_COGMAP_PROMPT
from ..utils.utils import THINK_LABEL, ANSWER_LABEL


_VISION_EXAMPLE = """\
Here is an example of your observation: blue cylinder 1 m straight ahead; red cylinder 2 m straight ahead; yellow cylinder 2 m at 45° to your front-left; green cylinder 3 m at 22.5° to your front-slight-right:
{image_placeholder}

The image shows all objects in the room. Each tile is numbered (1-N) in the top-left, matching the object order in the room layout.
For items with a facing direction, two copies are shown side-by-side: the left copy has its front facing the camera; the right copy has its front facing left.
Items without a meaningful facing direction are shown once.
{image_placeholder}
"""

# ---------------------------------------------------------------------------
# System-level prompt template (assembled once per config, not per sample)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
# Spatial {task_title}

You are a spatial reasoner in a 3D simulated environment. The world is rendered in 3D but abstracted into a discrete 2D grid of size N×M. Every entity, including yourself, is represented by integer coordinates (x, y) on this grid.

{goal_section}## Environment Rules

Multi-room rules (may exist multiple rooms):
- Your vision is confined to your current room.
- Doors block vision between rooms.
- Exception: When located in a doorway, door is open and invisible, you can see into both connected rooms.
- Rooms connect via doors on vertical (front/back) or horizontal (left/right) walls.
{extra_rules}- FOV is 90°. Track your position and facing after every Rotate() or JumpTo().
- Agent facing uses 8 headings: north/northeast/east/southeast/south/southwest/west/northwest.
- World axes are unchanged: +y=north, +x=east.
- Object facing (for non-agent objects) is still only one of north/east/south/west.

## Observation

Use the rendered image as the primary observation signal. Do not assume hidden objects; only use what has been observed.

## Available Actions

{action_instructions}

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
        is_active = self.config.exp_type == 'active'

        goal_section = (
            "**Goal**: Minimize total COST while building a complete and accurate map of the environment.\n"
            "- Exploration efficiency: maximize new information per step. Avoid revisiting observed areas.\n\n"
            if is_active else ""
        )

        action_instructions = ActionSequence.get_usage_instructions(True)

        if self.enable_think:
            format_instructions = (
                f"Always output:\n"
                f"{THINK_LABEL}\n[Your reasoning]\n"
                f"{ANSWER_LABEL}\n[Your answer]\n\n"
                "`FINAL ANSWER` must be `Actions: [ ... ]`.\n\n"
                "**Keep your response brief and concise. Avoid unnecessary verbosity.**"
            )
        else:
            format_instructions = (
                f"Always output:\n"
                f"{ANSWER_LABEL}\n[Your answer]\n\n"
                "`FINAL ANSWER` must be `Actions: [ ... ]`.\n\n"
                "**Keep your response brief and concise. Avoid unnecessary verbosity.**"
            )

        return _SYSTEM_PROMPT.format(
            task_title="Exploration Task" if is_active else "Reasoning Task",
            goal_section=goal_section,
            extra_rules=(
                "- Explore: jump to doors early (doorway sees both rooms), then cover each room systematically.\n"
                "- Coordinates: start=(0,0) north; +y=north, +x=east.\n"
            ) if is_active else "",
            action_instructions=action_instructions,
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
        is_active = self.config.exp_type == 'active'

        room_desc = get_room_description(room, agent)

        images = [self.image_handler.get_image('instruction'), self.image_handler.get_image('label')]
        images_path = [
            self.image_handler.get_image_path('instruction'),
            self.image_handler.get_image_path('label'),
        ]
        if not is_active:
            images.extend(exp_history['multi_modal_data'][self.config.image_placeholder])
            images_path.extend(exp_history['multi_modal_data_paths'])
        obs['multi_modal_data'] = {self.config.image_placeholder: images}

        lines = ["## Room Layout and Initial State", room_desc,
                 "Note: the rooms listed above are ALL rooms in this environment; do not assume additional rooms exist."]

        if is_active:
            lines.append(f"\nYou have a maximum of {self.config.max_exp_steps} exploration steps.")
            lines.append(
                "\n" + _VISION_EXAMPLE.format(image_placeholder=self.config.image_placeholder)
            )

        if not is_active and exp_history:
            lines.append(f"\n## Exploration History\n{exp_history['obs_str']}")
            lines.append(
                "\n" + _VISION_EXAMPLE.format(image_placeholder=self.config.image_placeholder)
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
        """Full cogmap prompt including schema, given to the agent after Term()."""
        cogmap_schema = GLOBAL_COGMAP_PROMPT
        prompt = (
            "Exploration complete. Now output your **global cognitive map** as JSON.\n"
            "Rules: integer positions only (never \"unknown\"); start=(0,0) north; include all observed objects.\n"
            "Facing rule: ONLY use \"north|south|east|west\" for every entry (including agent). "
            "Do NOT output diagonal facings (northeast/southeast/southwest/northwest); "
            "project to the nearest cardinal direction.\n\n"
            + cogmap_schema
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
