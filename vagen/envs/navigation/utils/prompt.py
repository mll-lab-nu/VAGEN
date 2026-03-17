"""
Prompt templates for the AI2-THOR navigation environment.

Structure:
  - system_prompt(): shared base prompt (role, actions, hints) + format instruction + optional examples
  - init_observation_template(): first observation (includes instruction)
  - action_template(): subsequent observations (no instruction, no format instruction)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Format instructions — the ONLY part that differs per format
# ---------------------------------------------------------------------------

_FORMAT_INSTRUCTIONS = {
    "free_think": (
        "You need to think first, then give your action. Respond in this format:\n"
        "<think>...</think><action>{action_example}</action>"
    ),
    "wm": (
        "You need to describe your observation, think, give your action, then predict "
        "what you will see next. Respond in this format:\n"
        "<observation>...</observation><think>...</think>"
        "<action>{action_example}</action><prediction>...</prediction>"
    ),
    "no_think": (
        "You need to only give your action. Respond in this format:\n"
        "<action>{action_example}</action>"
    ),
    "eval_mode": (
        "You can optionally think first, then give your action. Respond in this format:\n"
        "<think>...</think><action>{action_example}</action>"
    ),
}


def get_format_instruction(
    format_name: str,
    max_actions_per_step: int = 5,
    action_sep: str = "|",
) -> str:
    """Return the format-specific instruction string."""
    if format_name not in _FORMAT_INSTRUCTIONS:
        raise ValueError(f"Unknown format {format_name!r}. Available: {sorted(_FORMAT_INSTRUCTIONS)}")
    action_example = f"action1{action_sep} action2{action_sep} ..."
    return (
        f"You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'.\n"
        + _FORMAT_INSTRUCTIONS[format_name].format(action_example=action_example)
    )


# ---------------------------------------------------------------------------
# Shared system prompt
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = """\
You are a home robot and perform navigation tasks according to instructions.
Actions you can take: moveahead, moveback, moveright, moveleft, rotateright, rotateleft, lookup, lookdown.
moveahead: Move forward by some distance
moveback: Move backward by some distance
moveright: Move rightward by some distance
moveleft: Move leftward by some distance
rotateright: Rotate to the right by 90 degrees
rotateleft: Rotate to the left by 90 degrees
lookup: Tilt the camera upward by 30 degrees
lookdown: Tilt the camera downward by 30 degrees
The instruction will be provided in the first observation. Look at the image carefully and navigate to complete the instruction.
Hints:
1. You can take multiple actions at a time, in most cases, if you find the target object is far away from you, you can call moveahead, moveleft and move right multiple times.
2. If you find yourself seems to be stuck, you can lookdown to see if there's any object above or below you, you can also rotate to see if there's any object behind you."""

_EXAMPLE_TEMPLATE = """\
Example:
Round 1:
image_1
I can see the garbage can in the upper left corner of the image, next to the kitchen sink. \
To move there, we can go forward-left, but since there's a kitchen counter directly ahead, \
we should go left first.
<action>moveleft{sep} moveleft</action>
Round 2:
Env_feedback: Last action is executed successfully.
image_2
By moving leftward, we are getting closer to the garbage can. \
Now, the garbage can is in front of me, slightly to the left. There's a large area ahead.
<action>moveahead{sep} moveahead{sep} moveahead{sep} moveleft</action>
Round 3:
Env_feedback: Last action is executed successfully.
image_3
The garbage can is very close, still to our front-left. \
There is still space in front of me to get closer.
<action>moveahead{sep} moveahead{sep} moveleft</action>
Round 4:
Env_feedback: Success"""


def system_prompt(
    format_name: str = "free_think",
    max_actions_per_step: int = 5,
    action_sep: str = "|",
    example_count: int = 1,
) -> str:
    """Build the full system prompt: base + format instruction + optional examples.

    Args:
        example_count: number of examples to include. 0 = no examples.
    """
    parts = [_BASE_SYSTEM_PROMPT]
    parts.append(get_format_instruction(format_name, max_actions_per_step, action_sep))
    if example_count > 0:
        parts.append(_EXAMPLE_TEMPLATE.format(sep=action_sep))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Observation templates
# ---------------------------------------------------------------------------

def init_observation_template(observation: str, instruction: str) -> str:
    """First observation — includes instruction."""
    return (
        f"[Initial Observation]:\n"
        f"{observation}\n"
        f"Human Instruction: {instruction}\n"
        f"Decide your next action(s)."
    )


def action_template(
    valid_action,
    observation: str,
    env_feedback: str = "",
    reward=0.0,
    done=False,
) -> str:
    """Subsequent observations — no instruction repetition, no format instruction."""
    return (
        f"After your action, the extracted valid action is {valid_action}.\n"
        f"The environment feedback is: {env_feedback}\n"
        f"reward: {reward}\n"
        f"done: {done}\n"
        f"After that, the observation is:\n"
        f"{observation}\n"
        f"Decide your next action(s)."
    )
