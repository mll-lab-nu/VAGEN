"""
Prompt templates for the primitive_skill (ManiSkill robot manipulation) environment.

Supports 2 prompt formats:
  - free_think: <think>...</think><answer>...</answer>
  - wm: <observation>...</observation><think>...</think><answer>...</answer><prediction>...</prediction>
"""

from __future__ import annotations

from typing import List, Optional


# ---------------------------------------------------------------------------
# Format instructions
# ---------------------------------------------------------------------------

_FORMAT_INSTRUCTIONS = {
    "free_think": (
        "You need to think first, then give your answer. Respond in this format:\n"
        "<think>...</think><answer>{action_example}</answer>"
    ),
    "wm": (
        "You need to describe your observation, think, give your answer, then predict "
        "what you will see next. Respond in this format:\n"
        "<observation>...</observation><think>...</think>"
        "<answer>{action_example}</answer><prediction>...</prediction>"
    ),
}


def get_format_instruction(
    format_name: str,
    max_actions_per_step: int = 2,
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


VALID_FORMATS = list(_FORMAT_INSTRUCTIONS.keys())


# ---------------------------------------------------------------------------
# System prompt (static robot description + action space)
# ---------------------------------------------------------------------------

def system_prompt(
    format_name: str = "free_think",
    max_actions_per_step: int = 2,
    action_sep: str = "|",
    state_keys: Optional[List[str]] = None,
    add_example: bool = True,
) -> str:
    """Return full system prompt = robot description + format instruction."""
    parts = [_ROBOT_DESCRIPTION]
    parts.append(get_format_instruction(format_name, max_actions_per_step, action_sep))
    return "\n\n".join(parts)


_ROBOT_DESCRIPTION = """\
You are an AI assistant controlling a Franka Emika robot arm. Your goal is to understand human instructions and translate them into a sequence of executable actions for the robot, based on visual input and the instruction.

Action Space Guide
You can command the robot using the following actions:

1. pick(x, y, z) # To grasp an object located at position(x,y,z) in the robot's workspace.
2. place(x, y, z) # To place the object currently held by the robot's gripper at the target position (x,y,z).
3. push(x1, y1, z1, x2, y2, z2) # To push an object from position (x1,y1,z1) to (x2,y2,z2).

Hints:
1. The coordinates (x, y, z) are in millimeters and are all integers.
2. Please ensure that the coordinates are within the workspace limits.
3. The position is the center of the object, when you place, please consider the volume of the object. It's always fine to set z much higher when placing an item.
4. We will provide the object positions to you, but you need to match them to the object in the image by yourself. You're facing toward the negative x-axis, and the negative y-axis is to your left, the positive y-axis is to your right, and the positive z-axis is up.

Examples:
round1:
image1
Human Instruction: Put red cube on green cube and yellow cube on left target
Object positions:
[(62,-55,20),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)]
Reasoning: I can see from the picture that the red cube is on my left and green cube is on my right and near me.
Since I'm looking toward the negative x axis, and negative y-axis is to my left, (62,-55,20) would be the position of the red cube, (75,33,20) would be the position of the green cube and (-44,100,20) is the position of the yellow cube.
Also the (100,-43,0) would be the position of the left target, and (100,43,0) would be the porition of the right target.
I need to pick up red cube first and place it on the green cube, when placing, I should set z much higher.
Anwer: pick(62,-55,20)|place(75,33,50)
round2:
image2
Human Instruction: Put red cube on green cube and yellow cube on left target
Object positions:
[(75,33,50),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)]
Reasoning: Now the red cube is on the green cube, so I need to pick up the yellow cube and place it on the left target.
Anwer: pick(-44,100,20)|place(100,-43,50)"""


# ---------------------------------------------------------------------------
# Observation templates
# ---------------------------------------------------------------------------

def init_observation_template(
    observation: str = "<image>",
    instruction: str = "",
    x_workspace: tuple = (0, 0),
    y_workspace: tuple = (0, 0),
    z_workspace: tuple = (0, 0),
    object_positions: str = "",
    other_information: str = "",
    **kwargs,
) -> str:
    return f"""
[Initial Observation]:
{observation}
Human Instruction: {instruction}
x_workspace_limit: {x_workspace}
y_workspace_limit: {y_workspace}
z_workspace_limit: {z_workspace}
Object positions:
{object_positions}
Other information:
{other_information}
Decide your next action(s)."""


def action_template(
    valid_actions: list = None,
    observation: str = "<image>",
    instruction: str = "",
    x_workspace: tuple = (0, 0),
    y_workspace: tuple = (0, 0),
    z_workspace: tuple = (0, 0),
    object_positions: str = "",
    other_information: str = "",
    **kwargs,
) -> str:
    return f"""After your answer, the extracted valid action(s) is {valid_actions}.
After that, the observation is:
{observation}
Human Instruction: {instruction}
x_workspace_limit: {x_workspace}
y_workspace_limit: {y_workspace}
z_workspace_limit: {z_workspace}
Object positions:
{object_positions}
Other information:
{other_information}
Decide your next action(s)."""


def get_format_prompt(
    format_name: str = "free_think",
    max_actions_per_step: int = 2,
    action_sep: str = "|",
    state_keys: Optional[List[str]] = None,
    add_example: bool = True,
) -> str:
    """Alias — returns the format instruction string."""
    return get_format_instruction(format_name, max_actions_per_step, action_sep)
