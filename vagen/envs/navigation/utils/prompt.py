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
        "<think>...</think><answer>{action_example}</answer>"
    ),
    "wm": (
        "Describe your perception, explain your reasoning, predict the next state, and "
        "then give your action. Respond in this exact order:\n"
        "<perception>...</perception><reasoning>...</reasoning>"
        "<prediction>...</prediction><answer>{action_example}</answer>"
    ),
    "no_think": (
        "You need to only give your action. Respond in this format:\n"
        "<answer>{action_example}</answer>"
    ),
    "eval_mode": (
        "You can optionally think first, then give your action. Respond in this format:\n"
        "<think>...</think><answer>{action_example}</answer>"
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
Actions you can take: move_forward, move_backward, move_right, move_left, turn_right, turn_left, look_up, look_down.
move_forward: Move forward by some distance
move_backward: Move backward by some distance
move_right: Move rightward by some distance
move_left: Move leftward by some distance
turn_right: Rotate to the right by 90 degrees
turn_left: Rotate to the left by 90 degrees
look_up: Tilt the camera upward by 30 degrees
look_down: Tilt the camera downward by 30 degrees
The instruction will be provided in the first observation. Look at the image carefully and navigate to complete the instruction.
Hints:
1. You can take multiple actions at a time, in most cases, if you find the target object is far away from you, you can call move_forward, move_left and move_right multiple times.
2. If you find yourself seems to be stuck, you can look_down to see if there's any object above or below you, you can also rotate to see if there's any object behind you."""

_EXAMPLE_STEPS = (
    (
        "image_1",
        "The garbage can is upper-left, next to the sink, and a counter blocks forward motion.",
        "Move left first to clear the counter.",
        "The garbage can should become more centered and reachable.",
        "move_left{sep} move_left",
    ),
    (
        "image_2",
        "The garbage can is now in front and slightly left, with open space ahead.",
        "Move forward several times and then left.",
        "The robot should end close to the garbage can.",
        "move_forward{sep} move_forward{sep} move_forward{sep} move_left",
    ),
)


def _example_response(format_name: str, perception: str, reasoning: str, prediction: str, answer: str) -> str:
    if format_name == "wm":
        return (
            f"<perception>{perception}</perception>"
            f"<reasoning>{reasoning}</reasoning>"
            f"<prediction>{prediction}</prediction>"
            f"<answer>{answer}</answer>"
        )
    if format_name in {"free_think", "eval_mode"}:
        return f"<think>{reasoning}</think><answer>{answer}</answer>"
    return f"<answer>{answer}</answer>"


def _format_example(format_name: str, sep: str) -> str:
    rounds = ["Example 1:"]
    for index, (image, perception, reasoning, prediction, answer) in enumerate(_EXAMPLE_STEPS, 1):
        rounds.extend(
            [
                f"Round {index}:",
                image,
                _example_response(
                    format_name,
                    perception,
                    reasoning,
                    prediction,
                    answer.format(sep=sep),
                ),
            ]
        )
    rounds.extend(["Round 3:", "Env_feedback: Success"])
    return "\n".join(rounds)


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
    if example_count:
        parts.append(_format_example(format_name, action_sep))
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
