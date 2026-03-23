"""Prompt templates for the SVG environment.

Adapted for VAGEN's prompt format system (free_think / wm / free_wm).
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise SVG code generator.

SVG Quick Guide
Goal: Transform the provided image into precise SVG code that replicates the image.

Process:
1. First analyze the image carefully, identifying distinct visual elements
2. Identify colors, dimensions, positions, and relationships between elements
3. Generate accurate SVG code that reproduces the image, you can use path for better shape

Rewards:
- Overall visual similarity: +5.0
- Structural accuracy: +10.0"""


# ---------------------------------------------------------------------------
# Format configs (aligned with VAGEN's 3 formats)
# ---------------------------------------------------------------------------

FORMAT_CONFIGS = {
    "free_think": {
        "format": "<think>...</think><answer>...</answer>",
        "description": "You should first give your thought process, and then your answer.",
        "example": (
            "<think>I can see the image contains a red circle and a blue rectangle. "
            "The circle is positioned at the top-left, while the rectangle is at the bottom-right.</think>\n"
            '<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">\n'
            '  <circle cx="25" cy="25" r="15" fill="red" />\n'
            '  <rect x="60" y="60" width="30" height="20" fill="blue" />\n'
            "</svg></answer>"
        ),
    },
    "wm": {
        "format": (
            "<observation>...</observation>"
            "<think>...</think>"
            "<answer>...</answer>"
            "<prediction>...</prediction>"
        ),
        "description": (
            "You should first describe your observation of the image, "
            "then reason about it, provide your SVG code, "
            "and finally predict what the rendered result will look like."
        ),
        "example": (
            "<observation>I can see a red circle at the top-left corner and a blue rectangle "
            "at the bottom-right of the canvas.</observation>\n"
            "<think>I need to create an SVG with a viewBox of 0 0 100 100. "
            "The circle is centered at about (25,25) with radius 15, "
            "and the rectangle is at (60,60) with size 30x20.</think>\n"
            '<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">\n'
            '  <circle cx="25" cy="25" r="15" fill="red" />\n'
            '  <rect x="60" y="60" width="30" height="20" fill="blue" />\n'
            "</svg></answer>\n"
            "<prediction>The rendered image should closely match the original, "
            "with a similarity score of about 0.95.</prediction>"
        ),
    },
    "free_wm": {
        "format": (
            "<observation>...</observation> ... "
            "<answer>...</answer> ... "
            "<prediction>...</prediction>"
        ),
        "description": (
            "You should describe your observation, reason freely, "
            "provide your SVG code, and predict the result."
        ),
        "example": (
            "<observation>The image shows a red circle and a blue rectangle.</observation>\n"
            "The circle is at the top-left and the rectangle at bottom-right.\n"
            '<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">\n'
            '  <circle cx="25" cy="25" r="15" fill="red" />\n'
            '  <rect x="60" y="60" width="30" height="20" fill="blue" />\n'
            "</svg></answer>\n"
            "I expect this to match well.\n"
            "<prediction>Similarity around 0.95.</prediction>"
        ),
    },
}


def system_prompt(prompt_format: str = "free_think") -> str:
    """Return the system prompt with format-specific example."""
    config = FORMAT_CONFIGS.get(prompt_format, FORMAT_CONFIGS["free_think"])
    example = config.get("example", "")
    text = SYSTEM_PROMPT
    if example:
        text += "\n\nExample:\n" + example
    return text


def format_prompt(prompt_format: str = "free_think",
                  max_actions_per_step: int = 1,
                  action_sep: str = "~~",
                  add_example: bool = True) -> str:
    """Return format instructions for observations."""
    config = FORMAT_CONFIGS.get(prompt_format, FORMAT_CONFIGS["free_think"])
    text = (
        f"You can take up to {max_actions_per_step} action(s) at a time, "
        f"separated by {action_sep}.\n"
        f"{config['description']}\n"
        f"Your response should be in the format of:\n"
        f"{config['format']}"
    )
    if add_example:
        text += "\n\ne.g. " + config["example"]
    return text


def init_observation_template(observation: str) -> str:
    return (
        f"[Initial Observation]:\n{observation}\n"
        "Please carefully observe the image, and generate SVG code "
        "that reproduces it as accurately as possible.\n"
        "Decide on your SVG code."
    )


def action_template(valid_action: str, observation: str,
                     reward: float, done: bool) -> str:
    return (
        f"After your answer, the extracted valid SVG code is {valid_action}.\n"
        f"After that, the observation is:\n{observation}\n"
        f"reward: {reward}\ndone: {done}\n"
        "Please revise your code to make it more precise "
        "and similar to the original image.\n"
        "Decide on your revised SVG code."
    )
