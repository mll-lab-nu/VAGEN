def system_prompt():
    """Return the system prompt for FrozenLake solver"""
    return """You are a FrozenLake solver.
FrozenLake Quick Guide
Goal: Reach the goal (G).
Symbols (If image is provided there are no symbols):
_ Frozen | O Hole | G Goal | P Player | X Player fell into hole | V Player on goal
Rules:
1. Avoid falling into holes.
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.
Actions you can take: Left, Down, Right, Up."""


def init_observation_template(img_str):
    """Template for initial observation"""
    return f"""[Initial Observation]:
{img_str}
Decide your next action(s)."""


def action_template(valid_action, img_str):
    """Template for action feedback"""
    return f"""After your answer, the extracted valid action is {valid_action}.
After that, the observation is:
{img_str}
Decide your next action(s)."""


def format_prompt(max_actions_per_step, action_sep, add_example=True, prompt_format="free_think"):
    """Generate format prompt based on the specified format"""
    if prompt_format == "free_think":
        return free_think_format_prompt(max_actions_per_step, action_sep, add_example)
    elif prompt_format == "wm":
        return wm_format_prompt(max_actions_per_step, action_sep, add_example)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")


def free_think_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Generate format prompt for free_think format"""
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your reasoning, and then your answer.
Your response should be in the format of:
<think>...</think><answer>...</answer>"""

    if add_example:
        examples = f"""
Example 1:
<think>The goal is below me. I should go down to reach it while avoiding the hole on my left.</think>
<answer>Down</answer>

Example 2:
<think>The goal is to my right and there's a hole directly below me. I should go right first to avoid the hole.</think>
<answer>Right</answer>

Example 3:
<think>I can see the goal is up and to the left. I should move up first to get closer.</think>
<answer>Up</answer>
"""
        return base_prompt + "\n" + examples

    return base_prompt


def wm_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Generate format prompt for wm format with explicit row/column distinction"""
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
Your response must be in the format of:
<observation>...</observation><think>...</think><answer>...</answer><prediction>...</prediction>.

Rules for <observation> and <prediction>:
- You must strictly describe the relative position of the `goal` and any visible `hole` objects **relative to the player**.
- For each object, you MUST include:
  - exactly ONE vertical relationship: `above`, `below`, or `same row`
  - exactly ONE horizontal relationship: `left`, `right`, or `same column`
- Use ONLY the terms: `above`, `below`, `same row`, `left`, `right`, `same column`.
- Always use the phrasing pattern:
  "X is <vertical> and <horizontal> of the player".
- Do NOT use the word `same` alone.
- Do not include any extra information.

Rules for <answer>:
- Output 1 to {max_actions_per_step} action(s).
- Valid actions are: Up, Down, Left, Right.
- Separate multiple actions with `{action_sep}`."""

    if add_example:
        examples = f"""
Example 1:
<observation>The goal is below and right of the player, and there is a hole below and same column of the player</observation>
<think>I should move right to avoid the hole and get closer to the goal</think>
<answer>Right</answer>
<prediction>The goal will be below and same column of the player, and the hole will be below and left of the player</prediction>

Example 2:
<observation>The goal is above and left of the player</observation>
<think>I should move up to get closer to the goal</think>
<answer>Up</answer>
<prediction>The goal will be same row and left of the player</prediction>

Example 3:
<observation>The goal is same row and right of the player, and there is a hole above and right of the player</observation>
<think>I should move right to reach the goal</think>
<answer>Right</answer>
<prediction>The player will reach the goal</prediction>
"""
        return base_prompt + "\n" + examples

    return base_prompt