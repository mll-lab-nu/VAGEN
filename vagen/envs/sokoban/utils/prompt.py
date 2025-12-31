def system_prompt():
    """Return the system prompt for Sokoban solver"""
    return """You are a Sokoban solver.
Sokoban Quick Guide
Goal: Push all boxes onto targets.
Symbols (If image is provided there are no symbols):
# Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target
Rules:
1. Push boxes (can't pull).
2. Avoid walls.
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
<think>The box is one step below me, and the target is two steps below me. I should go down to reach the box and then push it down to the target.</think>
<answer>Down</answer>

Example 2:
<think>The box is to the right of me, and the target is further to the right. I need to move right to get behind the box and push it toward the target.</think>
<answer>Right</answer>

Example 3:
<think>The box is above me, and the target is above the box. I should move up to reach the box and then push it upward to the target.</think>
<answer>Up</answer>
"""
        return base_prompt + "\n" + examples

    return base_prompt




def wm_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Generate format prompt for wm_new format with explicit row/column distinction"""
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
Your response must be in the format of:
<observation>...</observation><think>...</think><answer>...</answer><prediction>...</prediction>.

Rules for <observation> and <prediction>:
- You must strictly describe the relative position of the `target` and any visible `box` objects **relative to the player**.
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
- Separate multiple actions with `{action_sep}`.
"""

    if add_example:
        examples = f"""
Example 1:
<observation>The box is below and right of the player, and the target is below and right of the player</observation>
<think>I should move right to align my column with the box and the target</think>
<answer>Right</answer>
<prediction>The box will be below and same column of the player, and the target will be below and same column of the player</prediction>

Example 2:
<observation>The box is above and left of the player, and the target is above and same column of the player</observation>
<think>I should move up to align my row with the box and reach the target's row position</think>
<answer>Up</answer>
<prediction>The box will be same row and left of the player, and the target will be same row and same column of the player</prediction>

Example 3:
<observation>The box is same row and right of the player, and the target is same row and left of the player</observation>
<think>I should move right to push the box right while keeping the target on my left</think>
<answer>Right</answer>
<prediction>The box will be same row and right of the player, and the target will be same row and left of the player</prediction>
"""
        return base_prompt + "\n" + examples

    return base_prompt