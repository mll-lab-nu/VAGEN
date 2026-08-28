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
    elif prompt_format in {"free_wm", "wm_think"}:
        return wm_think_format_prompt(max_actions_per_step, action_sep, add_example)
    elif prompt_format == "answer":
        return answer_format_prompt(max_actions_per_step, action_sep, add_example)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")

def free_think_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Generate format prompt for free_think format.

    Think, then answer -- the whole contract. Written to hold for both kinds of model:
    one that types `<think>` as text, and one whose chat template has already opened the
    block for it, so that the response begins inside the reasoning and its first tag is
    `</think>`. Hence "close it with `</think>`" rather than "emit `<think>`": the latter
    is unsatisfiable on the families that reserve the token.

    The requirement that the answer follow `</think>` is the load-bearing part for a
    native-thinking model. It is the only thing in the format that says *stop reasoning*.
    """
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your reasoning, and then your answer.
Your response should be in the format of:
<think>...</think><answer>...</answer>

Rules:
- Close your reasoning with `</think>` before answering; the answer must come after it.
- Output 1 to {max_actions_per_step} action(s) inside `<answer>`.
- Valid actions are: Up, Down, Left, Right.
- Separate multiple actions with `{action_sep}`.
- Do not put anything other than actions inside `<answer>`."""

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
    """Generate the repository-wide structured world-model format."""
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
Your response must be in the format of:
<perception>...</perception><reasoning>...</reasoning><prediction>...</prediction><answer>...</answer>.

Rules for <perception> and <prediction>:
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
<perception>The box is below and right of the player, and the target is below and right of the player</perception>
<reasoning>I should move right to align my column with the box and the target</reasoning>
<prediction>The box will be below and same column of the player, and the target will be below and same column of the player</prediction>
<answer>Right</answer>

Example 2:
<perception>The box is above and left of the player, and the target is above and same column of the player</perception>
<reasoning>I should move up to align my row with the box and reach the target's row position</reasoning>
<prediction>The box will be same row and left of the player, and the target will be same row and same column of the player</prediction>
<answer>Up</answer>

Example 3:
<perception>The box is same row and right of the player, and the target is same row and left of the player</perception>
<reasoning>I should move right to push the box right while keeping the target on my left</reasoning>
<prediction>The box will be same row and right of the player, and the target will be same row and left of the player</prediction>
<answer>Right</answer>
"""
        return base_prompt + "\n" + examples

    return base_prompt


def answer_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Just the action, for models that reason in their own thinking channel.

    Qwen3.5 and friends open a `<think>` block in the generation prompt and reason inside
    it before the visible response begins. Asking such a model *also* to narrate its
    reasoning into `<perception>`/`<reasoning>`/`<prediction>` tags makes it do the work
    twice: once natively and once for the parser. It also collides outright -- `<think>`
    is a reserved control token on those families, so the tag can never be produced as
    text. Use ``wm_think`` when native reasoning should precede structured WM output.

    So this format marks up only what has to be machine-read: the action. Everything
    before `<answer>` is the model's own reasoning and is left alone.
    """
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
End your response with your chosen action(s) in the format:
<answer>...</answer>

Rules for <answer>:
- Output 1 to {max_actions_per_step} action(s).
- Valid actions are: Up, Down, Left, Right.
- Separate multiple actions with `{action_sep}`.
- Do not put anything other than actions inside the tag."""

    if add_example:
        example = f"""

Example:
<answer>Right{action_sep}Up</answer>"""
        return base_prompt + example
    return base_prompt


def wm_think_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Structured WM after an optional model-native thinking block."""
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
If the chat template opens a native `<think>` block, finish it first with `</think>`.
After that, your response must use exactly this order:
<perception>...</perception><reasoning>...</reasoning><prediction>...</prediction><answer>...</answer>.

Rules for <perception> and <prediction>:
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
<perception>The box is below and right of the player, and the target is below and right of the player</perception>
<reasoning>I should move right to align my column with the box and the target.</reasoning>
<prediction>The box will be below and same column of the player, and the target will be below and same column of the player</prediction>
<answer>Right</answer>

Example 2:
<perception>The box is above and left of the player, and the target is above and same column of the player</perception>
<reasoning>I should move up to align my row with the box and reach the target's row position.</reasoning>
<prediction>The box will be same row and left of the player, and the target will be same row and same column of the player</prediction>
<answer>Up</answer>

Example 3:
<perception>The box is same row and right of the player, and the target is same row and left of the player</perception>
<reasoning>I should move right to push the box right while keeping the target on my left.</reasoning>
<prediction>The box will be same row and right of the player, and the target will be same row and left of the player</prediction>
<answer>Right</answer>
"""
        return base_prompt + "\n" + examples

    return base_prompt
