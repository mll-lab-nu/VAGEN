system_prompt= """You are a FrozenLake solver.

FrozenLake Quick Guide
Goal: Reach the goal (G).

Symbols (If image is provided there are no symbols):
_ Frozen | O Hole | G Goal | P Player | X Player fell into hole | √ Player on goal

Rules:
1. Avoid falling into holes.
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Actions you can take: Left, Down, Right, Up. 
"""

init_observation_template = """[Initial Observation]:
{observation}
Decide your next action(s).
"""

action_template = """After your answer, the extracted valid action is {valid_action}.
After that, the observation is:
{observation}
Decide your next action(s).
"""

free_think_format_prompt= """You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You answer should be in the format of:
<think>...</think><answer>...</answer>
e.g. <think>I can see the target is on my down left, I should go down then left</think><answer>Down{action_sep}Left</answer>
"""

no_think_format_prompt= """You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You answer should be in the format of:
<answer>...</answer>
e.g. <answer>Down{action_sep}Left</answer>
"""

grounding_format_prompt= """You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You answer should be in the format of:
<current_state>...</current_state><think>...</think><answer>...</answer>
e.g. <current_state>I'm in the row 2 col 3. The target is in the row 3 col 2.</current_state><think>I should go down then left to reach the target</think><answer>Down{action_sep}Left</answer>
"""

worldmodeling_format_prompt= """You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You answer should be in the format of:
<think>...</think><answer>...</answer><next_state>...</next_state>
e.g. <think>I can see the target is on my down left, I should go down then left</think><answer>Down{action_sep}Left</answer><next_state>I'm in the row 3 col 2. The target is in the row 3 col 2.</next_state>
"""

grounding_worldmodeling_format_prompt= """You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You answer should be in the format of:
<current_state>...</current_state><think>...</think><answer>...</answer><next_state>...</next_state>
e.g. <current_state>I'm in the row 2 col 3. The target is in the row 3 col 2.</current_state><think>I should go down then left to reach the target</think><answer>Down{action_sep}Left</answer><next_state>I'm in the row 3 col 2. The target is in the row 3 col 2.</next_state>
"""

grounding_symbol_format_prompt= """You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You answer should be in the format of:
<current_state>...</current_state><think>...</think><answer>...</answer>
e.g. <current_state>_P__\nG___\n_OO_\n____</current_state><think>I should go down then left to reach the target</think><answer>Down{action_sep}Left</answer>
"""

worldmodeling_format_symbol= """You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You answer should be in the format of:
<think>...</think><answer>...</answer><next_state>...</next_state>
e.g. <think>I can see the target is on my down left, I should go down then left</think><answer>Down{action_sep}Left</answer><next_state>____\n√___\n_OO_\n____</next_state>
"""

grounding_worldmodeling_symbol_format_prompt= """You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You answer should be in the format of:
<current_state>...</current_state><think>...</think><answer>...</answer><next_state>...</next_state>
e.g. <current_state>_P__\nG___\n_OO_\n____</current_state><think>I should go down then left to reach the target</think><answer>Down{action_sep}Left</answer><next_state>____\n√___\n_OO_\n____</next_state>
"""

format_prompt={
    "free_think": free_think_format_prompt,
    "no_think": no_think_format_prompt,
    "grounding": grounding_format_prompt,
    "worldmodeling": worldmodeling_format_prompt,
    "grounding_worldmodeling": grounding_worldmodeling_format_prompt,
    "grounding_symbol": grounding_symbol_format_prompt,
    "worldmodeling_symbol": worldmodeling_format_symbol,
    "grounding_worldmodeling_symbol": grounding_worldmodeling_symbol_format_prompt
}