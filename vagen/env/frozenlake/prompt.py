def system_prompt():
    return """You are a FrozenLake solver.

FrozenLake Quick Guide
Goal: Reach the goal (G).

Symbols (If image is provided there are no symbols):
_ Frozen | O Hole | G Goal | P Player | X Player fell into hole | √ Player on goal

Rules:
1. Avoid falling into holes.
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Actions you can take: Left, Down, Right, Up. 
"""

def init_observation_template(observation):
    return f"""[Initial Observation]:
{observation}
Decide your next action(s).
"""

def action_template(valid_action, observation):
    return f"""After your answer, the extracted valid action is {valid_action}.
After that, the observation is:
{observation}
Decide your next action(s).
"""

def free_think_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You answer should be in the format of:
<think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"\ne.g. <think>I can see the target is on my down left, I should go down then left</think><answer>Down{action_sep}Left</answer>"
        return base_prompt + example
    return base_prompt

def no_think_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You answer should be in the format of:
<answer>...</answer>"""
    
    if add_example:
        example = f"\ne.g. <answer>Down{action_sep}Left</answer>"
        return base_prompt + example
    return base_prompt

def grounding_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You answer should be in the format of:
<current_state>...</current_state><think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"\ne.g. <current_state>I'm in the row 2 col 3. The target is in the row 3 col 2.</current_state><think>I should go down then left to reach the target</think><answer>Down{action_sep}Left</answer>"
        return base_prompt + example
    return base_prompt

def worldmodeling_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You answer should be in the format of:
<think>...</think><answer>...</answer><next_state>...</next_state>"""
    
    if add_example:
        example = f"\ne.g. <think>I can see the target is on my down left, I should go down then left</think><answer>Down{action_sep}Left</answer><next_state>I'm in the row 3 col 2. The target is in the row 3 col 2.</next_state>"
        return base_prompt + example
    return base_prompt

def grounding_worldmodeling_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You answer should be in the format of:
<current_state>...</current_state><think>...</think><answer>...</answer><next_state>...</next_state>"""
    
    if add_example:
        example = f"\ne.g. <current_state>I'm in the row 2 col 3. The target is in the row 3 col 2.</current_state><think>I should go down then left to reach the target</think><answer>Down{action_sep}Left</answer><next_state>I'm in the row 3 col 2. The target is in the row 3 col 2.</next_state>"
        return base_prompt + example
    return base_prompt

def grounding_symbol_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You answer should be in the format of:
<current_state>...</current_state><think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"\ne.g. <current_state>_P__\nG___\n_OO_\n____</current_state><think>I should go down then left to reach the target</think><answer>Down{action_sep}Left</answer>"
        return base_prompt + example
    return base_prompt

def worldmodeling_symbol_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You answer should be in the format of:
<think>...</think><answer>...</answer><next_state>...</next_state>"""
    
    if add_example:
        example = f"\ne.g. <think>I can see the target is on my down left, I should go down then left</think><answer>Down{action_sep}Left</answer><next_state>____\n√___\n_OO_\n____</next_state>"
        return base_prompt + example
    return base_prompt

def grounding_worldmodeling_symbol_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You answer should be in the format of:
<current_state>...</current_state><think>...</think><answer>...</answer><next_state>...</next_state>"""
    
    if add_example:
        example = f"\ne.g. <current_state>_P__\nG___\n_OO_\n____</current_state><think>I should go down then left to reach the target</think><answer>Down{action_sep}Left</answer><next_state>____\n√___\n_OO_\n____</next_state>"
        return base_prompt + example
    return base_prompt

# Dictionary mapping format names to their corresponding functions
format_prompt = {
    "free_think": free_think_format_prompt,
    "no_think": no_think_format_prompt,
    "grounding": grounding_format_prompt,
    "worldmodeling": worldmodeling_format_prompt,
    "grounding_worldmodeling": grounding_worldmodeling_format_prompt,
    "grounding_symbol": grounding_symbol_format_prompt,
    "worldmodeling_symbol": worldmodeling_symbol_format_prompt,
    "grounding_worldmodeling_symbol": grounding_worldmodeling_symbol_format_prompt
}