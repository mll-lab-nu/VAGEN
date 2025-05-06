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
You should first give your thought process, and then your answer. 
Your response should be in the format of:
<think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think>I can see the target is on my down left, I should go down then left to reach the target</think><answer>Down{action_sep}Left</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def no_think_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should provide only your answer.
Your response should be in the format of:
<answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <answer>Down{action_sep}Left</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def grounding_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give the current state, then your thought process, and finally your answer.
Your response should be in the format of:
<state>...</state><think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <state>The player is on the above the target</state><think>I should go down then left to reach the target</think><answer>Down{action_sep}Left</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def worldmodeling_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your thought process, then your answer, and finally predict the next state.
The state should be in the format of {{"player":(row1,column1),"target":(row2,column2)}}
Your response should be in the format of:
<think>...</think><answer>...</answer><state>...</state>"""
    
    if add_example:
        example = f"""e.g. <think>I can see the target is on my down left, I should go down then left</think><answer>Down{action_sep}Left</answer><state>The player will reach the target</state>"""
        return base_prompt + '\n' + example
    return base_prompt

def grounding_worldmodeling_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give the current state, then your thought process, then your answer, and finally predict the next state.
The state should be in the format of {{"player":(row1,column1),"target":(row2,column2)}}
Your response should be in the format of:
<state>...</state><think>...</think><answer>...</answer><state>...</state>"""
    
    if add_example:
        example = f"""e.g. <state>{{"player":(2,3),"target":(3,2)}}</state><think>I should go down then left to reach the target</think><answer>Down{action_sep}Left</answer><state>{{"player":(3,2),"target":(3,2)}}</state>"""
        return base_prompt + '\n' + example
    return base_prompt

def grounding_symbol_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give the current state as a grid, then your thought process, and finally your answer.
The state should be represented as a grid using the symbols: _ Frozen | O Hole | G Goal | P Player | X Player fell into hole | √ Player on goal.
Your response should be in the format of:
<state>...</state><think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <state>_P__
G___
_OO_
____</state><think>I should go down then left to reach the target</think><answer>Down{action_sep}Left</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def worldmodeling_symbol_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your thought process, then your answer, and finally predict the next state as a grid.
The state should be represented as a grid using the symbols: _ Frozen | O Hole | G Goal | P Player | X Player fell into hole | √ Player on goal.
Your response should be in the format of:
<think>...</think><answer>...</answer><state>...</state>"""
    
    if add_example:
        example = f"""e.g. <think>I can see the target is on my down left, I should go down then left</think><answer>Down{action_sep}Left</answer><state>____
√___
_OO_
____</state>"""
        return base_prompt + '\n' + example
    return base_prompt

def grounding_worldmodeling_symbol_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give the current state as a grid, then your thought process, then your answer, and finally predict the next state as a grid.
The state should be represented as grids using the symbols: _ Frozen | O Hole | G Goal | P Player | X Player fell into hole | √ Player on goal.
Your response should be in the format of:
<state>...</state><think>...</think><answer>...</answer><state>...</state>"""
    
    if add_example:
        example = f"""e.g. <state>_P__
G___
_OO_
____</state><think>I should go down then left to reach the target</think><answer>Down{action_sep}Left</answer><state>____
√___
_OO_
____</state>"""
        return base_prompt + '\n' + example
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