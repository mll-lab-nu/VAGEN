def system_prompt():
    return """You are a Sokoban solver.

Sokoban Quick Guide
Goal: Push all boxes onto targets.

Symbols (If image is provided there are no symbols):
# Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target

Rules:
1. Push boxes (can't pull).
2. Avoid walls.

Actions you can take: Left, Down, Right, Up."""

def init_observation_template(observation):
    return f"""[Initial Observation]:
{observation}
Decide your next action(s)."""

def action_template(valid_action, observation):
    return f"""After your answer, the extracted valid action is {valid_action}.
After that, the observation is:
{observation}
Decide your next action(s)."""

def free_think_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your thought process, and then your answer.
Your response should be in the format of:
<think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think>The box is one step below me, and the target is two steps below me, I need to go down then push the box down to the target.</think><answer>Down{action_sep}Down</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def no_think_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should provide only your answer.
Your response should be in the format of:
<answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <answer>Down{action_sep}Down</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

# def grounding_format_prompt(max_actions_per_step, action_sep, add_example=True):
#     base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
# You should first give the description current state, then your thought process, and finally your answer.
# The state should be in the format of {{"player":(row1,column1),"box":(row2,column2),"target":(row3,column3)}}
# Your response should be in the format of:
# <state>...</state><think>...</think><answer>...</answer>"""
    
#     if add_example:
#         example = f"""e.g. <state>{{"player":(2,3),"box":(4,3),"target":(5,3)}}</state><think>I need to go down then push the box down to the target</think><answer>Down{action_sep}Down</answer>"""
#         return base_prompt + '\n' + example
#     return base_prompt

# def worldmodeling_format_prompt(max_actions_per_step, action_sep, add_example=True):
#     base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
# You should first give your thought process, then your answer, and finally predict the next state.
# The state should be in the format of {{"player":(row1,column1),"box":(row2,column2),"target":(row3,column3)}}
# Your response should be in the format of:
# <think>...</think><answer>...</answer><state>...</state>"""
    
#     if add_example:
#         example = f"""e.g. <think>I need to go down then push the box down to the target.</think><answer>Down{action_sep}Down</answer><state>{{"player":(4,3),"box":(5,3),"target":(5,3)}}</state>"""
#         return base_prompt + '\n' + example
#     return base_prompt

def grounding_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give the description current state, then your thought process, and finally your answer.
Your response should be in the format of:
<state>...</state><think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <state>The box is below the player and the target is below the box</state><think>I need to go down then push the box down to the target</think><answer>Down{action_sep}Down</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def worldmodeling_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your thought process, then your answer, and finally predict the next state.
Your response should be in the format of:
<think>...</think><answer>...</answer><state>...</state>"""
    
    if add_example:
        example = f"""e.g. <think>I need to go right then push the box down to the target.</think><answer>Right{action_sep}Down</answer><state>The player will be above the box, the target and box will be at the same place.</state>"""
        return base_prompt + '\n' + example
    return base_prompt

def grounding_worldmodeling_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give the description current state, then your thought process, then your answer, and finally predict the next state.
The state should be in the format of {{"player":(row1,column1),"box":(row2,column2),"target":(row3,column3)}}
Your response should be in the format of:
<state>...</state><think>...</think><answer>...</answer><state>...</state>"""
    
    if add_example:
        example = f"""e.g. <state>{{"player":(2,3),"box":(4,3),"target":(5,3)}}</state><think>I need to go down then push the box down to the target</think><answer>Down{action_sep}Down</answer><state>{{"player":(4,3),"box":(5,3),"target":(5,3)}}</state>"""
        return base_prompt + '\n' + example
    return base_prompt

def grounding_symbol_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give the description current state as a grid, then your thought process, and finally your answer.
The state should be represented as a grid using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.
Your response should be in the format of:
<state>...</state><think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <state>####
#_P#
#__#
#_X#
#_O#</state><think>I need to go down then push the box down to reach the target</think><answer>Down{action_sep}Down</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def worldmodeling_symbol_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your thought process, then your answer, and finally predict the next state as a grid.
The state should be represented as a grid using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.
Your response should be in the format of:
<think>...</think><answer>...</answer><state>...</state>"""
    
    if add_example:
        example = f"""e.g. <think>I need to go down then push the box down to reach the target</think><answer>Down{action_sep}Down</answer><state>####
#__#
#__#
#_P#
#_√#</state>"""
        return base_prompt + '\n' + example
    return base_prompt

def grounding_worldmodeling_symbol_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give the description current state as a grid, then your thought process, then your answer, and finally predict the next state as a grid.
The state should be represented as grids using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.
Your response should be in the format of:
<state>...</state><think>...</think><answer>...</answer><state>...</state>"""
    
    if add_example:
        example = f"""e.g. <state>####
#_P#
#__#
#_X#
#_O#</state><think>I need to go down then push the box down to reach the target</think><answer>Down{action_sep}Down</answer><state>####
#__#
#__#
#_P#
#_√#</state>"""
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