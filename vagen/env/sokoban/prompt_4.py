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
You should first give your reasoning, and then your answer.
Your response should be in the format of:
<think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think><reasoning>The box is one step below me, and the target is two steps below me, I need to go down then push the box down to the target.</reasoning></think><answer>Down{action_sep}Down</answer>"""
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

def grounding_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first your reasoning with observation and reasoning, and your answer.
Your response should be in the format of:
<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think><observation>The box is below the player and the target is below the box</observation><reasoning>I need to go down then push the box down to the target</reasoning></think><answer>Down{action_sep}Down</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def worldmodeling_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your reasoning, then predict the next state, and finally the answer.
Your response should be in the format of:
<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think><reasoning>I need to go right then push the box down to the target.</reasoning><prediction>The player will be above the box, the target and box will be at the same place.</prediction></think><answer>Right{action_sep}Down</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def grounding_worldmodeling_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give the description of your observation, then your reasoning, then predict the next state, and finally the answer.
The state should be in the format of {{"player":(row1,column1),"box":(row2,column2),"target":(row3,column3)}}
Your response should be in the format of:
<think><observation>...</observation><reasoning>...</reasoning><prediction>{{"player":(row1,column1),"box":(row2,column2),"target":(row3,column3)}}</prediction></think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think><observation>The box is below the player and the target is below the box</observation><reasoning>I need to go down then push the box down to the target</reasoning><prediction>The player will be above the box, the target and box will be at the same place.</prediction></think><answer>Down{action_sep}Down</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def grounding_symbol_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give the description of your observation as a grid, then your reasoning, and finally your answer.
The state should be represented as a grid using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.
Your response should be in the format of:
<think><observation>####
#_P#
#__#
#_X#
#_O#</observation><reasoning>...</reasoning></think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think><observation>####
#_P#
#__#
#_X#
#_O#</observation><reasoning>I need to go down then push the box down to reach the target</reasoning></think><answer>Down{action_sep}Down</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def worldmodeling_symbol_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your reasoning, then predict the next state as a grid, and finally your answer.
The state should be represented as a grid using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.
Your response should be in the format of:
<think><reasoning>...</reasoning><prediction>####
#__#
#__#
#_P#
#_√#</prediction></think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think><reasoning>I need to go down then push the box down to reach the target</reasoning><prediction>####
#__#
#__#
#_P#
#_√#</prediction></think><answer>Down{action_sep}Down</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def grounding_worldmodeling_symbol_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give the description of your observation as a grid, then your reasoning, then predict the next state as a grid, and finally your answer.
The state should be represented as grids using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.
Your response should be in the format of:
<think><observation>####
#_P#
#__#
#_X#
#_O#</observation><reasoning>...</reasoning><prediction>####
#__#
#__#
#_P#
#_√#</prediction></think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think><observation>####
#_P#
#__#
#_X#
#_O#</observation><reasoning>I need to go down then push the box down to reach the target</reasoning><prediction>####
#__#
#__#
#_P#
#_√#</prediction></think><answer>Down{action_sep}Down</answer>"""
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