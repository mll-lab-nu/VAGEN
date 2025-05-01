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
Your answer should be in the format of:
<think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"\ne.g. <think>The box is one step below me, and the target is two steps below me, I need to go down then push the box down to the target.</think><answer>Down{action_sep}Down</answer>"
        return base_prompt + example
    return base_prompt

def no_think_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
Your answer should be in the format of:
<answer>...</answer>"""
    
    if add_example:
        example = f"\ne.g. <answer>Down{action_sep}Down</answer>"
        return base_prompt + example
    return base_prompt

def grounding_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
Your answer should be in the format of:
<current_state>...</current_state><think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"\ne.g. <current_state>I'm at row 2 col 3. The box is at row 4 col 3. The target is at at row 5 col 3.</current_state><think>I need to go down then push the box down to the target</think><answer>Down{action_sep}Down</answer>"
        return base_prompt + example
    return base_prompt

def worldmodeling_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
Your answer should be in the format of:
<think>...</think><answer>...</answer><next_state>...</next_state>"""
    
    if add_example:
        example = f"\ne.g. <think>I need to go down then push the box down to the target.</think><answer>Down{action_sep}Down</answer><next_state>I'm at row 4 col 3. The box is at row 5 col 3. The target is at at row 5 col 3.</next_state>"
        return base_prompt + example
    return base_prompt

def grounding_worldmodeling_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
Your answer should be in the format of:
<current_state>...</current_state><think>...</think><answer>...</answer><next_state>...</next_state>"""
    
    if add_example:
        example = f"\ne.g. <current_state>I'm at row 2 col 3. The box is at row 4 col 3. The target is at at row 5 col 3.</current_state><think>I need to go down then push the box down to the target</think><answer>Down{action_sep}Down</answer><next_state>I'm at row 4 col 3. The box is at row 5 col 3. The target is at at row 5 col 3.</next_state>"
        return base_prompt + example
    return base_prompt

def grounding_symbol_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""Symbols:
# Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target
You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
Your answer should be in the format of:
<current_state>...</current_state><think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"\ne.g. <current_state>####\n#_P#\n#__#\n#_x#\n#_O#</current_state><think>I need to go down then push the box down to reach the target</think><answer>Down{action_sep}Down</answer>"
        return base_prompt + example
    return base_prompt

def worldmodeling_symbol_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
Your answer should be in the format of:
<think>...</think><answer>...</answer><next_state>...</next_state>"""
    
    if add_example:
        example = f"\ne.g. <think>I need to go down then push the box down to reach the target</think><answer>Down{action_sep}Down</answer><next_state>####\n#__#\n#__#\n#_P#\n#_√#</next_state>"
        return base_prompt + example
    return base_prompt

def grounding_worldmodeling_symbol_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
Your answer should be in the format of:
<current_state>...</current_state><think>...</think><answer>...</answer><next_state>...</next_state>"""
    
    if add_example:
        example = f"\ne.g. <current_state>####\n#_P#\n#__#\n#_x#\n#_O#</current_state><think>I need to go down then push the box down to reach the target</think><answer>Down{action_sep}Down</answer><next_state>####\n#__#\n#__#\n#_P#\n#_√#</next_state>"
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