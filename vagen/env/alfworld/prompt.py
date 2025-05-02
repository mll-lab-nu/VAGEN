def system_prompt():
    return """You are an ALFRED household robot designed to perform household tasks in a text-based environment.

Task Guide:
Goal: Complete tasks in the TextWorld environment.

You should follow text descriptions and choose from available commands to navigate and interact with the environment. 
All available actions will be listed at each step.

Rewards:
- Correct format: +0.5
- Completing the task: +10.0
- Invalid action: -1.0
"""

def init_observation_template(observation, commands=None, instruction=None):
    instruction_text = f"Task: {instruction}\n" if instruction else ""
    commands_text = f"Available actions: [{commands}]\n" if commands else ""
    
    return f"""[Initial Observation]:
{observation}
{instruction_text}{commands_text}
Decide your next action.
"""

def action_template(valid_action, observation, commands=None, reward=None, done=None, instruction=None):
    instruction_text = f"Task: {instruction}\n" if instruction else ""
    commands_text = f"Available actions: [{commands}]\n" if commands else ""
    reward_text = f"Accumulated reward: {reward}\n" if reward is not None else ""
    
    return f"""After your answer, the executed action was: {valid_action}
Current observation:
{observation}
{instruction_text}{commands_text}{reward_text}
Decide your next action.
"""

def free_think_format_prompt(max_actions_per_step=1, action_sep=",", add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your thought process, and then your answer. 
Your response should be in the format of:
<think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think>I need to go to the kitchen and then open the fridge. Let me first navigate to the kitchen.</think><answer>go to kitchen</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def no_think_format_prompt(max_actions_per_step=1, action_sep=",", add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should provide only your answer.
Your response should be in the format of:
<answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <answer>go to kitchen</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def grounding_format_prompt(max_actions_per_step=1, action_sep=",", add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give the current state, then your thought process, and finally your answer.
The state should be in the format of {{"location":"kitchen", "objects_visible":["fridge", "counter", "sink"]}}
Your response should be in the format of:
<current_state>...</current_state><think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <current_state>{{"location":"living room", "objects_visible":["sofa", "tv", "coffee table"]}}</current_state><think>I need to go to the kitchen to find the fridge.</think><answer>go to kitchen</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def worldmodeling_format_prompt(max_actions_per_step=1, action_sep=",", add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your thought process, then your answer, and finally predict the next state.
The state should be in the format of {{"location":"kitchen", "objects_visible":["fridge", "counter", "sink"]}}
Your response should be in the format of:
<think>...</think><answer>...</answer><next_state>...</next_state>"""
    
    if add_example:
        example = f"""e.g. <think>I'm in the living room and need to go to the kitchen to find the fridge.</think><answer>go to kitchen</answer><next_state>{{"location":"kitchen", "objects_visible":["fridge", "counter", "sink"]}}</next_state>"""
        return base_prompt + '\n' + example
    return base_prompt

def grounding_worldmodeling_format_prompt(max_actions_per_step=1, action_sep=",", add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give the current state, then your thought process, then your answer, and finally predict the next state.
The state should be in the format of {{"location":"kitchen", "objects_visible":["fridge", "counter", "sink"]}}
Your response should be in the format of:
<current_state>...</current_state><think>...</think><answer>...</answer><next_state>...</next_state>"""
    
    if add_example:
        example = f"""e.g. <current_state>{{"location":"living room", "objects_visible":["sofa", "tv", "coffee table"]}}</current_state><think>I need to go to the kitchen to find the fridge.</think><answer>go to kitchen</answer><next_state>{{"location":"kitchen", "objects_visible":["fridge", "counter", "sink"]}}</next_state>"""
        return base_prompt + '\n' + example
    return base_prompt

# Dictionary mapping format names to their corresponding functions
format_prompt = {
    "free_think": free_think_format_prompt,
    "no_think": no_think_format_prompt,
    "grounding": grounding_format_prompt,
    "worldmodeling": worldmodeling_format_prompt,
    "grounding_worldmodeling": grounding_worldmodeling_format_prompt
}