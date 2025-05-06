def system_prompt(**kwargs):
    return """You are a precise SVG code generator.

SVG Quick Guide
Goal: Transform the provided image into precise SVG code that replicates the image.

Process:
1. First analyze the image carefully, identifying distinct visual elements
2. Identify colors, dimensions, positions, and relationships between elements
3. Generate accurate SVG code that reproduces the image, you cam use path for better shape

Rewards:
- Overall visual similarity: +5.0
- Structural accuracy: +10.0
"""

def init_observation_template(observation):
    return f"""[Initial Observation]:
{observation}
Please carefully observe the image, and generate SVG code that reproduces it as accurately as possible.
Decide on your SVG code.
"""

def action_template(valid_action, observation, reward=None, done=None):
    return f"""After your answer, the extracted valid SVG code is {valid_action}.
After that, the observation is:
{observation}
reward: {reward}
done: {done}
Please revise your code to make it more precise and similar to the original image.
Decide on your revised SVG code.
"""

def free_think_format_prompt(max_actions_per_step=1, action_sep=",", add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your thought process, and then your answer. 
Your response should be in the format of:
<think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think>I can see the image contains a red circle and a blue rectangle. The circle is positioned at the top-left, while the rectangle is at the bottom-right.</think><answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def no_think_format_prompt(max_actions_per_step=1, action_sep=",", add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should provide only your answer.
Your response should be in the format of:
<answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def grounding_format_prompt(max_actions_per_step=1, action_sep=",", add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give the current state, then your thought process, and finally your answer.
The state should be in the format of {{"elements":[{{"type":"circle", "properties":{{"cx":25, "cy":25, "r":15, "fill":"red"}}}}, {{"type":"rect", "properties":{{"x":60, "y":60, "width":30, "height":20, "fill":"blue"}}}}]}}
Your response should be in the format of:
<current_state>...</current_state><think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <current_state>{{"elements":[{{"type":"circle", "properties":{{"cx":25, "cy":25, "r":15, "fill":"red"}}}}, {{"type":"rect", "properties":{{"x":60, "y":60, "width":30, "height":20, "fill":"blue"}}}}]}}</current_state><think>I need to adjust the red circle's position and the blue rectangle's size</think><answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def worldmodeling_format_prompt(max_actions_per_step=1, action_sep=",", add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your thought process, then your answer, and finally predict the next state.
The state should be in the format of {{"elements":[{{"type":"circle", "properties":{{"cx":25, "cy":25, "r":15, "fill":"red"}}}}, {{"type":"rect", "properties":{{"x":60, "y":60, "width":30, "height":20, "fill":"blue"}}}}]}}
Your response should be in the format of:
<think>...</think><answer>...</answer><next_state>...</next_state>"""
    
    if add_example:
        example = f"""e.g. <think>The image shows a red circle and a blue rectangle. I need to position them correctly.</think><answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer><next_state>{{"elements":[{{"type":"circle", "properties":{{"cx":25, "cy":25, "r":15, "fill":"red"}}}}, {{"type":"rect", "properties":{{"x":60, "y":60, "width":30, "height":20, "fill":"blue"}}}}], "similarity_score":0.95}}</next_state>"""
        return base_prompt + '\n' + example
    return base_prompt

def grounding_worldmodeling_format_prompt(max_actions_per_step=1, action_sep=",", add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give the current state, then your thought process, then your answer, and finally predict the next state.
The state should be in the format of {{"elements":[{{"type":"circle", "properties":{{"cx":25, "cy":25, "r":15, "fill":"red"}}}}, {{"type":"rect", "properties":{{"x":60, "y":60, "width":30, "height":20, "fill":"blue"}}}}]}}
Your response should be in the format of:
<current_state>...</current_state><think>...</think><answer>...</answer><next_state>...</next_state>"""
    
    if add_example:
        example = f"""e.g. <current_state>{{"elements":[{{"type":"circle", "properties":{{"cx":25, "cy":25, "r":15, "fill":"red"}}}}, {{"type":"rect", "properties":{{"x":60, "y":60, "width":30, "height":20, "fill":"blue"}}}}]}}</current_state><think>I need to adjust the red circle's position slightly and make the blue rectangle wider</think><answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="20" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="35" height="20" fill="blue" />
</svg></answer><next_state>{{"elements":[{{"type":"circle", "properties":{{"cx":20, "cy":25, "r":15, "fill":"red"}}}}, {{"type":"rect", "properties":{{"x":60, "y":60, "width":35, "height":20, "fill":"blue"}}}}], "similarity_score":0.98}}</next_state>"""
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