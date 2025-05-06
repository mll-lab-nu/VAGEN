def system_prompt(**kwargs):
    
    if kwargs.get("format", "default") in ["free_think", "default"]:
        example=f"""Example:
<think>I can see the image contains a red circle and a blue rectangle. The circle is positioned at the top-left, while the rectangle is at the bottom-right.</think>
<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>"""
    elif kwargs.get("format", "default") == "grounding":
        example=f"""Example:
<think><observation>I can see a red circle positioned at the top-left corner of the canvas, and a blue rectangle at the bottom-right. The circle has a radius of approximately 15 units and is centered at coordinates (25, 25). The rectangle is approximately 30 units wide by 20 units tall and positioned at coordinates (60, 60).</observation><reasoning>I need to create an SVG with a viewBox of 0 0 100 100 to properly position these elements. I'll add a circle element with the observed properties and a rectangle element with the observed properties.</reasoning></think>
<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>"""
    elif kwargs.get("format", "default") == "worldmodeling":
        example=f"""Example:
<think><reasoning>The image shows a red circle and a blue rectangle. I need to position them correctly.</reasoning><prediction>After implementing this SVG code, the result should closely match the original image. I expect a similarity score of at least 0.95, as the shapes and positions are relatively simple to reproduce.</prediction></think>
<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>"""
    elif kwargs.get("format", "default") == "grounding_worldmodeling":
        example=f"""Example:
<think><observation>I can see an image containing a red circle positioned at the top-left area of the canvas, approximately at coordinates (25, 25) with a radius of 15 units. There is also a blue rectangle at the bottom-right area, sized about 30x20 units and positioned at coordinates (60, 60).</observation><reasoning>Based on my observation, I need to create an SVG that precisely matches these elements. The circle appears to be slightly too far right, so I should adjust its x-coordinate to 20 instead of 25. The rectangle could benefit from being slightly wider.</reasoning><prediction>After implementing these adjustments, the resulting SVG should more closely match the original image. I expect the similarity score to improve to approximately 0.98, as the modified positions and dimensions will better represent the original graphic.</prediction></think>
<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="20" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="35" height="20" fill="blue" />
</svg></answer>"""
    elif kwargs.get("format", "default") == "no_think":
        example=f"""Example:
<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>"""
    return """You are a precise SVG code generator.

SVG Quick Guide
Goal: Transform the provided image into precise SVG code that replicates the image.

Process:
1. First analyze the image carefully, identifying distinct visual elements
2. Identify colors, dimensions, positions, and relationships between elements
3. Generate accurate SVG code that reproduces the image, you cam use path for better shape

Rewards:
- Overall visual similarity: +5.0
- Structural accuracy: +10.0""" + '\n' + example

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
You should first give your thought process with your observation and reasoning, and finally your answer.
The observation should be described in detail about what you see in the image.
Your response should be in the format of:
<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think><observation>I can see a red circle positioned at the top-left corner of the canvas, and a blue rectangle at the bottom-right. The circle has a radius of approximately 15 units and is centered at coordinates (25, 25). The rectangle is approximately 30 units wide by 20 units tall and positioned at coordinates (60, 60).</observation><reasoning>I need to create an SVG with a viewBox of 0 0 100 100 to properly position these elements. I'll add a circle element with the observed properties and a rectangle element with the observed properties.</reasoning></think><answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def worldmodeling_format_prompt(max_actions_per_step=1, action_sep=",", add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your thought process with reasoning and prediction of next state, then your answer.
The prediction should describe what you expect to see after your actions are executed.
Your response should be in the format of:
<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think><reasoning>The image shows a red circle at the top-left and a blue rectangle at the bottom-right. I need to create an SVG that accurately reproduces these elements with their correct positions and dimensions.</reasoning><prediction>After implementing this SVG code, the result should closely match the original image. I expect a similarity score of at least 0.95, as the shapes and positions are relatively simple to reproduce.</prediction></think><answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def grounding_worldmodeling_format_prompt(max_actions_per_step=1, action_sep=",", add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your thought process with the your observation and reasoning, then predict next state, and finally the answer.
Both the observation and prediction should describe what you see or expect to see in the environment.
Your response should be in the format of:
<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think><observation>I can see an image containing a red circle positioned at the top-left area of the canvas, approximately at coordinates (25, 25) with a radius of 15 units. There is also a blue rectangle at the bottom-right area, sized about 30x20 units and positioned at coordinates (60, 60).</observation><reasoning>Based on my observation, I need to create an SVG that precisely matches these elements. The circle appears to be slightly too far right, so I should adjust its x-coordinate to 20 instead of 25. The rectangle could benefit from being slightly wider.</reasoning><prediction>After implementing these adjustments, the resulting SVG should more closely match the original image. I expect the similarity score to improve to approximately 0.98, as the modified positions and dimensions will better represent the original graphic.</prediction></think><answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="20" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="35" height="20" fill="blue" />
</svg></answer>"""
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