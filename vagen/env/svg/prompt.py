def system_prompt(**kwargs):
    format_type = kwargs.get("format", "default")
    
    # Base system prompt
    base_prompt = """You are a precise SVG code generator.

SVG Quick Guide
Goal: Transform the provided image into precise SVG code that replicates the image.

Process:
1. First analyze the image carefully, identifying distinct visual elements
2. Identify colors, dimensions, positions, and relationships between elements
3. Generate accurate SVG code that reproduces the image, you cam use path for better shape

Rewards:
- Overall visual similarity: +5.0
- Structural accuracy: +10.0"""

    # Add example based on format type
    if format_type in FORMAT_CONFIGS:
        example = FORMAT_CONFIGS[format_type].get("example", "")
        if example:
            return base_prompt + '\n' + "Example:\n" + example
    
    return base_prompt

def init_observation_template(**kwargs):
    observation = kwargs.get("observation", None)
    return f"""[Initial Observation]:
{observation}
Please carefully observe the image, and generate SVG code that reproduces it as accurately as possible.
Decide on your SVG code.
"""

def action_template(**kwargs):
    valid_action = kwargs.get("valid_action", None)
    observation = kwargs.get("observation", None)
    reward = kwargs.get("reward", None)
    done = kwargs.get("done", None)
    
    return f"""After your answer, the extracted valid SVG code is {valid_action}.
After that, the observation is:
{observation}
reward: {reward}
done: {done}
Please revise your code to make it more precise and similar to the original image.
Decide on your revised SVG code.
"""

# Format configurations defining the structure of each format
FORMAT_CONFIGS = {
    "free_think": {
        "format": "<think>...</think><answer>...</answer>",
        "description": "You should first give your thought process, and then your answer.",
        "example": """<think>I can see the image contains a red circle and a blue rectangle. The circle is positioned at the top-left, while the rectangle is at the bottom-right.</think>
<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>"""
    },
    
    "no_think": {
        "format": "<answer>...</answer>",
        "description": "You should provide only your answer.",
        "example": """<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>"""
    },
    
    "grounding": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "description": "You should first give your thought process with your observation and reasoning, and finally your answer.",
        "additional_info": "The observation should be described in detail about what you see in the image.",
        "example": """<think><observation>I can see a red circle positioned at the top-left corner of the canvas, and a blue rectangle at the bottom-right. The circle has a radius of approximately 15 units and is centered at coordinates (25, 25). The rectangle is approximately 30 units wide by 20 units tall and positioned at coordinates (60, 60).</observation><reasoning>I need to create an SVG with a viewBox of 0 0 100 100 to properly position these elements. I'll add a circle element with the observed properties and a rectangle element with the observed properties.</reasoning></think>
<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>"""
    },
    
    "worldmodeling": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give your thought process with reasoning and prediction of next state, then your answer.",
        "additional_info": "The prediction should describe what you expect to see after your actions are executed.",
        "example": """<think><reasoning>The image shows a red circle at the top-left and a blue rectangle at the bottom-right. I need to create an SVG that accurately reproduces these elements with their correct positions and dimensions.</reasoning><prediction>After implementing this SVG code, the result should closely match the original image. I expect a similarity score of at least 0.95, as the shapes and positions are relatively simple to reproduce.</prediction></think>
<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>"""
    },
    
    "grounding_worldmodeling": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give your thought process with the your observation and reasoning, then predict next state, and finally the answer.",
        "additional_info": "Both the observation and prediction should describe what you see or expect to see in the environment.",
        "example": """<think><observation>I can see an image containing a red circle positioned at the top-left area of the canvas, approximately at coordinates (25, 25) with a radius of 15 units. There is also a blue rectangle at the bottom-right area, sized about 30x20 units and positioned at coordinates (60, 60).</observation><reasoning>Based on my observation, I need to create an SVG that precisely matches these elements. The circle appears to be slightly too far right, so I should adjust its x-coordinate to 20 instead of 25. The rectangle could benefit from being slightly wider.</reasoning><prediction>After implementing these adjustments, the resulting SVG should more closely match the original image. I expect the similarity score to improve to approximately 0.98, as the modified positions and dimensions will better represent the original graphic.</prediction></think>
<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="20" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="35" height="20" fill="blue" />
</svg></answer>"""
    },
    
    # For default format, use free_think format
    "default": {
        "format": "<think>...</think><answer>...</answer>",
        "description": "You should first give your thought process, and then your answer.",
        "example": """<think>I can see the image contains a red circle and a blue rectangle. The circle is positioned at the top-left, while the rectangle is at the bottom-right.</think>
<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>"""
    }
}

def format_prompt_generator(format_type):
    """
    Generates a prompt function for the specified format type.
    
    Args:
        format_type (str): The format type to generate a prompt function for
        
    Returns:
        function: A function that generates a prompt for the specified format
    """
    def prompt_function(**kwargs):
        """
        Generate a prompt for the specified format.
        
        Args:
            max_actions_per_step (int): Maximum number of actions allowed per step
            action_sep (str): Separator between actions
            add_example (bool): Whether to add an example
            
        Returns:
            str: The formatted prompt
        """
        max_actions_per_step = kwargs.get("max_actions_per_step", 1)
        action_sep = kwargs.get("action_sep", ",")
        add_example = kwargs.get("add_example", True)
        
        # Use default format if format_type not found
        config = FORMAT_CONFIGS.get(format_type, FORMAT_CONFIGS["default"])
        
        # Build the base prompt text
        base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
{config["description"]}"""
        
        # Add additional information if available
        if "additional_info" in config:
            base_prompt += f"\n{config['additional_info']}"
        
        # Add response format instruction
        base_prompt += f"""
Your response should be in the format of:
{config["format"]}"""
        
        # Add example if requested
        if add_example:
            example = config["example"]
            return base_prompt + '\n' + f"e.g. {example}"
        
        return base_prompt
    
    return prompt_function

# Generate the format prompt dictionary using the generator
format_prompt = {format_type: format_prompt_generator(format_type) 
                for format_type in FORMAT_CONFIGS}

if __name__ == "__main__":
    # Example usage
    max_actions_per_step = 1
    action_sep = ","
    
    for key, func in format_prompt.items():
        if key != "default":  # Skip printing default as it's the same as free_think
            print(f"{key} format prompt:")
            print(func(max_actions_per_step=max_actions_per_step, action_sep=action_sep, add_example=True))
            print("\n" + "="*50 + "\n")