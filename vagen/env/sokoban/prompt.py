def system_prompt(**kwargs):
    return """You are a Sokoban solver.
Sokoban Quick Guide
Goal: Push all boxes onto targets.
Symbols (If image is provided there are no symbols):
# Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target
Rules:
1. Push boxes (can't pull).
2. Avoid walls.
Actions you can take: Left, Down, Right, Up."""

def init_observation_template(**kwargs):
    observation = kwargs.get("img_str", "The player is near a box")
    return f"""[Initial Observation]:
{observation}
Decide your next action(s)."""

def action_template(**kwargs):
    valid_action = kwargs.get("valid_action", "Down")
    observation = kwargs.get("img_str", "The player pushed the box closer to the target")
    return f"""After your answer, the extracted valid action is {valid_action}.
After that, the observation is:
{observation}
Decide your next action(s)."""

# Format configurations defining the structure of each format
FORMAT_CONFIGS = {
    "free_think": {
        "format": "<think>...</think><answer>...</answer>",
        "description": "You should first give your reasoning, and then your answer.",
        "example": "<think><reasoning>The box is one step below me, and the target is two steps below me, I need to go down then push the box down to the target.</reasoning></think><answer>Down{action_sep}Down</answer>"
    },
    
    "no_think": {
        "format": "<answer>...</answer>",
        "description": "You should provide only your answer.",
        "example": "<answer>Down{action_sep}Down</answer>"
    },
    
    "grounding": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "description": "You should first give the description of your observation, then your reasoning, and finally your answer.",
        "example": "<think><observation>The box is below the player and the target is below the box</observation><reasoning>I need to go down then push the box down to the target</reasoning></think><answer>Down{action_sep}Down</answer>"
    },
    
    "worldmodeling": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give your reasoning, then predict the next state, and finally your answer.",
        "example": "<think><reasoning>I need to go right then push the box down to the target.</reasoning><prediction>The player will be above the box, the target and box will be at the same place.</prediction></think><answer>Right{action_sep}Down</answer>"
    },
    
    "grounding_worldmodeling": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give the description of your observation, then your reasoning, then predict the next state, and finally your answer.",
        "example": "<think><observation>The box is below the player and the target is below the box</observation><reasoning>I need to go down then push the box down to the target</reasoning><prediction>The player will be above the box, the target and box will be at the same place.</prediction></think><answer>Down{action_sep}Down</answer>"
    },
    
    "grounding_symbolic": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "description": "You should first give the description of your observation as a grid, then your reasoning, and finally your answer.",
        "additional_info": "The state should be represented as a grid using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.",
        "example": "<think><observation>####\n#_P#\n#__#\n#_X#\n#_O#</observation><reasoning>I need to go down then push the box down to reach the target</reasoning></think><answer>Down{action_sep}Down</answer>"
    },
    
    "worldmodeling_symbolic": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give your reasoning, then predict the next state as a grid, and finally your answer.",
        "additional_info": "The state should be represented as a grid using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.",
        "example": "<think><reasoning>I need to go down then push the box down to reach the target</reasoning><prediction>####\n#__#\n#__#\n#_P#\n#_√#</prediction></think><answer>Down{action_sep}Down</answer>"
    },
    
    "grounding_worldmodeling_symbolic": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give the description of your observation as a grid, then your reasoning, then predict the next state as a grid, and finally your answer.",
        "additional_info": "The observation and state should be represented as grids using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.",
        "example": "<think><observation>####\n#_P#\n#__#\n#_X#\n#_O#</observation><reasoning>I need to go down then push the box down to reach the target</reasoning><prediction>####\n#__#\n#__#\n#_P#\n#_√#</prediction></think><answer>Down{action_sep}Down</answer>"
    },
    
    "grounding_structured": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "description": "You should first give the description of your observation, then your reasoning, and finally your answer.",
        "additional_info": "The observation should be in the format of {\"player\":(row,column),\"box\":(row,column),\"target\":(row,column)}",
        "example": "<think><observation>{{\"player\":(1,2),\"box\":(3,2),\"target\":(4,2)}}</observation><reasoning>I need to go down then push the box down to the target</reasoning></think><answer>Down{action_sep}Down</answer>"
    },
    
    "worldmodeling_structured": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "The prediction should be in the format of {\"player\":(row,column),\"box\":(row,column),\"target\":(row,column)}",
        "example": "<think><reasoning>I need to go down then push the box down to the target</reasoning><prediction>{{\"player\":(3,2),\"box\":(4,2),\"target\":(4,2)}}</prediction></think><answer>Down{action_sep}Down</answer>"
    },
    
    "grounding_worldmodeling_structured": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give the description of your observation, then your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "The observation and prediction should be in the format of {\"player\":(row,column),\"box\":(row,column),\"target\":(row,column)}",
        "example": "<think><observation>{{\"player\":(1,2),\"box\":(3,2),\"target\":(4,2)}}</observation><reasoning>I need to go down then push the box down to the target</reasoning><prediction>{{\"player\":(3,2),\"box\":(4,2),\"target\":(4,2)}}</prediction></think><answer>Down{action_sep}Down</answer>"
    },
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
        action_sep = kwargs.get("action_sep", "|")
        add_example = kwargs.get("add_example", True)
        config = FORMAT_CONFIGS[format_type]
        
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
            example = config["example"].format(action_sep=action_sep)
            return base_prompt + '\n' + f"e.g. {example}"
        
        return base_prompt
    
    return prompt_function

# Generate the format prompt dictionary using the generator
format_prompt = {format_type: format_prompt_generator(format_type) 
                for format_type in FORMAT_CONFIGS}

if __name__ == "__main__":
    # Example usage
    max_actions_per_step = 2
    action_sep = "|"
    
    for key, func in format_prompt.items():
        print(f"{key} format prompt:")
        print(func(max_actions_per_step=max_actions_per_step, action_sep=action_sep, add_example=True))
        print("\n" + "="*50 + "\n")