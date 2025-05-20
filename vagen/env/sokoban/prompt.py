def system_prompt(**kwargs):
    return """You are a Sokoban solver.
Sokoban Quick Guide
Goal: Push all boxes onto targets.
Symbols (If image is provided there are no symbols):
# Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target
Rules:
1. Push boxes (can't pull).
2. Avoid walls.
3. You must clearly describe the position relationships between:
   - Player and Boxes (e.g., "The box is Up from my position")
   - Boxes and Targets (e.g., "The target is Right of the box")
4. For diagonal positions, always use two directional terms (e.g., "Up and Right", "Down and Left") that match the available actions.
5. Focus only on the relevant positioning of player, boxes, and targets - avoid unnecessary details.
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
        "example": "<think>I see that the box is Up and Right from my position, and the target is further Right from the box. I need to move Up first to align with the box's row, then move Right to position myself behind it, and finally push the box Right toward the target.</think><answer>Up{action_sep}Right{action_sep}Right</answer>"
    },
    
    "no_think": {
        "format": "<answer>...</answer>",
        "description": "You should provide only your answer.",
        "example": "<answer>Up{action_sep}Right{action_sep}Right</answer>"
    },
    
    "grounding": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "description": "You should first give the description of your observation, then your reasoning, and finally your answer.",
        "example": "<think><observation>The box is Up and Right from my position. The target is further Right from the box. There are no walls blocking the path.</observation><reasoning>I need to move Up first to align with the box's row, then move Right to position myself behind the box, and finally push the box Right toward the target.</reasoning></think><answer>Up{action_sep}Right{action_sep}Right</answer>"
    },
    
    "worldmodeling": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give your reasoning, then predict the next state, and finally your answer.",
        "example": "<think><reasoning>The box is Up and Left from my position, and the target is further Up from the box. I need to move Up to align with the box's column, then move Left to position myself behind it, and finally push the box Up toward the target.</reasoning><prediction>After my moves, I will be positioned Down from the box, and the box will be on the target.</prediction></think><answer>Up{action_sep}Left{action_sep}Up</answer>"
    },
    
    "grounding_worldmodeling": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give the description of your observation, then your reasoning, then predict the next state, and finally your answer.",
        "example": "<think><observation>The box is Down and Right from my position. The target is Down from the box. There are no walls blocking the path.</observation><reasoning>I need to first move Down to align with the box's column, then move Right to position myself behind the box, and finally push the box Down toward the target.</reasoning><prediction>After my moves, I will be positioned Up from the box, and the box will be on the target.</prediction></think><answer>Down{action_sep}Right{action_sep}Down</answer>"
    },
    
    "grounding_symbolic": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "description": "You should first give the description of your observation as a grid, then your reasoning, and finally your answer.",
        "additional_info": "The state should be represented as a grid using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.",
        "example": "<think><observation>####\n#P_#\n#_X#\n#__#\n#O_#</observation><reasoning>I observe that the box is Down and Right from my position, and the target is Down from the box. I need to move Down to align with the box's row, then move Right to get behind it, and finally push it Down toward the target.</reasoning></think><answer>Down{action_sep}Right{action_sep}Down</answer>"
    },
    
    "worldmodeling_symbolic": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give your reasoning, then predict the next state as a grid, and finally your answer.",
        "additional_info": "The state should be represented as a grid using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.",
        "example": "<think><reasoning>The box is Right and Up from my position, and the target is further Up from the box. I need to move Up to align with the box's column, then move Right to get behind it, and finally push it Up toward the target.</reasoning><prediction>####\n#_√#\n#_P#\n#__#\n####</prediction></think><answer>Up{action_sep}Right{action_sep}Up</answer>"
    },
    
    "grounding_worldmodeling_symbolic": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give the description of your observation as a grid, then your reasoning, then predict the next state as a grid, and finally your answer.",
        "additional_info": "The observation and state should be represented as grids using the symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target.",
        "example": "<think><observation>####\n#O_#\n#_X#\n#__#\n#_P#</observation><reasoning>The box is Up and Right from my position, and the target is further Up from the box. I need to move Up to align with the box's row, then move Right to get behind it, and finally push it Up toward the target.</reasoning><prediction>####\n#√_#\n#_P#\n#__#\n#__#</prediction></think><answer>Up{action_sep}Right{action_sep}Up</answer>"
    },
    
    "grounding_structured": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "description": "You should first give the description of your observation, then your reasoning, and finally your answer.",
        "additional_info": "The observation should be in the format of {\"player\":(row,column),\"box\":(row,column),\"target\":(row,column)}",
        "example": "<think><observation>{{\"player\":(4,2),\"box\":(2,3),\"target\":(1,3)}}</observation><reasoning>Based on the coordinates, the box is Up and Right from my position, and the target is Up from the box. I need to move Up to align with the box's column, then move Right to get behind the box, and finally push it Up to the target.</reasoning></think><answer>Up{action_sep}Up{action_sep}Right{action_sep}Up</answer>"
    },
    
    "worldmodeling_structured": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "The prediction should be in the format of {\"player\":(row,column),\"box\":(row,column),\"target\":(row,column)}",
        "example": "<think><reasoning>The box is Left and Down from my position, and the target is further Left from the box. I need to move Left to align with the box's row, then move Down to get behind it, and finally push it Left toward the target.</reasoning><prediction>{{\"player\":(3,2),\"box\":(3,1),\"target\":(3,1)}}</prediction></think><answer>Left{action_sep}Down{action_sep}Left</answer>"
    },
    
    "grounding_worldmodeling_structured": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give the description of your observation, then your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "The observation and prediction should be in the format of {\"player\":(row,column),\"box\":(row,column),\"target\":(row,column)}",
        "example": "<think><observation>{{\"player\":(2,1),\"box\":(1,3),\"target\":(1,4)}}</observation><reasoning>Based on the coordinates, the box is Up and Right from my position, and the target is Right from the box. I need to move Up first, then Right to position myself behind the box, and finally push it Right to reach the target.</reasoning><prediction>{{\"player\":(1,3),\"box\":(1,4),\"target\":(1,4)}}</prediction></think><answer>Up{action_sep}Right{action_sep}Right</answer>"
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