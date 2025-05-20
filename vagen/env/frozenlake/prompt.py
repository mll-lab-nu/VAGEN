def system_prompt(**kwargs):
    return """You are a FrozenLake solver.
FrozenLake Quick Guide
goal: Reach the goal (G).
Symbols (If image is provided there are no symbols):
_ Frozen | O Hole | G Goal | P Player | X Player fell into hole | √ Player on goal
Rules:
1. Avoid falling into holes.
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.
Actions you can take: Left, Down, Right, Up. 
3. You must describe the positional relationship between the player and the goal, and when necessary, note the relationship between the player and holes.
4. When describing positions, use directional terms that align with the available actions (Left, Down, Right, Up). 
For example, if the goal is to the bottom-right of the player, describe "The goal is Down and Right from your current position"
"""

def init_observation_template(**kwargs):
    observation = kwargs.get("observation", "The player is on the above the goal")
    return f"""[Initial Observation]:
{observation}
Decide your next action(s).
"""

def action_template(**kwargs):
    valid_action, observation= kwargs.get("valid_action", "Down"), kwargs.get("observation", "The player is on the above the goal")
    return f"""After your answer, the extracted valid action is {valid_action}.
After that, the observation is:
{observation}
Decide your next action(s).
"""

# Format configurations defining the structure of each format
FORMAT_CONFIGS = {
    "free_think": {
        "format": "<think>...</think><answer>...</answer>",
        "description": "You should first give your reasoning, and then your answer.",
        "example": "<think>The goal is located Down and Right from my current position. There is a hole located two steps Down from my position that I need to avoid. The safest path appears to be moving Right first and then Down to reach the goal.</think><answer>Right{action_sep}Down</answer>"
    },
    
    "no_think": {
        "format": "<answer>...</answer>",
        "description": "You should provide only your answer.",
        "example": "<answer>Right{action_sep}Down</answer>"
    },
    
    "grounding": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think>",
        "description": "You should first describe the observation, then your reasoning, and finally your answer.",
        "example": "<think><observation>The goal is located Down and Right from the player's position. There is a hole located two steps Down that must be avoided.</observation><reasoning>To safely reach the goal, I should move Right first to avoid the hole, and then move Down to reach the goal position.</reasoning></think><answer>Right{action_sep}Down</answer>"
    },
    
    "worldmodeling": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first give your reasoning, then predict the next state, and finally your answer.",
        "example": "<think><reasoning>The goal is Down and Right from my current position. There's a hole directly Down that I need to avoid. The safest path is to move Right first, then Down to reach the goal.</reasoning><prediction>If I move Right, I'll be positioned directly above the goal. Then moving Down will place me on the goal tile.</prediction></think><answer>Right{action_sep}Down</answer>"
    },
    
    "grounding_worldmodeling": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "description": "You should first describe the observation, then your reasoning, then predict the next state, and finally your answer.",
        "example": "<think><observation>The goal is positioned Down and Right from the player. There is a hole Down from the player that must be avoided.</observation><reasoning>To safely navigate to the goal, I should first move Right to avoid the hole, then move Down to reach the goal.</reasoning><prediction>After moving Right, I'll be positioned directly above the goal. Then moving Down will place me on the goal tile, completing the level.</prediction></think><answer>Right{action_sep}Down</answer>"
    },
    
    "grounding_symbolic": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "description": "You should first describe the observation as a grid, then your reasoning, and finally your answer.",
        "additional_info": "The observation should be represented as a grid using the symbols: _ Frozen | O Hole | G Goal | P Player | X Player fell into hole | √ Player on goal.",
        "example": "<think><observation>_P__\n__G_\n_O__\n____</observation><reasoning>I see that the goal is Down and Right from my position. There is a hole directly Down that I need to avoid. I should move Right first to avoid the hole, then move Down to reach the goal.</reasoning></think><answer>Right{action_sep}Down</answer>"
    },
    
    "worldmodeling_symbolic": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think>",
        "description": "You should first give your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "The prediction should be represented as a grid using the symbols: _ Frozen | O Hole | G Goal | P Player | X Player fell into hole | √ Player on goal.",
        "example": "<think><reasoning>The goal is Down and Right from my position. There is a hole directly Down that I need to avoid. I should move Right first to avoid the hole, then Down to reach the goal.</reasoning><prediction>____\n__√_\n_O__\n____</prediction></think><answer>Right{action_sep}Down</answer>"
    },
    
    "grounding_worldmodeling_symbolic": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think>",
        "description": "You should first describe the observation as a grid, then your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "The observation and state should be represented as grids using the symbols: _ Frozen | O Hole | G Goal | P Player | X Player fell into hole | √ Player on goal.",
        "example": "<think><observation>_P__\n__G_\n_O__\n____</observation><reasoning>I observe that the goal is Down and Right from my position. There is a hole directly Down that I need to avoid. I should move Right first to avoid the hole, then Down to reach the goal.</reasoning><prediction>____\n__√_\n_O__\n____</prediction></think><answer>Right{action_sep}Down</answer>"
    },
    "grounding_structured": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "description": "You should first describe the observation as a grid, then your reasoning, and finally your answer.",
        "additional_info": "The observation should be in the format of {{'player':(row,column),'goal':(row,column)}}",
        "example": "<think><observation>{{'player':(0,1),'goal':(1,2)}}</observation><reasoning>The goal is Down and Right from my current position. I need to carefully navigate to reach the goal. The safest path is to move Right first, then Down to reach the goal.</reasoning></think><answer>Right{action_sep}Down</answer>"
    },
    "worldmodeling_structured": {
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think>",
        "description": "You should first give your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "The prediction should be in the format of {{'player':(row,column),'goal':(row,column)}}",
        "example": "<think><reasoning>The goal is Down and Right from my position. To reach it safely, I should move Right first to avoid the hole below me, then Down to reach the goal.</reasoning><prediction>{{'player':(1,2),'goal':(1,2)}}</prediction></think><answer>Right{action_sep}Down</answer>"
    },
    "grounding_worldmodeling_structured": {
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think>",
        "description": "You should first describe the observation as a grid, then your reasoning, then predict the next state, and finally your answer.",
        "additional_info": "The observation and prediction should be in the format of {{'player':(row,column),'goal':(row,column)}}",
        "example": "<think><observation>{{'player':(0,1),'goal':(1,2)}}</observation><reasoning>I observe that the goal is Down and Right from my current position. There appears to be a hole at (2,1) that I need to avoid. The safest path is to move Right first, then Down to reach the goal.</reasoning><prediction>{{'player':(1,2),'goal':(1,2)}}</prediction></think><answer>Right{action_sep}Down</answer>"
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
        add_example = kwargs.get("add_example", False)
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