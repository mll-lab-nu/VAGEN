def system_prompt(**kwargs):
    """
    Generate a system prompt for robot arm control with examples in various formats.
    
    Args:
        **kwargs: Keyword arguments including:
            format (str): The format for examples. Options are:
                - "default" or "free_think": Unstructured thinking process
                - "grounding": Observation and reasoning separation
                - "worldmodeling": Reasoning with predictions
                - "grounding_worldmodeling": Combined observation, reasoning, and prediction
                - "no_think": No thinking, just actions
    """
    
    # Format configurations
    format_configs = {
        "free_think": {
            "think_format": "<think>...</think>",
            "example_template": """Example:
round1:
image1
Human Instruction: Put red cube on green cube and yellow cube on left target
Object positions:
[(62,-55,20),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)]
<think>I can see from the picture that the red cube is on my left and green cube is on my right and near me. 
Since I'm looking toward the negative x axis, and negative y-axis is to my left, (62,-55,20) would be the position of the red cube, (75,33,20) would be the position of the green cube and (-44,100,20) is the position of the yellow cube. 
Also the (100,-43,0) would be the position of the left target, and (100,43,0) would be the position of the right target.
I need to pick up red cube first and place it on the green cube, when placing, I should set z much higher.</think>
<answer>pick(62,-55,20)|place(75,33,70)</answer>
round2:
image2
Human Instruction: Put red cube on green cube and yellow cube on left target
Object positions:
[(75,33,60),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)]
<think>Now the red cube is on the green cube, so I need to pick up the yellow cube and place it on the left target.</think>
<answer>pick(-44,100,20)|place(100,-43,30)</answer>"""
        },
        
        "grounding": {
            "think_format": "<think><observation>...</observation><reasoning>...</reasoning></think>",
            "example_template": """Example:
round1:
image1
Human Instruction: Put red cube on green cube and yellow cube on left target
Object positions:
[(62,-55,20),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)]
<think><observation>I can see from the picture that the red cube is on my left and green cube is on my right and near me. Since I'm looking toward the negative x axis, and negative y-axis is to my left, (62,-55,20) would be the position of the red cube, (75,33,20) would be the position of the green cube and (-44,100,20) is the position of the yellow cube. Also the (100,-43,0) would be the position of the left target, and (100,43,0) would be the position of the right target.</observation><reasoning>I need to pick up red cube first and place it on the green cube, when placing, I should set z much higher.</reasoning></think>
<answer>pick(62,-55,20)|place(75,33,70)</answer>
round2:
image2
Human Instruction: Put red cube on green cube and yellow cube on left target
Object positions:
[(75,33,60),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)]
<think><observation>Now the red cube is on the green cube.</observation><reasoning>I need to pick up the yellow cube and place it on the left target.</reasoning></think>
<answer>pick(-44,100,20)|place(100,-43,30)</answer>"""
        },
        
        "worldmodeling": {
            "think_format": "<think><reasoning>...</reasoning><prediction>...</prediction></think>",
            "example_template": """Example:
round1:
image1
Human Instruction: Put red cube on green cube and yellow cube on left target
Object positions:
[(62,-55,20),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)]
<think><reasoning>I can see from the picture that the red cube is on my left and green cube is on my right and near me. 
Since I'm looking toward the negative x axis, and negative y-axis is to my left, (62,-55,20) would be the position of the red cube, (75,33,20) would be the position of the green cube and (-44,100,20) is the position of the yellow cube. 
Also the (100,-43,0) would be the position of the left target, and (100,43,0) would be the position of the right target.
I need to pick up red cube first and place it on the green cube, when placing, I should set z much higher.</reasoning><prediction>After executing this action, the red cube will be at position (75,33,60), stacked on top of the green cube at (75,33,20)</prediction></think>
<answer>pick(62,-55,20)|place(75,33,70)</answer>
round2:
image2
Human Instruction: Put red cube on green cube and yellow cube on left target
Object positions:
[(75,33,60),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)]
<think><reasoning>Now the red cube is on the green cube, so I need to pick up the yellow cube and place it on the left target.</reasoning><prediction>After executing this action, the yellow cube will be at position (100,-43,20), placed on the left target at (100,-43,0)</prediction></think>
<answer>pick(-44,100,20)|place(100,-43,30)</answer>"""
        },
        
        "grounding_worldmodeling": {
            "think_format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think>",
            "example_template": """Example:
round1:
image1
Human Instruction: Put red cube on green cube and yellow cube on left target
Object positions:
[(62,-55,20),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)]
<think><observation>I can see from the picture that the red cube is on my left and green cube is on my right and near me. Since I'm looking toward the negative x axis, and negative y-axis is to my left, (62,-55,20) would be the position of the red cube, (75,33,20) would be the position of the green cube and (-44,100,20) is the position of the yellow cube. Also the (100,-43,0) would be the position of the left target, and (100,43,0) would be the position of the right target.</observation><reasoning>I need to pick up red cube first and place it on the green cube, when placing, I should set z much higher.</reasoning><prediction>After executing this action, the red cube will be at position (75,33,60), stacked on top of the green cube at (75,33,20)</prediction></think>
<answer>pick(62,-55,20)|place(75,33,70)</answer>
round2:
image2
Human Instruction: Put red cube on green cube and yellow cube on left target
Object positions:
[(75,33,60),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)]
<think><observation>Now the red cube is on the green cube at position (75,33,60), with the green cube still at (75,33,20).</observation><reasoning>I need to pick up the yellow cube and place it on the left target.</reasoning><prediction>After executing this action, the yellow cube will be at position (100,-43,20), placed on the left target at (100,-43,0)</prediction></think>
<answer>pick(-44,100,20)|place(100,-43,30)</answer>"""
        },
        
        "no_think": {
            "think_format": "",
            "example_template": """Example:
round1:
image1
Human Instruction: Put red cube on green cube and yellow cube on left target
Object positions:
[(62,-55,20),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)]
<answer>pick(62,-55,20)|place(75,33,70)</answer>
round2:
image2
Human Instruction: Put red cube on green cube and yellow cube on left target
Object positions:
[(75,33,60),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)]
<answer>pick(-44,100,20)|place(100,-43,30)</answer>"""
        }
    }
    
    # Default to free_think if format not specified or not recognized
    format_key = kwargs.get("format", "default")
    if format_key == "default":
        format_key = "free_think"
    
    config = format_configs.get(format_key, format_configs["free_think"])
    
    # Base prompt for all formats
    base_prompt = """You are an AI assistant controlling a Franka Emika robot arm. Your goal is to understand human instructions and translate them into a sequence of executable actions for the robot, based on visual input and the instruction.

Action Space Guide
You can command the robot using the following actions:

1. pick(x, y, z) # To grasp an object located at position(x,y,z) in the robot's workspace.
2. place(x, y, z) # To place the object currently held by the robot's gripper at the target position (x,y,z).
3. push(x1, y1, z1, x2, y2, z2) # To push an object from position (x1,y1,z1) to (x2,y2,z2).

Hints: 
1. The coordinates (x, y, z) are in millimeters and are all integers.
2. Please ensure that the coordinates are within the workspace limits.
3. The position is the center of the object, when you place, please consider the volume of the object. It's always fine to set z much higher when placing an item.
4. We will provide the object positions to you, but you need to match them to the object in the image by yourself. You're facing toward the negative x-axis, and the negative y-axis is to your left, the positive y-axis is to your right, and the positive z-axis is up."""
    
    return base_prompt + '\n' + config["example_template"]

def init_observation_template(observation, instruction, x_workspace, y_workspace, z_workspace, object_positions, other_information, object_names=None):
    return f"""
[Initial Observation]:
{observation}
Human Instruction: {instruction}
x_workspace_limit: {x_workspace}
y_workspace_limit: {y_workspace}
z_workspace_limit: {z_workspace}
Object positions: 
{object_positions}
Other information:
{other_information}
Decide your next action(s)."""

def action_template(valid_actions, observation, instruction, x_workspace, y_workspace, z_workspace, object_positions, other_information, object_names=None):
    return f"""After your answer, the extracted valid action(s) is {valid_actions}.
After that, the observation is:
{observation}
Human Instruction: {instruction}
x_workspace_limit: {x_workspace}
y_workspace_limit: {y_workspace}
z_workspace_limit: {z_workspace}
Object positions: 
{object_positions}
Other information:
{other_information}
Decide your next action(s)."""

def format_prompt_factory(format_type):
    """
    Factory function to create the appropriate format prompt function based on format type.
    
    Args:
        format_type (str): The format type for the prompt
        
    Returns:
        function: The appropriate format prompt function
    """
    
    # Format configurations
    format_configs = {
        "free_think": {
            "think_format": "<think>...</think>",
            "description": "You should first give your thought process, and then your answer.",
            "example_template": "<think>I need to pick the red cube at (100,100,40) first and place it on top of the green cube at (200,200,60)</think><answer>pick(100,100,40){action_sep}place(200,200,100)</answer>"
        },
        
        "no_think": {
            "think_format": "",
            "description": "You should provide only your answer.",
            "example_template": "<answer>pick(100,100,40){action_sep}place(200,200,100)</answer>"
        },
        
        "grounding": {
            "think_format": "<think><observation>...</observation><reasoning>...</reasoning></think>",
            "description": "You should first give your thought process with observation and reasoning, and then your answer.",
            "example_template": "<think><observation>I see a red cube at (100,100,40) and a green cube at (200,200,60).</observation><reasoning>I need to pick the red cube and place it on top of the green cube</reasoning></think><answer>pick(100,100,40){action_sep}place(200,200,100)</answer>"
        },
        
        "worldmodeling": {
            "think_format": "<think><reasoning>...</reasoning><prediction>...</prediction></think>",
            "description": "You should first give your thought process with reasoning and prediction, and then your answer.",
            "example_template": "<think><reasoning>I need to pick the {target_object} at (100,100,40) and place it at (80,100,60)</reasoning><prediction>After executing this action, the {target_object} will be at (80,100,60)</prediction></think><answer>pick(100,100,40){action_sep}place(80,100,60)</answer>"
        },
        
        "grounding_worldmodeling": {
            "think_format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think>",
            "description": "You should first give your thought process with observation, reasoning, and prediction, and then your answer.",
            "example_template": "<think><observation>I see a red cube at (100,100,40) and a green cube at (200,200,60).</observation><reasoning>I need to pick the red cube and place it on top of the green cube</reasoning><prediction>After executing this action, the red cube will be at position (200,200,100), stacked on top of the green cube at (200,200,60)</prediction></think><answer>pick(100,100,40){action_sep}place(200,200,100)</answer>"
        }
    }
    
    def prompt_function(max_actions_per_step, action_sep, state_keys, add_example=True):
        """
        Generate format-specific prompts based on configuration.
        
        Args:
            max_actions_per_step (int): Maximum number of actions allowed per step
            action_sep (str): Separator between actions
            state_keys (list): List of object states to track/predict
            add_example (bool): Whether to add an example
            
        Returns:
            str: The formatted prompt
        """
        config = format_configs.get(format_type, format_configs["free_think"])
        
        base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
{config["description"]}
Your response should be in the format of:
{config["think_format"]}<answer>...</answer>"""
        
        if add_example:
            # Use the first key as the object being manipulated in the example
            target_object = state_keys[0].replace("_position", "") if state_keys else "red_cube"
            example = config["example_template"].format(action_sep=action_sep, target_object=target_object)
            return base_prompt + '\n' + f"e.g. {example}"
        
        return base_prompt
    
    return prompt_function

# Generate the format prompt dictionary using the factory
format_prompt = {
    "free_think": format_prompt_factory("free_think"),
    "no_think": format_prompt_factory("no_think"),
    "grounding": format_prompt_factory("grounding"),
    "worldmodeling": format_prompt_factory("worldmodeling"),
    "grounding_worldmodeling": format_prompt_factory("grounding_worldmodeling")
}

if __name__ == "__main__":
    # Example usage
    max_actions_per_step = 2
    action_sep = "|"
    state_keys = ["red_cube_position", "green_cube_position", "yellow_cube_position"]
    
    for key, func in format_prompt.items():
        print(f"{key} format prompt:")
        print(func(max_actions_per_step, action_sep, state_keys))
        print("\n" + "="*50 + "\n")