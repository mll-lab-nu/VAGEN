def system_prompt(**kwargs):
    """
    Returns the system prompt for the robot arm control.
    
    Returns:
        str: The system prompt
    """
    return """You are an AI assistant controlling a Franka Emika robot arm. Your goal is to understand human instructions and translate them into a sequence of executable actions for the robot, based on visual input and the instruction.

Action Space Guide
You can command the robot using the following actions:

1. pick(x, y, z) # To grasp an object located at position(x,y,z) in the robot's workspace.
2. place(x, y, z) # To place the object currently held by the robot's gripper at the target position (x,y,z).
3. push(x1, y1, z1, x2, y2, z2) # To push an object from position (x1,y1,z1) to (x2,y2,z2).

Hints: 
1. The coordinates (x, y, z) are in millimeters and are all integers.
2. Please ensure that the coordinates are within the workspace limits.
3. The position is the center of the object, when you place, please consider the volume of the object. It's always fine to set z much higher when placing an item.
4. We will provide the object positions to you, but you need to match them to the object in the image by yourself. You're facing toward the negative x-axis, and the negative y-axis is to your left, the positive y-axis is to your right, and the positive z-axis is up. 

Examples:
round1:
image1
Human Instruction: Put red cube on green cube and yellow cube on left target
Object positions:
[(62,-55,20),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)]
Reasoning: I can see from the picture that the red cube is on my left and green cube is on my right and near me. 
Since I'm looking toward the negative x axis, and negative y-axis is to my left, (62,-55,20) would be the position of the red cube, (75,33,20) would be the position of the green cube and (-44,100,20) is the position of the yellow cube. 
Also the (100,-43,0) would be the position of the left target, and (100,43,0) would be the porition of the right target.
I need to pick up red cube first and place it on the green cube, when placing, I should set z much higher.
Anwer: pick(62,-55,20)|place(75,33,50)
round2:
image2
Human Instruction: Put red cube on green cube and yellow cube on left target
Object positions:
[(75,33,50),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)]
Reasoning: Now the red cube is on the green cube, so I need to pick up the yellow cube and place it on the left target.
Anwer: pick(-44,100,20)|place(100,-43,50)
"""

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

def create_state_examples(state_keys):
    """
    Creates initial and next state examples based on state keys.
    
    Args:
        state_keys (list): List of object states to track/predict
        
    Returns:
        tuple: (state_format, init_state_example, next_state_example, target_object)
    """
    # Create state format dictionary
    state_format = {key: "(x,y,z)" for key in state_keys}
    
    # Create initial state example
    init_state_example = {}
    for i, key in enumerate(state_keys):
        if i == 0:
            init_state_example[key] = "(100,100,40)"  # This will be the object to pick
        else:
            init_state_example[key] = f"({i*100+200},{i*100+200},{50})"
    
    # Create next state example showing the object movement
    next_state_example = {}
    for i, key in enumerate(state_keys):
        if i == 0:
            next_state_example[key] = "(80,100,50)"  # This is where it's placed
        else:
            next_state_example[key] = init_state_example[key]  # Other objects remain in place
    
    # Get target object for examples
    target_object = state_keys[0] if state_keys else "red_cube_position"
    
    return state_format, init_state_example, next_state_example, target_object

def create_base_prompt(max_actions_per_step, action_sep, description):
    """
    Creates a base prompt with shared formatting.
    
    Args:
        max_actions_per_step (int): Maximum number of actions allowed per step
        action_sep (str): Separator between actions
        description (str): Description of response format
        
    Returns:
        str: Base prompt
    """
    return f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
{description}"""

def create_format_prompt(prompt_type, max_actions_per_step, action_sep, state_keys, add_example=True):
    """
    General format prompt creator that handles all prompt types.
    
    Args:
        prompt_type (str): Type of prompt format ("free_think", "no_think", "grounding", etc.)
        max_actions_per_step (int): Maximum number of actions allowed per step
        action_sep (str): Separator between actions
        state_keys (list): List of object states to track/predict
        add_example (bool): Whether to add an example
        
    Returns:
        str: The formatted prompt
    """
    # Format descriptions and response formats for each prompt type
    format_configs = {
        "free_think": {
            "description": "You should first give your thought process, and then your answer.",
            "response_format": "<think>...</think><answer>...</answer>",
            "example_format": "<think>I need to pick the {target_object} at (100,100,40) first and place it at (80,100,50)</think><answer>pick(100,100,40){action_sep}place(80,100,50)</answer>"
        },
        "no_think": {
            "description": "You should provide only your answer.",
            "response_format": "<answer>...</answer>",
            "example_format": "<answer>pick(100,100,40){action_sep}place(80,100,50)</answer>"
        },
        "grounding": {
            "description": "You should first give the current state, then your thought process, and finally your answer.\nThe state should be in the format of {state_format}",
            "response_format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
            "example_format": "<think><observation>{init_state_example}</observation><reasoning>I need to pick the {target_object} at (100,100,40) and place it at (80,100,50)</reasoning></think><answer>pick(100,100,40){action_sep}place(80,100,50)</answer>"
        },
        "worldmodeling": {
            "description": "You should give your thought process, then try to predict the next state, and finally your answer.\nThe state should be in the format of {state_format}",
            "response_format": "<think><reasoning>...</reasoning><description>...</description></think><answer>...</answer>",
            "example_format": "<think><reasoning>I need to pick the {target_object} at (100,100,40) and place it at (80,100,50)</reasoning></think><description>{next_state_example}</description><answer>pick(100,100,40){action_sep}place(80,100,50)</answer>"
        },
        "grounding_worldmodeling": {
            "description": "You should first describe the current state, then your thought process, then predict the next state, and finally your answer.\nThe state should be in the format of {state_format}",
            "response_format": "<think><observation>...</observation><reasoning>...</reasoning><description>...</description></think><answer>...</answer>",
            "example_format": "<think><observation>{init_state_example}</observation><reasoning>I need to pick the {target_object} at (100,100,40) and place it at (80,100,50)</reasoning><prediction>{next_state_example}</prediction></think><answer>pick(100,100,40){action_sep}place(80,100,50)</answer>"
        }
    }
    
    # Get configuration for the requested prompt type
    config = format_configs.get(prompt_type)
    if not config:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    # Create state examples if needed
    state_format, init_state_example, next_state_example, target_object = ({}, {}, {}, "") 
    if prompt_type in ["grounding", "worldmodeling", "grounding_worldmodeling"]:
        state_format, init_state_example, next_state_example, target_object = create_state_examples(state_keys)
    
    # Format the description
    description = config["description"].format(state_format=state_format)
    
    # Create base prompt
    base_prompt = create_base_prompt(
        max_actions_per_step, 
        action_sep, 
        description + f"\nYour response should be in the format of:\n{config['response_format']}"
    )
    
    # Add example if requested
    if add_example:
        example = config["example_format"].format(
            init_state_example=init_state_example,
            next_state_example=next_state_example,
            target_object=target_object.replace('_position', ''),
            action_sep=action_sep
        )
        return base_prompt + '\n' + f"e.g. {example}"
    
    return base_prompt

# Define individual format functions that use the general function
def free_think_format_prompt(max_actions_per_step, action_sep, state_keys, add_example=True):
    return create_format_prompt("free_think", max_actions_per_step, action_sep, state_keys, add_example)

def no_think_format_prompt(max_actions_per_step, action_sep, state_keys, add_example=True):
    return create_format_prompt("no_think", max_actions_per_step, action_sep, state_keys, add_example)

def grounding_format_prompt(max_actions_per_step, action_sep, state_keys, add_example=True):
    return create_format_prompt("grounding", max_actions_per_step, action_sep, state_keys, add_example)

def worldmodeling_format_prompt(max_actions_per_step, action_sep, state_keys, add_example=True):
    return create_format_prompt("worldmodeling", max_actions_per_step, action_sep, state_keys, add_example)

def grounding_worldmodeling_format_prompt(max_actions_per_step, action_sep, state_keys, add_example=True):
    return create_format_prompt("grounding_worldmodeling", max_actions_per_step, action_sep, state_keys, add_example)

# Dictionary mapping format names to their corresponding functions
format_prompt = {
    "free_think": free_think_format_prompt,
    "no_think": no_think_format_prompt,
    "grounding": grounding_format_prompt,
    "worldmodeling": worldmodeling_format_prompt,
    "grounding_worldmodeling": grounding_worldmodeling_format_prompt
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
