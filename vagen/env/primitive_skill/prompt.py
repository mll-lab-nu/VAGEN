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

def init_observation_template(**kwargs):
    object_names = kwargs.get("object_names", None)
    observation = kwargs.get("observation")
    instruction = kwargs.get("instruction")
    x_workspace = kwargs.get("x_workspace")
    y_workspace = kwargs.get("y_workspace")
    z_workspace = kwargs.get("z_workspace")
    object_positions = kwargs.get("object_positions")
    other_information = kwargs.get("other_information")
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

def action_template(**kwargs):
    object_names = kwargs.get("object_names", None)
    observation = kwargs.get("observation")
    instruction = kwargs.get("instruction")
    x_workspace = kwargs.get("x_workspace")
    y_workspace = kwargs.get("y_workspace")
    z_workspace = kwargs.get("z_workspace")
    object_positions = kwargs.get("object_positions")
    valid_actions = kwargs.get("valid_actions")
    other_information = kwargs.get("other_information")
    
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

# --- FORMAT_CONFIGS Definition with Fixed Examples ---
FORMAT_CONFIGS = {
    "free_think": {
        "description": "You should first give your thought process, and then your answer.",
        "format": "<think>...</think><answer>...</answer>",
        "example": "e.g. <think>I need to pick the red_cube_pos at (10,20,30) and place it on the green_block_pos at (50,60,40).</think><answer>pick(10,20,30){action_sep}place(50,60,70)</answer>"
    },
    "no_think": {
        "description": "You should provide only your answer.",
        "format": "<answer>...</answer>",
        "example": "e.g. <answer>pick(10,20,30){action_sep}place(50,60,70)</answer>"
    },
    "grounding": {
        "description": "You should first give your thought process with observation and reasoning, and then your answer.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "example": "<think><observation>I see a red cube at (100,100,40) and a green cube at (200,200,60).</observation><reasoning>I need to pick the red cube and place it on top of the green cube</reasoning></think><answer>pick(100,100,40){action_sep}place(200,200,100)</answer>"
    },
    "worldmodeling": {
        "description": "You should first give your thought process with reasoning and prediction of next state, and then your answer.",
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": "<think><reasoning>I need to pick the red cube at (100,100,40) and place it at (80,100,60)</reasoning><prediction>After executing this action, the red cube will be at (80,100,60)</prediction></think><answer>pick(100,100,40){action_sep}place(80,100,60)</answer>"
    },
    "grounding_worldmodeling": {
        "description": "You should first give your thought process with observation, reasoning, and prediction, and then your answer.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": "<think><observation>I see a red cube at (100,100,40) and a green cube at (200,200,60).</observation><reasoning>I need to pick the red cube and place it on top of the green cube</reasoning><prediction>After executing this action, the red cube will be at position (200,200,100), stacked on top of the green cube at (200,200,60)</prediction></think><answer>pick(100,100,40){action_sep}place(200,200,100)</answer>"
    },
    "grounding_structured": {
        "description": "You should first give your thought process with observation and reasoning, and then your answer.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "additional_info": "The observation should be in the format of {state_format}",
        "example": "<think><observation>{{'red_cube':(100,100,40),'green_cube':(200,200,60)}}</observation><reasoning>I need to pick the red cube and place it on top of the green cube</reasoning></think><answer>pick(100,100,40){action_sep}place(200,200,100)</answer>"
    },
    "worldmodeling_structured": {
        "description": "You should first give your thought process with reasoning and prediction of next state, and then your answer.",
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "additional_info": "The prediction should be in the format of {state_format}",
        "example": "<think><reasoning>I need to pick the red cube and place it on top of the green cube</reasoning><prediction>{{'red_cube':(200,200,100),'green_cube':(200,200,60)}}</prediction></think><answer>pick(100,100,40){action_sep}place(200,200,100)</answer>"
    },
    "grounding_worldmodeling_structured": {
        "description": "You should first give your thought process with observation, reasoning, and prediction, and then your answer.",
        "additional_info": "The observation and prediction should be in the format of {state_format}",
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": "<think><observation>{{'red_cube':(100,100,40),'green_cube':(200,200,60)}}</observation><reasoning>I need to pick the red cube and place it on top of the green cube</reasoning><prediction>{{'red_cube':(200,200,100),'green_cube':(200,200,60)}}</prediction></think><answer>pick(100,100,40){action_sep}place(200,200,100)</answer>"
    }
}

# --- Generic Format Prompt Generator ---
def format_prompt_generator(format_type):
    def prompt_function(**kwargs):
        max_actions_per_step = kwargs.get("max_actions_per_step", 1)
        action_sep = kwargs.get("action_sep", "|")
        add_example = kwargs.get("add_example", True)
        config = FORMAT_CONFIGS[format_type]
        state_keys = kwargs.get("state_keys", None)
        state_format = str({key: "(x,y,z)" for key in state_keys})
        
        # Build the base prompt text
        base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
{config["description"]}"""
        
        
        # Add response format instruction
        base_prompt += f"""
Your response should be in the format of:
{config["format"]}"""
        
        # Add additional information if available
        if "additional_info" in config:
            base_prompt += f"\n{config['additional_info'].format(state_format=state_format)}"
            
        # Add example if requested
        if add_example:
            example = config["example"].format(action_sep=action_sep)
            return base_prompt + '\n' + f"e.g. {example}"
        
        return base_prompt
    
    return prompt_function

# --- Populate format_prompt dictionary ---
format_prompt = {
    ft: format_prompt_generator(ft) 
    for ft in FORMAT_CONFIGS
}


if __name__ == "__main__":
    # Example usage
    max_actions_per_step = 2
    action_sep = "|"
    state_keys = ["red_cube", "green_cube", "yellow_cube"]
    for key, func in format_prompt.items():
        print(f"{key} format prompt:")
        print(func(max_actions_per_step=max_actions_per_step, action_sep=action_sep,state_keys=state_keys, add_example=True))
        print("\n" + "="*50 + "\n")