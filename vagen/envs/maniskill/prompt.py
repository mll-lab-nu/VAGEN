"""Prompt templates for the ManiSkill primitive-skill environment.

Adapted for VAGEN's prompt format system (free_think / wm / free_wm).
"""


SYSTEM_PROMPT = """You are an AI assistant controlling a Franka Emika robot arm. Your goal is to understand human instructions and translate them into a sequence of executable actions for the robot, based on visual input and the instruction.

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
Anwer: pick(-44,100,20)|place(100,-43,50)"""


FORMAT_CONFIGS = {
    "free_think": {
        "format": "<think>...</think><answer>...</answer>",
        "description": "You should first give your thought process, and then your answer.",
        "example": (
            "<think>I need to pick the red_cube at (10,20,30) and place it "
            "on the green_block at (50,60,40).</think>"
            "<answer>pick(10,20,30){sep}place(50,60,70)</answer>"
        ),
    },
    "wm": {
        "format": (
            "<observation>...</observation>"
            "<think>...</think>"
            "<answer>...</answer>"
            "<prediction>...</prediction>"
        ),
        "description": (
            "You should first describe your observation, then reason about it, "
            "provide your actions, and predict the next state."
        ),
        "example": (
            "<observation>I see a red cube at (100,100,40) and a green cube "
            "at (200,200,60).</observation>"
            "<think>I need to pick the red cube and place it on top of the "
            "green cube.</think>"
            "<answer>pick(100,100,40){sep}place(200,200,100)</answer>"
            "<prediction>After executing, the red cube will be at "
            "(200,200,100).</prediction>"
        ),
    },
    "free_wm": {
        "format": (
            "<observation>...</observation> ... "
            "<answer>...</answer> ... "
            "<prediction>...</prediction>"
        ),
        "description": (
            "You should describe your observation, reason freely, "
            "provide your actions, and predict the result."
        ),
        "example": (
            "<observation>Red cube at (100,100,40), green cube at "
            "(200,200,60).</observation>\n"
            "I should stack red on green.\n"
            "<answer>pick(100,100,40){sep}place(200,200,100)</answer>\n"
            "Red cube should now be on top.\n"
            "<prediction>Red cube at (200,200,100).</prediction>"
        ),
    },
}


def system_prompt() -> str:
    return SYSTEM_PROMPT


def format_prompt(
    prompt_format: str = "free_think",
    max_actions_per_step: int = 2,
    action_sep: str = "|",
    add_example: bool = True,
) -> str:
    config = FORMAT_CONFIGS.get(prompt_format, FORMAT_CONFIGS["free_think"])
    text = (
        f"You can take up to {max_actions_per_step} action(s) at a time, "
        f"separated by {action_sep}.\n"
        f"{config['description']}\n"
        f"Your response should be in the format of:\n"
        f"{config['format']}"
    )
    if add_example:
        example = config["example"].replace("{sep}", action_sep)
        text += "\n\ne.g. " + example
    return text


def init_observation_template(
    observation: str,
    instruction: str,
    x_workspace,
    y_workspace,
    z_workspace,
    object_positions: str,
    other_information: str,
) -> str:
    return f"""[Initial Observation]:
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


def action_template(
    valid_actions,
    observation: str,
    instruction: str,
    x_workspace,
    y_workspace,
    z_workspace,
    object_positions: str,
    other_information: str,
) -> str:
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
