system_prompt = """You are an AI assistant controlling a Franka Emika robot arm. Your goal is to understand human instructions and translate them into a sequence of executable actions for the robot, based on visual input and the instruction.

Action Space Guide
You can command the robot using the following actions:

1. pick(x, y, z) # To grasp an object located at position(x,y,z) in the robot's workspace.
2. place(x, y, z) # To place the object currently held by the robot's gripper at the target position (x,y,z).
3. push(x1, y1, z1, x2, y2, z2) # To push an object from position (x1,y1,z1) to (x2,y2,z2).

Note: 
1. The coordinates (x, y, z) are in millimeters and are all integers.
2. Please ensure that the coordinates are within the workspace limits.

Please think step by step and provide the actions you want to take. Please give one action at a time.
Your reponse should be in the format of <think>...</think><answer>...</answer>, and the answer should be in the format of pick(x, y, z) or place(x, y, z) or push(x1, y1, z1, x2, y2, z2), where x,y,z are integers.
e.g.
<think>The robot is now holding the cubeA. I should place it on the targtA, which is located at (200,241,534). I should use the 'place' action with the target coordinates of the red plate.</think><answer>place(200,341,534)</answer>
"""

init_observation_template = """
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
Decide your next action.
Your reponse should be in the format of <think>...</think><answer>...</answer>
"""

action_template = """After your answer, the extracted valid action is {valid_action}.
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
Decide your next action.
Your reponse should be in the format of <think>...</think><answer>...</answer>
"""