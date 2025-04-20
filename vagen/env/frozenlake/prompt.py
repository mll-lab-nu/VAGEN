system_prompt_text = """You are a FrozenLake solver.

FrozenLake Quick Guide
Goal: Reach the goal (G).

Symbols:
_ Frozen | O Hole | G Goal | P Player | X Player fell into hole | √ Player on goal

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Actions you can take: Left, Down, Right, Up. 
You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
Left: move left to the cell to the left.
Down: move down to the cell below.
Right: move right to the cell to the right.
Up: move up to the cell above.

Rewards:
Fall into hole: 0
Reach goal: +10.0
Format correct: +0.5

Please think step by step and provide the actions you want to take.
Your response should be in the format of <think>...</think><answer>...</answer>
"""

system_prompt_vision = """You are a FrozenLake solver.

FrozenLake Quick Guide
Goal: Reach the goal (G).

Symbols:
Light blue: Frozen surface | Black: Hole | Green: Goal | Red: Player

Rules:
1. Avoid falling into holes.
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Actions you can take: Left, Down, Right, Up. 
You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.

Rewards:
Fall into hole: 0
Reach goal: +10.0
Format correct: +0.5

Please think step by step and provide the actions you want to take.
Your response should be in the format of <think>...</think><answer>...</answer>
"""

init_observation_template = """
[Initial Observation]:
{observation}
Decide your next action(s).
Please think step by step and provide the actions you want to take.
Your response should be in the format of <think>...</think><answer>...</answer>
"""

action_template = """After your answer, the extracted valid action is {valid_action}.
After that, the observation is:
{observation}
reward: {reward}
done: {done}
Decide your next action(s).
Please think step by step and provide the actions you want to take.
Your response should be in the format of <think>...</think><answer>...</answer>
"""