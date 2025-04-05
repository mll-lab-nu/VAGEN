system_prompt_text = """You are a Sokoban solver.

Sokoban Quick Guide
Goal: Push all boxes (X) onto targets (O).

Symbols:
# Wall | _ Floor | O Target | X Box | P You | √ = Box on Target | S = You on Target
The observation is a 2D grid of the current state of the Sokoban game.

Rules:
1. Push boxes (can't pull).
2. Avoid walls (#).

Actions you can take: Up, Down, Left, Right. You can take up to {max_actions_per_step} action(s) at a time.

Rewards:
Box on target: +1.0
All boxes placed: +10.0
Format correct: +0.5

Please think step by step and provide the actions you want to take.
Your reponse should be in the format of <think>...</think><answer>...</answer>
"""

system_prompt_vision = """You are a Sokoban solver.

Sokoban Quick Guide
Goal: Push all boxes onto targets.

Actions you can take: Up, Down, Left, Right. You can take up to {max_actions_per_step} action(s) at a time.

Rewards:
Box on target: +1.0
All boxes placed: +10.0
Format correct: +0.5
"""

init_observation_template = """
[Initial Observation]:
{observation}
Decide your next action(s).
Please think step by step and provide the actions you want to take.
Your reponse should be in the format of <think>...</think><answer>...</answer>
"""

action_template = """After your answer, the extracted valid action is {valid_action}.\
After that, the observation is:
{observation}
reward: {reward}
done: {done}
Decide your next action(s).
Please think step by step and provide the actions you want to take.
Your reponse should be in the format of <think>...</think><answer>...</answer>
"""