# This FORMAT_CONFIGS is for the robot navigation task,
# structured like the FORMAT_CONFIGS in your first (FrozenLake) example.
FORMAT_CONFIGS = {
    "free_think": {
        "description": "You should first give your thought process, and then your answer.",
        "format": "<think>...</think><answer>...</answer>",
        "example": """<think>The target object is directly ahead and to the right of me. There are no obstacles blocking my path to reach it. I'll move forward multiple times and then right to approach the target.</think><answer>moveahead{action_sep}moveahead{action_sep}moveahead{action_sep}moveright{action_sep}moveahead</answer>"""
    },
    
    "no_think": {
        "description": "You should provide only your answer.",
        "format": "<answer>...</answer>",
        "example": """<answer>moveahead{action_sep}moveahead{action_sep}moveahead{action_sep}moveright{action_sep}moveahead</answer>"""
    },
    
    "grounding": {
        "description": "You should first give your thought process with your observation and reasoning, and finally your answer.\nThe observation should be described in detail about what you see in the environment.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "example": """<think><observation>The target vase is to my right. There's a clear path with no obstacles between me and the target.</observation><reasoning>Since the target is to my right and there are no obstacles, I'll move forward multiple times and then right to reach it.</reasoning></think><answer>moveahead{action_sep}moveahead{action_sep}moveahead{action_sep}moveright{action_sep}moveahead</answer>"""
    },
    
    "worldmodeling": {
        "description": "You should first give your thought process with reasoning and prediction of next state, then your answer.\nThe prediction should describe what you expect to see after your actions are executed.",
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": """<think><reasoning>The refrigerator is to my right. There are no obstacles in my path. I'll move forward multiple times and then right to reach it.</reasoning><prediction>After my movements, the refrigerator should be directly in front of me with no obstacles between us.</prediction></think><answer>moveahead{action_sep}moveahead{action_sep}moveahead{action_sep}rotateright{action_sep}moveahead</answer>"""
    },
    
    "grounding_worldmodeling": {
        "description": "You should first give your thought process with the your observation, reasoning, and prediction of next state, then your answer.\nBoth the observation and prediction should describe what you see or expect to see in the environment.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": """<think><observation>The target book is on the desk to my right. There are no obstacles blocking my path to the desk.</observation><reasoning>Since the target is to my right with no obstacles, I'll move forward multiple times then right to reach it.</reasoning><prediction>After my movements, the book should be directly in front of me with no obstacles between us.</prediction></think><answer>moveahead{action_sep}moveahead{action_sep}moveahead{action_sep}moveright{action_sep}moveahead</answer>"""
    }
}

# system_prompt function as per your second code block (robot navigation)
def system_prompt(**kwargs):
    example = "" # Default empty example
    # Internally uses kwargs.get("format"), as in your original code
    selected_format = kwargs.get("format", "default")

    if selected_format in ["free_think", "default"]:
        example=f"""Example:
Round 1:
image_1
<think>I can see the garbage can in the upper left corner of the image, next to the kitchen sink. To move there, we can go forward-left, but since there's a kitchen counter directly ahead, we should go left first. Following the strategy, I can go by first moving leftward.</think>
<answer>moveleft, moveleft</answer>
Round 2:
Env_feedback: Last action is executed successfully.
image_2
<think>From the secene, I see that by moving leftward, we are getting closer to the garbage can. Now, the garbage can is in front of me, slightly to the left. And there's a large area ahead of us. Following the strategy, I can go by first moving forward then moving leftward.</think>
<answer>moveahead, moveahead,moveahead,moveleft</answer>
Round 3:
Env_feedback: Last action is executed successfully.
image_3
<think>From the image we can see the garbage can is very close to us, still to our front-left. Moving leftward might be blocked but i can see that there is still space in front of me to get closer to the garbage can. Following the strategy, we can take about two steps forward then one step left to reach the garbage can.</think>
<answer>moveahead, moveahead,moveleft</answer>
Round 4:
Env_feedback: Success"""
    elif selected_format == "grounding":
        example=f"""Example:
Round 1:
image_1
<think><observation>There is a garbage can in the upper left corner of the image, next to the kitchen sink. To move there, we can go forward-left, but since there's a kitchen counter directly ahead, we should go left first.</observation><reasoning>Following the strategy, I can go by first moving leftward.</reasoning></think>
<answer>moveleft, moveleft</answer>
Round 2:
Env_feedback: Last action is executed successfully.
image_2
<think><observation>From the secene, I see that by moving leftward, we are getting closer to the garbage can. Now, the garbage can is in front of me, slightly to the left. And there's a large area ahead of us.</observation><reasoning>Following the strategy, I can go by first moving forward then moving leftward.</reasoning></think>
<answer>moveahead, moveahead,moveahead,moveleft</answer>
Round 3:
Env_feedback: Last action is executed successfully.
image_3
<think><observation>From the image we can see the garbage can is very close to us, still to our front-left. Moving leftward might be blocked but i can see that there is still space in front of me to get closer to the garbage can.</observation><reasoning>Following the strategy, we can take about two steps forward then one step left to reach the garbage can.</reasoning></think>
<answer>moveahead, moveahead,moveleft</answer>
Round 4:
Env_feedback: Success"""
    elif selected_format == "worldmodeling":
        example=f"""Example:
Round 1:
image_1
<think><reasoning>I can see the garbage can in the upper left corner of the image, next to the kitchen sink. To move there, we can go forward-left, but since there's a kitchen counter directly ahead, we should go left first.</reasoning><prediction>I will be infront of the garbage</prediction></think>
<answer>moveleft, moveleft</answer>
Round 2:
Env_feedback: Last action is executed successfully.
image_2
<think><reasoning>From the secene, I see that by moving leftward, we are getting closer to the garbage can. Now, the garbage can is in front of me, slightly to the left. And there's a large area ahead of us.</reasoning><prediction>I will be closer to the garbage</prediction></think>
<answer>moveahead, moveahead,moveahead,moveleft</answer>
Round 3:
Env_feedback: Last action is executed successfully.
image_3
<think><reasoning>From the image we can see the garbage can is very close to us, still to our front-left. Moving leftward might be blocked but i can see that there is still space in front of me to get closer to the garbage can.</reasoning><prediction>I will reach the garbage</prediction></think>
<answer>moveahead, moveahead,moveleft</answer>
Round 4:
Env_feedback: Success"""
    elif selected_format == "grounding_worldmodeling":
        example=f"""Example:
Round 1:
image_1
<think><observation>There is a garbage can in the upper left corner of the image, next to the kitchen sink. To move there, we can go forward-left, but since there's a kitchen counter directly ahead, we should go left first.</observation><reasoning>Following the strategy, I can go by first moving leftward.</reasoning><prediction>I will be infront of the garbage</prediction></think>
<answer>moveleft, moveleft</answer>
Round 2:
Env_feedback: Last action is executed successfully.
image_2
<think><observation>From the secene, I see that by moving leftward, we are getting closer to the garbage can. Now, the garbage can is in front of me, slightly to the left. And there's a large area ahead of us.</observation><reasoning>Following the strategy, I can go by first moving forward then moving leftward.</reasoning><prediction>I will be closer to the garbage</prediction></think>
<answer>moveahead, moveahead,moveahead,moveleft</answer>
Round 3:
Env_feedback: Last action is executed successfully.
image_3
<think><observation>From the image we can see the garbage can is very close to us, still to our front-left. Moving leftward might be blocked but i can see that there is still space in front of me to get closer to the garbage can.</observation><reasoning>Following the strategy, we can take about two steps forward then one step left to reach the garbage can.</reasoning><prediction>I will reach the garbage</prediction></think>
<answer>moveahead, moveahead,moveleft</answer>
Round 4:
Env_feedback: Success"""
    elif selected_format == "no_think":
        example=f"""Example:
Round 1:
image_1
<answer>moveleft, moveleft</answer>
Round 2:
Env_feedback: Last action is executed successfully.
image_2
<answer>moveahead, moveahead,moveahead,moveleft</answer>
Round 3:
Env_feedback: Last action is executed successfully.
image_3
<answer>moveahead, moveahead,moveleft</answer>
Round 4:
Env_feedback: Success"""
        
    base_prompt_text = """You are a home robot and perform navigation tasks according to instructions.
Actions you can take: moveahead, moveback, moveright, moveleft, rotateright, rotateleft, lookup, lookdown. 
moveahead: Move forward by 0.4 meter
moveback: Move backward by 0.4 meter
moveright: Move rightward by 0.4 meter
moveleft: Move leftward by 0.4 meter
rotateright: Rotate to the right by 90 degrees
rotateleft: Rotate to the left by 90 degrees
lookup: Tilt the camera upward by 30 degrees
lookdown: Tilt the camera downward by 30 degrees
Rewards:
Format correct: +0.5
Achieve the human instruction: +10.0
The instruction will be provided with each observation. Look at the image carefully and navigate to complete the instruction.
Hints:
1. You can take multiple actions at a time, in most cases, if you find the target object is far away from you, you can call moveahead, moveleft and move right multiple times.
2. If you find yourself seems to be stuck, you can lookdown to see if there's any object above or below you, you can also rotate to see if there's any object behind you.
3. You must clearly specify the target object's position relative to you using directional terms (ahead, left, right).
4. When describing your navigation plan, use the same directional terms as the available actions (moveahead, moveleft, moveright).
"""
    return base_prompt_text + '\n' + example

# init_observation_template and action_template from your second code block
def init_observation_template(**kwargs):
    observation = kwargs.get("observation", "No observation provided.")
    instruction = kwargs.get("instruction", "No instruction provided.")
    return f"""[Initial Observation]:
{observation}
Human Instruction: {instruction}
Decide your next action(s)."""

def action_template(**kwargs):
    observation = kwargs.get("observation", "No observation provided.")
    instruction = kwargs.get("instruction", "No instruction provided.")
    valid_action = kwargs.get("valid_action", "No valid action provided.")
    env_feedback = kwargs.get("env_feedback", "No environment feedback provided.")
    reward = kwargs.get("reward", "No reward provided.")
    done = kwargs.get("done", "No done status provided.")
    return f"""After your answer, the extracted valid action is {valid_action}.
The environment feedback is: {env_feedback}
reward: {reward}
done: {done}
After that, the observation is:
{observation}
Human Instruction: {instruction}
Decide your next action(s)."""

# format_prompt_generator function, similar to your first (FrozenLake) example
def format_prompt_generator(format_type):
    """
    Generates a prompt function for the specified robot navigation format type.
    This returned function creates the per-turn instruction for the LLM.
    """
    def prompt_function(**kwargs):
        """
        Generate a prompt for the specified format for the robot navigation task.
        
        Args:
            max_actions_per_step (int): Max actions. Defaults to 5 (common for robot).
            action_sep (str): Separator. Defaults to ',' (common for robot).
            add_example (bool): Whether to add an example. Defaults to True.
            
        Returns:
            str: The formatted prompt.
        """
        # Defaults suitable for the robot navigation task
        max_actions_per_step = kwargs.get("max_actions_per_step", 5)
        action_sep = kwargs.get("action_sep", ",")
        add_example = kwargs.get("add_example", True) # Default to True as per robot examples
        
        if format_type not in FORMAT_CONFIGS:
            raise ValueError(f"Unknown format_type: {format_type}")
        config = FORMAT_CONFIGS[format_type]
        
        base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'.
{config["description"]}"""
        
        if "additional_info" in config: # In case it's added to FORMAT_CONFIGS later
            base_prompt += f"\n{config['additional_info']}"
        
        base_prompt += f"""
Your response should be in the format of:
{config["format"]}"""
        
        if add_example:
            # The 'e.g.' is already part of the example string in this FORMAT_CONFIGS
            example_text = config["example"].format(action_sep=action_sep)
            return base_prompt + '\n' + f"e.g. {example_text}"
        
        return base_prompt
    
    return prompt_function

# format_prompt dictionary, as in your first (FrozenLake) example
format_prompt = {
    ft: format_prompt_generator(ft) 
    for ft in FORMAT_CONFIGS  # Iterate directly over keys in FORMAT_CONFIGS
}


if __name__ == "__main__":
    # Example usage
    max_actions_per_step = 2
    action_sep = "|"
    
    for key, func in format_prompt.items():
        print(f"{key} format prompt:")
        print(func(max_actions_per_step=max_actions_per_step, action_sep=action_sep))
        print("\n" + "="*50 + "\n")