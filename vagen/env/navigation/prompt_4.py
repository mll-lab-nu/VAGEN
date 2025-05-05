def system_prompt(**kwargs):
    
    if kwargs.get("format", "default") in ["free_think", "default"]:
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
    elif kwargs.get("format", "default") == "grounding":
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
    elif kwargs.get("format", "default") == "worldmodeling":
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
    elif kwargs.get("format", "default") == "grounding_worldmodeling":
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
    elif kwargs.get("format", "default") == "no_think":
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
    return """You are a home robot and perform navigation tasks according to instructions.
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
2. If you find yourself seems to be stuck, you can lookdown to see if there's any object above or below you, you can also rotate to see if there's any object behind you."""+'\n' + example 

def init_observation_template(observation, instruction):
    return f"""[Initial Observation]:
{observation}
Human Instruction: {instruction}
Decide your next action(s)."""

def action_template(valid_action, observation, reward, done, instruction,env_feedback):
    return f"""After your answer, the extracted valid action is {valid_action}.
The environment feedback is: {env_feedback}
reward: {reward}
done: {done}
After that, the observation is:
{observation}
Human Instruction: {instruction}
Decide your next action(s)."""

def free_think_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your thought process, and then your answer. 
Your response should be in the format of:
<think>...</think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think>I can see from the sight the target object is right in the top left of me, I will move forward, then move left to access it.</think><answer>moveahead{action_sep}moveahead{action_sep}moveahead{action_sep}moveleft{action_sep}moveleft</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def no_think_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should provide only your answer.
Your response should be in the format of:
<answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <answer>moveahead{action_sep}moveahead{action_sep}moveahead{action_sep}moveleft{action_sep}moveleft</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def grounding_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your thought process with your observation and reasoning, and finally your answer.
The observation should be described in detail about what you see in the environment.
Your response should be in the format of:
<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think><observation>I am in a living room. There is a couch to my left, a TV in front of me, and a doorway to the kitchen on my right. The target object, a vase, appears to be on a shelf near the kitchen doorway.</observation><reasoning>I need to move toward the kitchen doorway to reach the vase. I'll move forward to get closer to the center of the room, then turn right and move toward the kitchen.</reasoning></think><answer>moveahead{action_sep}moveahead{action_sep}rotateright{action_sep}moveahead{action_sep}moveahead</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def worldmodeling_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your thought process with reasoning and prediction of next state,  then your answer.
The prediction should describe what you expect to see after your actions are executed.
Your response should be in the format of:
<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think><reasoning>I can see the kitchen doorway to my right, and I need to go there to find the refrigerator. I'll turn right and move forward.</reasoning><prediction>I am now in the kitchen doorway. In front of me is the kitchen counter with a sink. To the left I can see a refrigerator against the wall. There's a kitchen island in the center of the room.</prediction></think><answer>rotateright{action_sep}moveahead{action_sep}moveahead</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

def grounding_worldmodeling_format_prompt(max_actions_per_step, action_sep, add_example=True):
    base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
You should first give your thought process with the your observation and reasoning, then predict next state, and finally the answer.
Both the observation and prediction should describe what you see or expect to see in the environment.
Your response should be in the format of:
<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>"""
    
    if add_example:
        example = f"""e.g. <think><observation>I am at the entrance of a bedroom. There is a bed to the left, a desk with a lamp on the right, and a closet straight ahead. The target object, a book, appears to be on the desk.</observation><reasoning>I need to move toward the desk to reach the book. I'll turn right and move forward.</reasoning><prediction>I am now standing in front of the desk. The desk has a lamp, a computer, and several books on it. The target book is within reach on the right side of the desk.</prediction></think><answer>rotateright{action_sep}moveahead{action_sep}moveahead</answer>"""
        return base_prompt + '\n' + example
    return base_prompt

# Dictionary mapping format names to their corresponding functions
format_prompt = {
    "free_think": free_think_format_prompt,
    "no_think": no_think_format_prompt,
    "grounding": grounding_format_prompt,
    "worldmodeling": worldmodeling_format_prompt,
    "grounding_worldmodeling": grounding_worldmodeling_format_prompt
}