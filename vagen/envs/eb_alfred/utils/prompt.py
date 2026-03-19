from typing import List, Optional


# ERA-aligned task examples teaching ALFRED-specific mechanics
# (cleaning, heating, slicing, storage, object placement)
TASK_EXAMPLES = [
    {
        "task": "Pick up the alarm clock and turn on the lamp",
        "think": "I need to find the alarm clock, pick it up, then find the desk lamp and turn it on.",
        "actions": ["find a AlarmClock", "pick up the AlarmClock", "find a DeskLamp", "turn on the DeskLamp"],
    },
    {
        "task": "Set the box on the table",
        "think": "I need to find the box, pick it up, then find the dining table and put it down.",
        "actions": ["find a Box", "pick up the Box", "find a DiningTable", "put down the object in hand"],
    },
    {
        "task": "Move the towel on the hanger into the cabinet",
        "think": "I need to find the hand towel, pick it up, find a cabinet, open it, put the towel inside, and close the cabinet.",
        "actions": ["find a HandTowel", "pick up the HandTowel", "find a Cabinet", "open the Cabinet", "put down the object in hand", "close the Cabinet"],
    },
    {
        "task": "Put a clean pan in the refrigerator",
        "think": "To clean the pan, I need to: pick it up, put it in the sink, turn on the faucet, turn off the faucet, then pick up the now-clean pan and put it in the fridge.",
        "actions": ["find a Pan", "pick up the Pan", "find a Sink", "put down the object in hand", "find a Faucet", "turn on the Faucet", "turn off the Faucet", "find a Pan", "pick up the Pan", "find a Fridge", "open the Fridge", "put down the object in hand", "close the Fridge"],
    },
    {
        "task": "Slice a loaf of bread put a slice on the counter",
        "think": "To slice bread I need a knife first. Pick up knife, find bread, slice it. Put knife away, then pick up a bread slice and place it on the counter.",
        "actions": ["find a Knife", "pick up the Knife", "find a Bread", "slice the Bread", "find a CounterTop", "put down the object in hand", "find a Bread", "pick up the Bread", "find a CounterTop", "put down the object in hand"],
    },
]


def system_prompt(task_instruction: Optional[str] = None, action_list: Optional[List[str]] = None, add_task_examples: bool = True):
    """
    System prompt for EB-ALFRED household robot tasks.

    When task_instruction and action_list are provided (after reset),
    includes the per-episode task and available actions so that
    no-concat mode always has access to them.
    """
    base = """You are a robot operating in a home. Given a task, you must accomplish the task using a defined set of actions to achieve the desired outcome.

## Action Descriptions and Validity Rules
- Find: Parameterized by the name of the receptacle to navigate to. Always valid if the object exists in the scene.
- Pick up: Parameterized by the name of the object to pick. Only valid if close to the object, not already holding something, and the object is not in a closed receptacle.
- Put down: Parameterized by the name of the object to put down to a nearby receptacle. Only valid if holding an object.
- Drop: Parameterized by the name of the object to put down. Different from 'put down' as this does not guarantee the held object will be put into a specified receptacle.
- Open: Parameterized by the name of the receptacle to open. Only valid if the receptacle is closed and close to the receptacle.
- Close: Parameterized by the name of the receptacle to close. Only valid if the receptacle is open and close to the receptacle.
- Turn on: Parameterized by the name of the object to turn on. Only valid if the object is turned off and close to the object.
- Turn off: Parameterized by the name of the object to turn off. Only valid if the object is turned on and close to the object.
- Slice: Parameterized by the name of the object to slice. Only valid if the object is sliceable and close to the object.

## Guidelines
1. Output a plan of actions. Each plan should include no more than 20 actions.
2. Always locate an object using 'find' before interacting with it.
3. Make sure to match the action name and its corresponding action id in the output. Use 'put down' rather than 'drop' to place objects in specific receptacles.
4. Do not repeat the same failed action sequence. Try to modify the action sequence because previous actions did not lead to success.
5. Objects may have multiple instances (e.g., Cabinet_2, Cabinet_3). Explore different instances if needed.
6. Use environment feedback to refine your plan. If an action fails, reflect on the reason and adjust accordingly."""

    if add_task_examples and TASK_EXAMPLES:
        base += "\n\n## Task Examples"
        for i, ex in enumerate(TASK_EXAMPLES):
            if action_list is not None:
                # Build action-to-id lookup from the current episode's action list
                name_to_id = {a.lower(): idx for idx, a in enumerate(action_list)}
                parts = []
                for a in ex["actions"]:
                    aid = name_to_id.get(a.lower())
                    if aid is not None:
                        parts.append(f"[{aid}, {a}]")
                    else:
                        parts.append(a)
                actions_str = "| ".join(parts)
            else:
                actions_str = "| ".join(ex["actions"])
            base += f"\n\nExample {i+1}: {ex['task']}\n<think>{ex['think']}</think>\n<answer>{actions_str}</answer>"

    if task_instruction is not None:
        base += f"\n\n## Current Task\n{task_instruction}"

    if action_list is not None:
        actions_str = "\n".join(f"[{i}, {a}]" for i, a in enumerate(action_list))
        base += f"\n\n## Available Actions (0~{len(action_list) - 1})\n{actions_str}"

    return base


def init_observation_template(img_str):
    """Template for initial observation after reset.

    Task instruction and available actions are now in the system prompt,
    so the initial observation only contains the image.
    """
    return f"""[Current Observation]:
{img_str}

Decide your next action."""


def action_template(last_action, env_feedback, img_str, task_instruction=None):
    """Template for step observation with feedback.

    Encourages structured reasoning: analyze the feedback,
    reflect on why the last action succeeded or failed,
    then plan the next logical step.
    """
    task_line = f"\n[Task]: {task_instruction}\n" if task_instruction else ""
    return f"""[Last Action]: {last_action}
[Feedback]: {env_feedback}
{task_line}
[Current Observation]:
{img_str}

You MUST first analyze the feedback above. If the action succeeded, plan the next logical step to complete the task. If it failed, explain why and try a different approach. Do NOT repeat the same failed action."""


def format_prompt(max_actions_per_step, action_sep, add_example=True, prompt_format="free_think"):
    """Generate format prompt based on the specified format."""
    if prompt_format == "free_think":
        return free_think_format_prompt(max_actions_per_step, action_sep, add_example)
    elif prompt_format == "wm":
        return wm_format_prompt(max_actions_per_step, action_sep, add_example)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")


def free_think_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Generate format prompt for free_think format."""
    if max_actions_per_step == 1:
        base = """You should output 1 action at a time.
Output the action as [action_id, action_name] using the ID from the available actions list.
Your response should be in the format of:
<think>...</think><answer>[N, action name]</answer>"""
    else:
        base = f"""You should output a plan of up to {max_actions_per_step} actions at a time, separated by "{action_sep}".
Output each action as [action_id, action_name] using the ID from the available actions list.
Your response should be in the format of:
<think>...</think><answer>[N1, action1]{action_sep} [N2, action2]{action_sep} ...</answer>"""

    if add_example:
        if max_actions_per_step == 1:
            examples = """
Example 1:
<think>I need to find a mug first. Let me navigate to where mugs might be.</think>
<answer>[5, find a Mug]</answer>

Example 2:
<think>The mug is nearby and I'm not holding anything. I should pick it up.</think>
<answer>[12, pick up the Mug]</answer>

Example 3:
<think>I'm holding the mug and I'm near the table. Let me put it down.</think>
<answer>[38, put down the object in hand]</answer>"""
        else:
            examples = f"""
Example 1 (multi-step plan):
<think>I need to find the alarm clock, pick it up, then find the desk lamp and turn it on.</think>
<answer>[3, find a AlarmClock]{action_sep} [15, pick up the AlarmClock]{action_sep} [7, find a DeskLamp]{action_sep} [42, turn on the DeskLamp]</answer>

Example 2 (single action when unsure):
<think>I am not sure where the mug is. Let me find it first.</think>
<answer>[5, find a Mug]</answer>

Example 3 (replanning after failure):
<think>The last action failed because the cabinet was closed. I need to open it first, then pick up the object.</think>
<answer>[20, open the Cabinet]{action_sep} [12, pick up the Mug]</answer>"""
        return base + "\n" + examples

    return base


def wm_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Generate format prompt for wm format with observation and prediction tags."""
    base = f"""You should output {max_actions_per_step} action(s) at a time.
Output the action as [action_id, action_name] using the ID from the available actions list.
Your response must be in the format of:
<observation>...</observation><think>...</think><answer>[N, action name]</answer><prediction>...</prediction>.

Rules for <observation>:
- Describe the current scene: what objects you see, your position, what you are holding, and relevant receptacle states.

Rules for <prediction>:
- Predict what will change after your action: where you will be, what you will see, and the expected result.

Rules for <answer>:
- Output exactly 1 action as [action_id, action_name]."""

    if add_example:
        examples = """
Example 1:
<observation>I see a kitchen with a counter, a microwave, and a mug on the counter. I am not holding anything.</observation>
<think>I need to pick up the mug. First, I should find it to get close to it.</think>
<answer>[5, find a Mug]</answer>
<prediction>I will navigate to the mug and see it up close on the counter.</prediction>

Example 2:
<observation>I am close to a Mug on the counter. I am not holding anything. The mug is within reach.</observation>
<think>The mug is nearby and I'm not holding anything. I should pick it up.</think>
<answer>[12, pick up the Mug]</answer>
<prediction>I will be holding the mug. The counter will no longer have the mug on it.</prediction>

Example 3:
<observation>I am holding a Mug. I see a table nearby with an empty spot.</observation>
<think>I'm holding the mug and I'm near the table. Let me put it down.</think>
<answer>[38, put down the object in hand]</answer>
<prediction>The mug will be placed on the table. I will no longer be holding anything.</prediction>"""
        return base + "\n" + examples

    return base
