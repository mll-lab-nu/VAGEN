from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────
# ERA-aligned system prompt
# ──────────────────────────────────────────────────────────────────────
# This matches the prompt the ERA EPL-Only model was SFT'd on, so the
# model stays in-distribution.  The output (ERA special tokens) is
# normalised back to VAGEN tags by normalize_era_tokens() in utils.py.
# ──────────────────────────────────────────────────────────────────────

ERA_SYSTEM_PROMPT_TEMPLATE = """\
## You are a robot operating in a home. Given a task, you must accomplish the task using a defined set of actions to achieve the desired outcome.

## Action Descriptions and Validity Rules
- Find: Parameterized by the name of the receptacle to navigate to. So long as the object is present in the scene, this skill is always valid
- Pick up: Parameterized by the name of the object to pick. Only valid if the robot is close to the object, not holding another object, and the object is not inside a closed receptacle.
- Put down: Parameterized by the name of the object to put down to a nearby receptacle. Only valid if the robot is holding an object.
- Drop: Parameterized by the name of the object to put down. It is different from Put down action, as this does not guarantee the held object will be put into a specified receptacle.
- Open: Parameterized by the name of the receptacle to open. Only valid if the receptacle is closed and the robot is close to the receptacle.
- Close: Parameterized by the name of the receptacle to close. Only valid if the receptacle is open and the robot is close to the receptacle.
- Turn on: Parameterized by the name of the object to turn on. Only valid if the object is turned off and the robot is close to the object.
- Turn off: Parameterized by the name of the object to turn off. Only valid if the object is turned on and the robot is close to the object.
- Slice: Parameterized by the name of the object to slice. Only valid if the object is sliceable and the robot is close to the object.

## The available action id (0 ~ {max_action_id}) and action names are: {available_actions}.

## Guidelines
1. **Output Plan**: Avoid generating empty plan. Each plan should include no more than 20 actions.
2. **Visibility**: Always locate a visible object by the 'find' action before interacting with it.
3. **Action Guidelines**: Make sure match the action name and its corresponding action id in the output. Avoid performing actions that do not meet the defined validity criteria. For instance, if you want to put object in a receptacle, use 'put down' rather than 'drop' actions.
4. **Prevent Repeating Action Sequences**: Do not repeatedly execute the same action or sequence of actions. Try to modify the action sequence because previous actions do not lead to success.
5. **Multiple Instances**: There may be multiple instances of the same object, distinguished by an index following their names, e.g., Cabinet_2, Cabinet_3. You can explore these instances if you do not find the desired object in the current receptacle.
6. **Reflection on History and Feedback**: Use interaction history and feedback from the environment to refine and improve your current plan. If the last action is invalid, reflect on the reason, such as not adhering to action rules or missing preliminary actions, and adjust your plan accordingly.

    ** Generation Guide **
    - Include the thinking process between <|think_start|> and <|think_end|>
    - Include only the target action in <|action_start|> and <|action_end|>, i.e. the content inside <|action_start|> and <|action_end|> should be nothing more than [action_id, 'action_name'], where the action id is an integer and the action name is the corresponding name. Do not include any other thing, such as '"'.
    """


def system_prompt(
    task_instruction: Optional[str] = None,
    action_list: Optional[List[str]] = None,
    add_task_examples: bool = True,
):
    """
    Build ERA-aligned system prompt for EB-ALFRED.

    The prompt exactly matches what the ERA EPL-Only model was trained on,
    so the model stays in-distribution.  Output normalisation (ERA tokens →
    VAGEN tags) happens downstream in normalize_era_tokens().
    """
    if action_list is not None:
        max_id = len(action_list) - 1
        available = ", ".join(
            f"[{i}, '{a}']" for i, a in enumerate(action_list)
        )
    else:
        max_id = "?"
        available = "(not yet available)"

    return ERA_SYSTEM_PROMPT_TEMPLATE.format(
        max_action_id=max_id,
        available_actions=available,
    )


def init_observation_template(img_str, task_instruction=None):
    """ERA-style initial user message.

    ERA format: <image>\\n instruction: {task} \\n interaction_history: [] ...
    """
    inst = task_instruction or ""
    return (
        f"{img_str}\n"
        f" instruction: {inst} \n"
        f" interaction_history: [] \n"
        "Based on the above information, please provide the action "
        "for the next step to complete the task. Think, then act."
    )


def action_template(last_action, env_feedback, img_str, task_instruction=None,
                    step_id=0, thinking="", action_id=None):
    """ERA-style step user message with structured interaction history.

    Matches ERA's exact format: interaction_history is a list of dicts with
    step_id, thinking, action [id, name], and env_feedback.
    """
    if action_id is not None:
        action_field = [action_id, last_action]
    else:
        action_field = last_action
    history = [{"step_id": step_id, "thinking": thinking,
                "action": action_field, "env_feedback": env_feedback}]
    inst = task_instruction or ""
    return (
        f"{img_str}\n"
        f" instruction: {inst} \n"
        f" interaction_history: {history} \n"
        "Based on the above information, please provide the action "
        "for the next step to complete the task. Think, then act."
    )


def format_prompt(max_actions_per_step, action_sep, add_example=True, prompt_format="free_think"):
    """Generate format prompt based on the specified format.

    For ERA-aligned mode, the generation guide is already in the system
    prompt, so we return a minimal reminder.
    """
    if prompt_format == "free_think":
        return free_think_format_prompt(max_actions_per_step, action_sep, add_example)
    elif prompt_format == "wm":
        return wm_format_prompt(max_actions_per_step, action_sep, add_example)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")


def free_think_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """Minimal format prompt — the ERA system prompt already has the Generation Guide."""
    # The ERA system prompt already instructs the model on output format.
    # Adding extra format instructions can confuse an SFT model, so keep it short.
    return ""


def wm_format_prompt(max_actions_per_step, action_sep, add_example=True):
    """World-model format prompt (not used by ERA, kept for compatibility)."""
    base = f"""You should output {max_actions_per_step} action(s) at a time.
Output the action as [action_id, 'action_name'] using the ID from the available actions list.
Your response must be in the format of:
<observation>...</observation><think>...</think><answer>[N, 'action name']</answer><prediction>...</prediction>.

Rules for <observation>:
- Describe the current scene: what objects you see, your position, what you are holding, and relevant receptacle states.

Rules for <prediction>:
- Predict what will change after your action: where you will be, what you will see, and the expected result.

Rules for <answer>:
- Output exactly 1 action as [action_id, 'action_name']."""

    if add_example:
        examples = """
Example 1:
<observation>I see a kitchen with a counter, a microwave, and a mug on the counter. I am not holding anything.</observation>
<think>I need to pick up the mug. First, I should find it to get close to it.</think>
<answer>[5, 'find a Mug']</answer>
<prediction>I will navigate to the mug and see it up close on the counter.</prediction>

Example 2:
<observation>I am close to a Mug on the counter. I am not holding anything. The mug is within reach.</observation>
<think>The mug is nearby and I'm not holding anything. I should pick it up.</think>
<answer>[12, 'pick up the Mug']</answer>
<prediction>I will be holding the mug. The counter will no longer have the mug on it.</prediction>

Example 3:
<observation>I am holding a Mug. I see a table nearby with an empty spot.</observation>
<think>I'm holding the mug and I'm near the table. Let me put it down.</think>
<answer>[38, 'put down the object in hand']</answer>
<prediction>The mug will be placed on the table. I will no longer be holding anything.</prediction>"""
        return base + "\n" + examples

    return base
