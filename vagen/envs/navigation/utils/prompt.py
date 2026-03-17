"""
Prompt templates for the AI2-THOR navigation environment.

Formats:
  - free_think: <think>...</think><action>...</action>
  - wm: <observation>...</observation><think>...</think><action>...</action><prediction>...</prediction>
  - no_think: <action>...</action>
"""

FORMAT_CONFIGS = {
    "free_think": {
        "description": "You should first give your thought process, and then your action.",
        "format": "<think>...</think><action>...</action>",
        "example": (
            "<think>I can see from the sight the target object is right in the top left of me, "
            "I will move forward, then move left to access it.</think>"
            "<action>moveahead{action_sep}moveahead{action_sep}moveahead{action_sep}moveleft{action_sep}moveleft</action>"
        ),
    },
    "wm": {
        "description": (
            "You should first describe your observation, then give your reasoning, "
            "then your action, and finally predict what you expect to see after the action."
        ),
        "format": "<observation>...</observation><think>...</think><action>...</action><prediction>...</prediction>",
        "example": (
            "<observation>I am at the entrance of a bedroom. There is a bed to the left, a desk with a lamp "
            "on the right, and a closet straight ahead. The target object, a book, appears to be on the desk."
            "</observation>"
            "<think>I need to move toward the desk to reach the book. I'll turn right and move forward.</think>"
            "<action>rotateright{action_sep}moveahead{action_sep}moveahead</action>"
            "<prediction>I am now standing in front of the desk. The desk has a lamp, a computer, "
            "and several books on it. The target book is within reach on the right side of the desk.</prediction>"
        ),
    },
    "no_think": {
        "description": "You should provide only your action.",
        "format": "<action>...</action>",
        "example": "<action>moveahead{action_sep}moveahead{action_sep}moveahead{action_sep}moveleft{action_sep}moveleft</action>",
    },
}


def system_prompt(**kwargs) -> str:
    """Return the system prompt including format-specific multi-round example."""
    selected_format = kwargs.get("format", "free_think")

    if selected_format in ("free_think", "default"):
        example = (
            "Example:\n"
            "Round 1:\nimage_1\n"
            "<think>I can see the garbage can in the upper left corner of the image, next to the kitchen sink. "
            "To move there, we can go forward-left, but since there's a kitchen counter directly ahead, "
            "we should go left first. Following the strategy, I can go by first moving leftward.</think>\n"
            "<action>moveleft, moveleft</action>\n"
            "Round 2:\nEnv_feedback: Last action is executed successfully.\nimage_2\n"
            "<think>From the scene, I see that by moving leftward, we are getting closer to the garbage can. "
            "Now, the garbage can is in front of me, slightly to the left. And there's a large area ahead of us. "
            "Following the strategy, I can go by first moving forward then moving leftward.</think>\n"
            "<action>moveahead, moveahead,moveahead,moveleft</action>\n"
            "Round 3:\nEnv_feedback: Last action is executed successfully.\nimage_3\n"
            "<think>From the image we can see the garbage can is very close to us, still to our front-left. "
            "Moving leftward might be blocked but i can see that there is still space in front of me to get closer "
            "to the garbage can. Following the strategy, we can take about two steps forward then one step left "
            "to reach the garbage can.</think>\n"
            "<action>moveahead, moveahead,moveleft</action>\n"
            "Round 4:\nEnv_feedback: Success"
        )
    elif selected_format == "wm":
        example = (
            "Example:\n"
            "Round 1:\nimage_1\n"
            "<observation>There is a garbage can in the upper left corner of the image, next to the kitchen sink. "
            "A kitchen counter is directly ahead blocking the path forward.</observation>\n"
            "<think>Following the strategy, I should go left first to avoid the counter.</think>\n"
            "<action>moveleft, moveleft</action>\n"
            "<prediction>I will be closer to the garbage can, with the kitchen counter now to my right.</prediction>\n"
            "Round 2:\nEnv_feedback: Last action is executed successfully.\nimage_2\n"
            "<observation>By moving leftward, we are getting closer to the garbage can. "
            "The garbage can is in front of me, slightly to the left. There's a large area ahead.</observation>\n"
            "<think>I can go by first moving forward then moving leftward.</think>\n"
            "<action>moveahead, moveahead,moveahead,moveleft</action>\n"
            "<prediction>I will be very close to the garbage can, it should be right in front of me.</prediction>\n"
            "Round 3:\nEnv_feedback: Last action is executed successfully.\nimage_3\n"
            "<observation>The garbage can is very close to us, still to our front-left. "
            "There is still space in front of me.</observation>\n"
            "<think>We can take about two steps forward then one step left to reach the garbage can.</think>\n"
            "<action>moveahead, moveahead,moveleft</action>\n"
            "<prediction>I will reach the garbage can.</prediction>\n"
            "Round 4:\nEnv_feedback: Success"
        )
    elif selected_format == "no_think":
        example = (
            "Example:\n"
            "Round 1:\nimage_1\n"
            "<action>moveleft, moveleft</action>\n"
            "Round 2:\nEnv_feedback: Last action is executed successfully.\nimage_2\n"
            "<action>moveahead, moveahead,moveahead,moveleft</action>\n"
            "Round 3:\nEnv_feedback: Last action is executed successfully.\nimage_3\n"
            "<action>moveahead, moveahead,moveleft</action>\n"
            "Round 4:\nEnv_feedback: Success"
        )
    else:
        example = ""

    base_prompt = (
        "You are a home robot and perform navigation tasks according to instructions.\n"
        "Actions you can take: moveahead, moveback, moveright, moveleft, rotateright, rotateleft, lookup, lookdown. \n"
        "moveahead: Move forward by some distance\n"
        "moveback: Move backward by some distance\n"
        "moveright: Move rightward by some distance\n"
        "moveleft: Move leftward by some distance\n"
        "rotateright: Rotate to the right by 90 degrees\n"
        "rotateleft: Rotate to the left by 90 degrees\n"
        "lookup: Tilt the camera upward by 30 degrees\n"
        "lookdown: Tilt the camera downward by 30 degrees\n"
        "Rewards:\n"
        "Format correct: +0.5\n"
        "Achieve the human instruction: +10.0\n"
        "The instruction will be provided with each observation. Look at the image carefully and navigate to complete the instruction.\n"
        "Hints:\n"
        "1. You can take multiple actions at a time, in most cases, if you find the target object is far away from you, "
        "you can call moveahead, moveleft and move right multiple times.\n"
        "2. If you find yourself seems to be stuck, you can lookdown to see if there's any object above or below you, "
        "you can also rotate to see if there's any object behind you."
    )
    return base_prompt + "\n" + example


def init_observation_template(**kwargs) -> str:
    observation = kwargs.get("observation", "No observation provided.")
    instruction = kwargs.get("instruction", "No instruction provided.")
    return (
        f"[Initial Observation]:\n"
        f"{observation}\n"
        f"Human Instruction: {instruction}\n"
        f"Decide your next action(s)."
    )


def action_template(**kwargs) -> str:
    observation = kwargs.get("observation", "No observation provided.")
    instruction = kwargs.get("instruction", "No instruction provided.")
    valid_action = kwargs.get("valid_action", "No valid action provided.")
    env_feedback = kwargs.get("env_feedback", "No environment feedback provided.")
    reward = kwargs.get("reward", "No reward provided.")
    done = kwargs.get("done", "No done status provided.")
    return (
        f"After your action, the extracted valid action is {valid_action}.\n"
        f"The environment feedback is: {env_feedback}\n"
        f"reward: {reward}\n"
        f"done: {done}\n"
        f"After that, the observation is:\n"
        f"{observation}\n"
        f"Human Instruction: {instruction}\n"
        f"Decide your next action(s)."
    )


def format_prompt_generator(format_type: str):
    """Factory that returns a per-format prompt function."""

    def prompt_function(**kwargs) -> str:
        max_actions_per_step = kwargs.get("max_actions_per_step", 5)
        action_sep = kwargs.get("action_sep", ",")
        add_example = kwargs.get("add_example", True)

        config = FORMAT_CONFIGS[format_type]

        base = (
            f"You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'.\n"
            f"{config['description']}\n"
            f"Your response should be in the format of:\n"
            f"{config['format']}"
        )

        if add_example:
            example_text = config["example"].format(action_sep=action_sep)
            return base + "\n" + f"e.g. {example_text}"

        return base

    return prompt_function


format_prompt = {ft: format_prompt_generator(ft) for ft in FORMAT_CONFIGS}
