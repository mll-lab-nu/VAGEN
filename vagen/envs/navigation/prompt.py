"""
Prompt templates for the AI2-THOR navigation environment.

Ported from vagen-legacy/vagen/env/navigation/prompt.py.
"""

FORMAT_CONFIGS = {
    "free_think": {
        "description": "You should first give your thought process, and then your answer.",
        "format": "<think>...</think><answer>...</answer>",
        "example": (
            "<think>I can see from the sight the target object is right in the top left of me, "
            "I will move forward, then move left to access it.</think>"
            "<answer>moveahead{action_sep}moveahead{action_sep}moveahead{action_sep}moveleft{action_sep}moveleft</answer>"
        ),
    },
    "no_think": {
        "description": "You should provide only your answer.",
        "format": "<answer>...</answer>",
        "example": "<answer>moveahead{action_sep}moveahead{action_sep}moveahead{action_sep}moveleft{action_sep}moveleft</answer>",
    },
    "grounding": {
        "description": (
            "You should first give your thought process with your observation and reasoning, and finally your answer.\n"
            "The observation should be described in detail about what you see in the environment."
        ),
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "example": (
            "<think><observation>I am in a living room. There is a couch to my left, a TV in front of me, "
            "and a doorway to the kitchen on my right. The target object, a vase, appears to be on a shelf near "
            "the kitchen doorway.</observation><reasoning>I need to move toward the kitchen doorway to reach the vase. "
            "I'll move forward to get closer to the center of the room, then turn right and move toward the kitchen."
            "</reasoning></think>"
            "<answer>moveahead{action_sep}moveahead{action_sep}rotateright{action_sep}moveahead{action_sep}moveahead</answer>"
        ),
    },
    "worldmodeling": {
        "description": (
            "You should first give your thought process with reasoning and prediction of next state, then your answer.\n"
            "The prediction should describe what you expect to see after your actions are executed."
        ),
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": (
            "<think><reasoning>I can see the kitchen doorway to my right, and I need to go there to find the "
            "refrigerator. I'll turn right and move forward.</reasoning><prediction>I am now in the kitchen doorway. "
            "In front of me is the kitchen counter with a sink. To the left I can see a refrigerator against the wall. "
            "There's a kitchen island in the center of the room.</prediction></think>"
            "<answer>rotateright{action_sep}moveahead{action_sep}moveahead</answer>"
        ),
    },
    "grounding_worldmodeling": {
        "description": (
            "You should first give your thought process with the your observation, reasoning, and prediction "
            "of next state, then your answer.\n"
            "Both the observation and prediction should describe what you see or expect to see in the environment."
        ),
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": (
            "<think><observation>I am at the entrance of a bedroom. There is a bed to the left, a desk with a lamp "
            "on the right, and a closet straight ahead. The target object, a book, appears to be on the desk."
            "</observation><reasoning>I need to move toward the desk to reach the book. I'll turn right and move "
            "forward.</reasoning><prediction>I am now standing in front of the desk. The desk has a lamp, a computer, "
            "and several books on it. The target book is within reach on the right side of the desk.</prediction></think>"
            "<answer>rotateright{action_sep}moveahead{action_sep}moveahead</answer>"
        ),
    },
}


def system_prompt(**kwargs) -> str:
    """Return the system prompt including format-specific multi-round example."""
    selected_format = kwargs.get("format", "free_think")

    # Multi-round examples keyed by format
    if selected_format in ("free_think", "default"):
        example = (
            "Example:\n"
            "Round 1:\nimage_1\n"
            "<think>I can see the garbage can in the upper left corner of the image, next to the kitchen sink. "
            "To move there, we can go forward-left, but since there's a kitchen counter directly ahead, "
            "we should go left first. Following the strategy, I can go by first moving leftward.</think>\n"
            "<answer>moveleft, moveleft</answer>\n"
            "Round 2:\nEnv_feedback: Last action is executed successfully.\nimage_2\n"
            "<think>From the secene, I see that by moving leftward, we are getting closer to the garbage can. "
            "Now, the garbage can is in front of me, slightly to the left. And there's a large area ahead of us. "
            "Following the strategy, I can go by first moving forward then moving leftward.</think>\n"
            "<answer>moveahead, moveahead,moveahead,moveleft</answer>\n"
            "Round 3:\nEnv_feedback: Last action is executed successfully.\nimage_3\n"
            "<think>From the image we can see the garbage can is very close to us, still to our front-left. "
            "Moving leftward might be blocked but i can see that there is still space in front of me to get closer "
            "to the garbage can. Following the strategy, we can take about two steps forward then one step left "
            "to reach the garbage can.</think>\n"
            "<answer>moveahead, moveahead,moveleft</answer>\n"
            "Round 4:\nEnv_feedback: Success"
        )
    elif selected_format == "grounding":
        example = (
            "Example:\n"
            "Round 1:\nimage_1\n"
            "<think><observation>There is a garbage can in the upper left corner of the image, next to the kitchen sink. "
            "To move there, we can go forward-left, but since there's a kitchen counter directly ahead, we should go "
            "left first.</observation><reasoning>Following the strategy, I can go by first moving leftward.</reasoning></think>\n"
            "<answer>moveleft, moveleft</answer>\n"
            "Round 2:\nEnv_feedback: Last action is executed successfully.\nimage_2\n"
            "<think><observation>From the secene, I see that by moving leftward, we are getting closer to the garbage can. "
            "Now, the garbage can is in front of me, slightly to the left. And there's a large area ahead of us."
            "</observation><reasoning>Following the strategy, I can go by first moving forward then moving leftward."
            "</reasoning></think>\n"
            "<answer>moveahead, moveahead,moveahead,moveleft</answer>\n"
            "Round 3:\nEnv_feedback: Last action is executed successfully.\nimage_3\n"
            "<think><observation>From the image we can see the garbage can is very close to us, still to our front-left. "
            "Moving leftward might be blocked but i can see that there is still space in front of me to get closer to "
            "the garbage can.</observation><reasoning>Following the strategy, we can take about two steps forward then "
            "one step left to reach the garbage can.</reasoning></think>\n"
            "<answer>moveahead, moveahead,moveleft</answer>\n"
            "Round 4:\nEnv_feedback: Success"
        )
    elif selected_format == "worldmodeling":
        example = (
            "Example:\n"
            "Round 1:\nimage_1\n"
            "<think><reasoning>I can see the garbage can in the upper left corner of the image, next to the kitchen sink. "
            "To move there, we can go forward-left, but since there's a kitchen counter directly ahead, we should go "
            "left first.</reasoning><prediction>I will be infront of the garbage</prediction></think>\n"
            "<answer>moveleft, moveleft</answer>\n"
            "Round 2:\nEnv_feedback: Last action is executed successfully.\nimage_2\n"
            "<think><reasoning>From the secene, I see that by moving leftward, we are getting closer to the garbage can. "
            "Now, the garbage can is in front of me, slightly to the left. And there's a large area ahead of us."
            "</reasoning><prediction>I will be closer to the garbage</prediction></think>\n"
            "<answer>moveahead, moveahead,moveahead,moveleft</answer>\n"
            "Round 3:\nEnv_feedback: Last action is executed successfully.\nimage_3\n"
            "<think><reasoning>From the image we can see the garbage can is very close to us, still to our front-left. "
            "Moving leftward might be blocked but i can see that there is still space in front of me to get closer to "
            "the garbage can.</reasoning><prediction>I will reach the garbage</prediction></think>\n"
            "<answer>moveahead, moveahead,moveleft</answer>\n"
            "Round 4:\nEnv_feedback: Success"
        )
    elif selected_format == "grounding_worldmodeling":
        example = (
            "Example:\n"
            "Round 1:\nimage_1\n"
            "<think><observation>There is a garbage can in the upper left corner of the image, next to the kitchen sink. "
            "To move there, we can go forward-left, but since there's a kitchen counter directly ahead, we should go "
            "left first.</observation><reasoning>Following the strategy, I can go by first moving leftward.</reasoning>"
            "<prediction>I will be infront of the garbage</prediction></think>\n"
            "<answer>moveleft, moveleft</answer>\n"
            "Round 2:\nEnv_feedback: Last action is executed successfully.\nimage_2\n"
            "<think><observation>From the secene, I see that by moving leftward, we are getting closer to the garbage can. "
            "Now, the garbage can is in front of me, slightly to the left. And there's a large area ahead of us."
            "</observation><reasoning>Following the strategy, I can go by first moving forward then moving leftward."
            "</reasoning><prediction>I will be closer to the garbage</prediction></think>\n"
            "<answer>moveahead, moveahead,moveahead,moveleft</answer>\n"
            "Round 3:\nEnv_feedback: Last action is executed successfully.\nimage_3\n"
            "<think><observation>From the image we can see the garbage can is very close to us, still to our front-left. "
            "Moving leftward might be blocked but i can see that there is still space in front of me to get closer to "
            "the garbage can.</observation><reasoning>Following the strategy, we can take about two steps forward then "
            "one step left to reach the garbage can.</reasoning><prediction>I will reach the garbage</prediction></think>\n"
            "<answer>moveahead, moveahead,moveleft</answer>\n"
            "Round 4:\nEnv_feedback: Success"
        )
    elif selected_format == "no_think":
        example = (
            "Example:\n"
            "Round 1:\nimage_1\n"
            "<answer>moveleft, moveleft</answer>\n"
            "Round 2:\nEnv_feedback: Last action is executed successfully.\nimage_2\n"
            "<answer>moveahead, moveahead,moveahead,moveleft</answer>\n"
            "Round 3:\nEnv_feedback: Last action is executed successfully.\nimage_3\n"
            "<answer>moveahead, moveahead,moveleft</answer>\n"
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
        f"After your answer, the extracted valid action is {valid_action}.\n"
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
