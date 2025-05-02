import re
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Union



def parse_llm_raw_response(response: str,special_token_list=None,action_sep=',',max_actions=3) -> Dict:
    """
    assume a good format is <think>...</think><answer>...</answer>
    returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with <think> and <answer> tags
    - think_content: the content inside <think> tag
    - action_content: the content inside <answer> tag
    - actions: a list of actions extracted from action_content
    """

    pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
    match = re.search(pattern, response, re.DOTALL)
    format_correct = match is not None
    
    if not match:
        think_content, action_content, actions = "", "", []
    else:
        think_content, action_content = match.group(1), match.group(2)
        if special_token_list is not None:
            for special_token in special_token_list: # remove all special tokens in responses to forbid confusion in training
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
        actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
            action_content = (" " + action_sep + " ").join(actions)

    llm_response = "<think>" + think_content.strip() + "</think>" + "<answer>" + action_content.strip() + "</answer>"
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }



def convert_numpy_to_PIL(numpy_array: np.ndarray) -> Image.Image:
    """Convert a numpy array to a PIL RGB image."""
    if numpy_array.shape[-1] == 3:
        # Convert numpy array to RGB PIL Image
        return Image.fromarray(numpy_array, mode='RGB')
    else:
        raise ValueError(f"Unsupported number of channels: {numpy_array.shape[-1]}. Expected 3 (RGB).")









if __name__ == "__main__":
    text = """
    <think>
    I am thinking about the problem.
    </think>
    <answer>
    answer1, answer2, answer3
    </answer>
    """
    print(parse_llm_raw_response(text))