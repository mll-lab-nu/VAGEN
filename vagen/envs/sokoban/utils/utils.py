import re
from typing import Dict, List
from PIL import Image
import numpy as np

def parse_free_think(response: str, action_sep: str = ",", max_actions: int = 3) -> Dict:
    """
    Parse free_think format response: <think>...</think><answer>...</answer>
    
    Args:
        response: Raw LLM response string
        action_sep: Separator between actions
        max_actions: Maximum number of actions to extract
    
    Returns:
        Dict containing parsed components and validation info
    """
    # Pattern to match <think>...</think><answer>...</answer>
    pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
    match = re.search(pattern, response, re.DOTALL)
    
    format_correct = match is not None
    
    if not match:
        think_content = ""
        action_content = ""
        actions = []
    else:
        think_content = match.group(1).strip()
        action_content = match.group(2).strip()
        
        # Split actions by separator and clean them
        actions = [action.strip().lower() for action in action_content.split(action_sep) if action.strip()]
        
        # Limit to max_actions
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = action_sep.join(actions)
    
    # Reconstruct formatted response
    llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>"
    
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }

def parse_wm(response: str, action_sep: str = ",", max_actions: int = 3) -> Dict:
    """
    Parse wm_new format response:
    <observation>...</observation>
    <think>...</think>
    <answer>...</answer>
    <prediction>...</prediction>
    """
    pattern = (
        r'<observation>(.*?)</observation>\s*'
        r'<think>(.*?)</think>\s*'
        r'<answer>(.*?)</answer>\s*'
        r'<prediction>(.*?)</prediction>'
    )

    match = re.search(pattern, response, re.DOTALL)
    format_correct = match is not None

    if not match:
        observation_content = ""
        think_content = ""
        prediction_content = ""
        action_content = ""
        actions: List[str] = []
    else:
        observation_content = match.group(1).strip()
        think_content = match.group(2).strip()
        action_content = match.group(3).strip()
        prediction_content = match.group(4).strip()

        # Parse actions
        actions = [
            action.strip().lower()
            for action in action_content.split(action_sep)
            if action.strip()
        ]

        # Limit number of actions
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = action_sep.join(actions)

    # Reconstruct formatted response (canonical)
    llm_response = (
        f"<observation>{observation_content}</observation>"
        f"<think>{think_content}</think>"
        f"<answer>{action_content}</answer>"
        f"<prediction>{prediction_content}</prediction>"
    )

    # For backward-compat with old keys: treat think as reasoning
    reasoning_content = think_content

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "observation_content": observation_content,
        "think_content": think_content,
        "reasoning_content": reasoning_content,
        "prediction_content": prediction_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }

def parse_response(response: str, prompt_format: str = "free_think", action_sep: str = ",", max_actions: int = 3) -> Dict:
    """Parse LLM response based on the specified prompt format"""
    if prompt_format == "free_think":
        return parse_free_think(response, action_sep, max_actions)
    elif prompt_format == "wm":
        return parse_wm(response, action_sep, max_actions)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")
    
def numpy_to_pil(numpy_array: np.ndarray) -> Image.Image:
    """Convert numpy (H, W, 3) to PIL.Image in RGB."""
    if numpy_array.shape[-1] == 3:
        return Image.fromarray(numpy_array.astype(np.uint8), mode="RGB")
    raise ValueError(f"Unsupported channels: {numpy_array.shape[-1]}. Expected 3 (RGB).")