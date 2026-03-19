import re
from typing import Dict, List, Optional
from PIL import Image
import numpy as np


def parse_free_think(response: str, action_sep: str = ",", max_actions: int = 1) -> Dict:
    """
    Parse free_think format response: <think>...</think><answer>...</answer>

    For EB-ALFRED, the <answer> tag typically contains a single action name
    (e.g., "find a Cabinet") or an action ID (e.g., "42").
    """
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

        if max_actions == 1:
            actions = [action_content.strip()] if action_content.strip() else []
        else:
            actions = [a.strip() for a in action_content.split(action_sep) if a.strip()]
            if len(actions) > max_actions:
                actions = actions[:max_actions]
                action_content = action_sep.join(actions)

    llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>"

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }


def parse_wm(response: str, action_sep: str = ",", max_actions: int = 1) -> Dict:
    """
    Parse wm format response:
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

        if max_actions == 1:
            actions = [action_content.strip()] if action_content.strip() else []
        else:
            actions = [a.strip() for a in action_content.split(action_sep) if a.strip()]
            if len(actions) > max_actions:
                actions = actions[:max_actions]
                action_content = action_sep.join(actions)

    llm_response = (
        f"<observation>{observation_content}</observation>"
        f"<think>{think_content}</think>"
        f"<answer>{action_content}</answer>"
        f"<prediction>{prediction_content}</prediction>"
    )

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


def normalize_era_tokens(response: str) -> str:
    """
    Convert ERA special tokens to VAGEN plain tags so the parser can handle
    models trained with ERA's SFT format.

    ERA format:   <|think_start|>...<|think_end|><|action_start|>...<|action_end|>
    VAGEN format: <think>...</think><answer>...</answer>
    """
    response = response.replace("<|think_start|>", "<think>")
    response = response.replace("<|think_end|>", "</think>")
    response = response.replace("<|action_start|>", "<answer>")
    response = response.replace("<|action_end|>", "</answer>")
    return response


def parse_response(
    response: str,
    prompt_format: str = "free_think",
    action_sep: str = ",",
    max_actions: int = 1,
) -> Dict:
    """Parse LLM response based on the specified prompt format."""
    response = normalize_era_tokens(response)
    if prompt_format == "free_think":
        return parse_free_think(response, action_sep, max_actions)
    elif prompt_format == "wm":
        return parse_wm(response, action_sep, max_actions)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")


def match_action(
    action_name: str,
    action_list: List[str],
    action_map: Dict[str, str],
) -> Optional[str]:
    """
    Match a parsed action against the valid action set.

    Supports multiple formats (in priority order):
      - ERA-style [id, action_name]: "[42, find a Cabinet]"
      - Legacy (id: N) suffix: "find a Cabinet (id: 42)"
      - Plain action ID: "42"
      - Action name (case-insensitive): "find a Cabinet"

    Returns the original action string if matched, None otherwise.
    """
    name = action_name.strip()

    # Try ERA-style [id, action_name] format
    bracket_match = re.match(r'^\[(\d+),\s*(.+?)\]$', name)
    if bracket_match:
        idx = int(bracket_match.group(1))
        if 0 <= idx < len(action_list):
            return action_list[idx]
        # ID out of range; try name part
        fallback_name = bracket_match.group(2).strip()
        return action_map.get(fallback_name.lower())

    # Try legacy "(id: N)" suffix
    id_match = re.search(r'\(id:\s*(\d+)\)\s*$', name)
    if id_match:
        idx = int(id_match.group(1))
        if 0 <= idx < len(action_list):
            return action_list[idx]
        name = name[:id_match.start()].strip()

    # Try as integer action ID
    try:
        idx = int(name)
        if 0 <= idx < len(action_list):
            return action_list[idx]
    except ValueError:
        pass

    # Try exact match by name (case-insensitive)
    return action_map.get(name.lower())


def numpy_to_pil(numpy_array: np.ndarray) -> Image.Image:
    """Convert numpy (H, W, 3) to PIL.Image in RGB."""
    if numpy_array.shape[-1] == 3:
        return Image.fromarray(numpy_array.astype(np.uint8), mode="RGB")
    raise ValueError(f"Unsupported channels: {numpy_array.shape[-1]}. Expected 3 (RGB).")
