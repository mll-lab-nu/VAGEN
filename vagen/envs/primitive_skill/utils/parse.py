"""
Response parsing and reward utilities for the primitive_skill environment.

Formats:
  - free_think: <think>...</think><answer>...</answer>
  - wm: <observation>...</observation><think>...</think><answer>...</answer><prediction>...</prediction>
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Parse patterns (same structure as navigation)
# ---------------------------------------------------------------------------

_PARSE_PATTERNS = {
    "free_think": r"<think>(.*?)</think>\s*<answer>(.*?)</answer>",
    "wm": (
        r"<observation>(.*?)</observation>\s*"
        r"<think>(.*?)</think>\s*"
        r"<answer>(.*?)</answer>\s*"
        r"<prediction>(.*?)</prediction>"
    ),
}


def parse_response(
    response: str,
    prompt_format: str = "free_think",
    action_sep: str = "|",
    max_actions: int = 2,
) -> Dict[str, Any]:
    """Parse an LLM response and extract actions.

    Returns dict with keys:
        llm_raw_response, actions, format_correct,
        and optional think/observation/prediction text.
    """
    result: Dict[str, Any] = {
        "llm_raw_response": response,
        "actions": [],
        "format_correct": False,
    }

    pattern = _PARSE_PATTERNS.get(prompt_format)
    if pattern is None:
        raise ValueError(f"Unknown prompt_format: {prompt_format}")

    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return result

    if prompt_format == "free_think":
        result["think"] = match.group(1).strip()
        action_text = match.group(2).strip()
    elif prompt_format == "wm":
        result["observation"] = match.group(1).strip()
        result["think"] = match.group(2).strip()
        action_text = match.group(3).strip()
        result["prediction"] = match.group(4).strip()

    result["format_correct"] = True
    actions = [a.strip() for a in action_text.split(action_sep) if a.strip()][:max_actions]
    result["actions"] = actions
    return result


def compute_reward(
    parsed: Dict[str, Any],
    valid_actions: List[str],
    success: bool,
    stage_reward: float,
    format_reward: float = 0.1,
    success_reward: float = 10.0,
) -> float:
    """Compute step reward.

    - format_reward: given each step if format is correct and actions are valid
    - stage_reward: incremental reward from completing task stages
    - success_reward: given when full task is completed
    """
    reward = 0.0

    # Format bonus (each step if format correct and actions valid)
    if parsed["format_correct"] and len(valid_actions) > 0 and len(valid_actions) == len(parsed["actions"]):
        reward += format_reward

    # Stage progression reward
    reward += stage_reward

    # Success bonus
    if success:
        reward += success_reward

    return reward
