"""
Response parsing and reward utilities for the primitive_skill environment.

Formats:
  - free_think: <think>...</think><answer>...</answer>
  - wm: <perception>...</perception><reasoning>...</reasoning><prediction>...</prediction><answer>...</answer>
"""

from __future__ import annotations

from typing import Any, Dict, List

from vagen.envs._common.response_format import (
    parse_free_think_sections,
    parse_wm_sections,
    split_actions,
)

PROMPT_FORMATS = frozenset({"wm", "free_think"})

# ---------------------------------------------------------------------------
# Parse patterns (same structure as navigation)
# ---------------------------------------------------------------------------

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

    if prompt_format == "free_think":
        sections = parse_free_think_sections(response)
    elif prompt_format == "wm":
        sections = parse_wm_sections(response)
    else:
        raise ValueError(f"Unknown prompt_format: {prompt_format}")

    result.update(
        perception_content=sections.perception,
        reasoning_content=sections.reasoning,
        prediction_content=sections.prediction,
        action_content=sections.answer,
        format_correct=sections.format_correct,
        actions=split_actions(sections.answer, action_sep, max_actions, lower=False),
    )
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
