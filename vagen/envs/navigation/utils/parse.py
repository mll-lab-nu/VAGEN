"""
Response parsing and reward utilities for the navigation environment.

Formats:
  - free_think: <think>...</think><answer>...</answer>
  - wm: <perception>...</perception><reasoning>...</reasoning><prediction>...</prediction><answer>...</answer>
  - no_think: <answer>...</answer>
  - eval_mode: only requires <answer>...</answer> (everything else optional, lenient)
"""

from __future__ import annotations

from typing import Any, Dict, List

from vagen.envs._common.response_format import (
    parse_answer_sections,
    parse_free_think_sections,
    parse_wm_sections,
    split_actions,
)

PROMPT_FORMATS = frozenset({"wm", "free_think", "no_think", "eval_mode"})

# ---------------------------------------------------------------------------
# Parse patterns
# ---------------------------------------------------------------------------

def parse_response(
    response: str,
    prompt_format: str = "free_think",
    action_sep: str = "|",
    max_actions: int = 5,
) -> Dict[str, Any]:
    """Parse an LLM response and extract actions.

    Returns dict with keys:
        llm_raw_response, actions, format_correct,
        and optional think/observation/prediction text.
    """
    result: Dict[str, Any] = {"llm_raw_response": response, "actions": [], "format_correct": False}

    if prompt_format == "free_think":
        sections = parse_free_think_sections(response)
    elif prompt_format == "wm":
        sections = parse_wm_sections(response)
    elif prompt_format == "no_think":
        sections = parse_answer_sections(response)
    elif prompt_format == "eval_mode":
        sections = parse_answer_sections(response, lenient=True)
    else:
        raise ValueError(f"Unknown prompt_format: {prompt_format}")

    result.update(
        perception_content=sections.perception,
        reasoning_content=sections.reasoning,
        prediction_content=sections.prediction,
        action_content=sections.answer,
        format_correct=sections.format_correct,
        actions=split_actions(sections.answer, action_sep, max_actions),
    )
    return result


def compute_reward(
    parsed: Dict[str, Any],
    valid_actions: List[str],
    success: bool,
    format_reward: float = 0.5,
    per_turn_format_reward: float = 0.0,
    success_reward: float = 10.0,
    is_format_correct_so_far: bool = True,
) -> float:
    """Compute step reward.

    - per_turn_format_reward: given every step if format is correct this turn
    - format_reward: given at episode end only if ALL turns had correct format
    - success_reward: given when goal is reached
    """
    reward = 0.0

    # Per-turn format bonus (given each step if this turn's format is correct)
    if parsed["format_correct"] and valid_actions:
        reward += per_turn_format_reward

    # Success bonus
    if success:
        reward += success_reward
        # End-of-episode format bonus (only if all turns were correct)
        if is_format_correct_so_far and parsed["format_correct"]:
            reward += format_reward

    return reward
