from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image

from vagen.envs._common.response_format import (
    canonical_free_think,
    canonical_wm,
    parse_answer_sections,
    parse_free_think_sections,
    parse_wm_sections,
    split_actions,
)

PROMPT_FORMATS = frozenset({"wm", "wm_think", "free_think", "answer"})


def _result(response: str, sections, action_sep: str, max_actions: int, *, kind: str) -> Dict:
    actions = split_actions(sections.answer, action_sep, max_actions)
    action_content = action_sep.join(actions)
    canonical = (
        canonical_wm(sections)
        if kind == "wm"
        else canonical_free_think(sections)
        if kind == "free_think"
        else f"<answer>{action_content}</answer>"
    )
    return {
        "llm_raw_response": response,
        "llm_response": canonical,
        "perception_content": sections.perception,
        "reasoning_content": sections.reasoning,
        "prediction_content": sections.prediction,
        "action_content": action_content,
        "actions": actions,
        "format_correct": sections.format_correct,
    }


def parse_free_think(response: str, action_sep: str = ",", max_actions: int = 3) -> Dict:
    """Parse the shared ``<think>...</think><answer>...</answer>`` protocol."""
    sections = parse_free_think_sections(response, allow_native_answer=True)
    return _result(response, sections, action_sep, max_actions, kind="free_think")


def parse_wm(response: str, action_sep: str = ",", max_actions: int = 3) -> Dict:
    """Parse perception -> reasoning -> prediction -> answer."""
    sections = parse_wm_sections(response)
    return _result(response, sections, action_sep, max_actions, kind="wm")


def parse_native_wm(response: str, action_sep: str = ",", max_actions: int = 3) -> Dict:
    """Parse canonical WM after an optional native ``<think>`` block."""
    sections = parse_wm_sections(response, allow_native_thinking=True)
    return _result(response, sections, action_sep, max_actions, kind="wm")


def parse_answer_only(response: str, action_sep: str = ",", max_actions: int = 3) -> Dict:
    """Parse an answer after optional model-native visible reasoning."""
    sections = parse_answer_sections(response, lenient=True)
    return _result(response, sections, action_sep, max_actions, kind="answer")


def parse_free_wm(response: str, action_sep: str = ",", max_actions: int = 3) -> Dict:
    """Compatibility name for WM with an optional native-thinking prefix."""
    return parse_native_wm(response, action_sep, max_actions)


def parse_response(
    response: str,
    prompt_format: str = "free_think",
    action_sep: str = ",",
    max_actions: int = 3,
) -> Dict:
    """Parse an environment action response using the repository-wide protocol."""
    if prompt_format == "free_think":
        return parse_free_think(response, action_sep, max_actions)
    if prompt_format == "wm":
        return parse_wm(response, action_sep, max_actions)
    if prompt_format in {"free_wm", "wm_think"}:
        return parse_native_wm(response, action_sep, max_actions)
    if prompt_format == "answer":
        return parse_answer_only(response, action_sep, max_actions)
    raise ValueError(f"Unknown prompt format: {prompt_format}")


def numpy_to_pil(numpy_array: np.ndarray) -> Image.Image:
    """Convert numpy (H, W, 3) to PIL.Image in RGB."""
    if numpy_array.shape[-1] == 3:
        return Image.fromarray(numpy_array.astype(np.uint8), mode="RGB")
    raise ValueError(f"Unsupported channels: {numpy_array.shape[-1]}. Expected 3 (RGB).")
