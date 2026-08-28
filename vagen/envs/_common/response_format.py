"""Canonical response formats shared by every environment.

Environment implementations own their action vocabulary and reward semantics.  The
shape of a model response is repository-wide protocol, so its tags and ordering live
here rather than in five independent regular expressions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PERCEPTION = "perception"
REASONING = "reasoning"
PREDICTION = "prediction"
ANSWER = "answer"
THINK = "think"

WM_FORMAT = (
    "<perception>...</perception>"
    "<reasoning>...</reasoning>"
    "<prediction>...</prediction>"
    "<answer>...</answer>"
)
FREE_THINK_FORMAT = "<think>...</think><answer>...</answer>"
ANSWER_FORMAT = "<answer>...</answer>"

_TAG_ALIASES = {
    PERCEPTION: (PERCEPTION, "observation"),
    REASONING: (REASONING, "think", "thought"),
    PREDICTION: (PREDICTION,),
    ANSWER: (ANSWER, "action"),
}
_NATIVE_BOX = re.compile(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", re.DOTALL)


@dataclass(frozen=True)
class ResponseSections:
    perception: str = ""
    reasoning: str = ""
    prediction: str = ""
    answer: str = ""
    native_thinking: str = ""
    format_correct: bool = False
    used_native_answer: bool = False


def _tag(name: str) -> str:
    return rf"<{name}>(.*?)</{name}>"


def loose_section(response: str, name: str) -> str:
    """Extract a malformed section for rollout salvage, never for format credit."""
    for alias in _TAG_ALIASES[name]:
        match = re.search(_tag(alias), response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    for alias in _TAG_ALIASES[name]:
        match = re.search(
            rf"^\s*{alias}\s*:\s*(.+)$", response, re.MULTILINE | re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
    return ""


def _native_thinking_prefix(response: str, body_tag: str) -> tuple[str, str]:
    """Return (thinking, remainder) for explicit or template-opened thinking.

    Engines that enforce a thinking budget can insert ``</think>`` while the model later
    emits its own close as well. The final close before the structured body is therefore
    the real boundary; earlier closes remain part of the native-thinking transcript.
    """
    close = response.rfind("</think>")
    body_start = response.find(f"<{body_tag}>", close + len("</think>"))
    if close >= 0 and body_start >= 0:
        opening = re.match(r"^\s*<think>", response)
        start = opening.end() if opening else 0
        return response[start:close].strip(), response[close + len("</think>"):]
    return "", response


def parse_wm_sections(response: str, *, allow_native_thinking: bool = False) -> ResponseSections:
    """Parse the canonical WM suffix, optionally after native model thinking.

    Strict correctness requires the complete response, apart from surrounding
    whitespace, to use perception -> reasoning -> prediction -> answer.  Legacy tags
    and labels are read only so an environment may salvage an action while withholding
    format reward.
    """
    native_thinking, body = (
        _native_thinking_prefix(response, PERCEPTION)
        if allow_native_thinking
        else ("", response)
    )
    pattern = re.compile(
        rf"^\s*{_tag(PERCEPTION)}\s*{_tag(REASONING)}\s*"
        rf"{_tag(PREDICTION)}\s*{_tag(ANSWER)}\s*$",
        re.DOTALL,
    )
    match = pattern.match(body)
    if match:
        return ResponseSections(
            perception=match.group(1).strip(),
            reasoning=match.group(2).strip(),
            prediction=match.group(3).strip(),
            answer=match.group(4).strip(),
            native_thinking=native_thinking,
            format_correct=True,
        )
    return ResponseSections(
        perception=loose_section(response, PERCEPTION),
        reasoning=loose_section(response, REASONING),
        prediction=loose_section(response, PREDICTION),
        answer=loose_section(response, ANSWER),
        native_thinking=native_thinking,
    )


def parse_free_think_sections(response: str, *, allow_native_answer: bool = False) -> ResponseSections:
    """Parse ``<think>...</think><answer>...</answer>``.

    A native-thinking chat template may have emitted the opening ``<think>`` in the
    prompt, so response text beginning inside the block and later closing it is also a
    canonical free-think response.
    """
    reasoning, body = _native_thinking_prefix(response, ANSWER)
    answer_match = re.fullmatch(rf"\s*{_tag(ANSWER)}\s*", body, re.DOTALL)
    if answer_match and (reasoning or "</think>" in response):
        return ResponseSections(
            reasoning=reasoning,
            answer=answer_match.group(1).strip(),
            format_correct=True,
        )

    native = _NATIVE_BOX.search(body) if allow_native_answer else None
    # Once thinking closed, drafts inside it are not actions. Search the remainder first.
    answer = loose_section(body, ANSWER)
    if not answer and native:
        answer = native.group(1).strip()
    return ResponseSections(
        reasoning=reasoning or loose_section(response, REASONING),
        answer=answer,
        used_native_answer=native is not None and not loose_section(body, ANSWER),
    )


def parse_answer_sections(response: str, *, lenient: bool = False) -> ResponseSections:
    """Parse answer-only output; ``lenient`` permits surrounding native reasoning."""
    if lenient:
        match = re.fullmatch(r"\s*(.*?)<answer>(.*?)</answer>\s*", response, re.DOTALL)
        reasoning = match.group(1).strip() if match else ""
        answer_group = 2
    else:
        match = re.fullmatch(r"\s*<answer>(.*?)</answer>\s*", response, re.DOTALL)
        reasoning = ""
        answer_group = 1
    if match:
        return ResponseSections(
            reasoning=reasoning,
            answer=match.group(answer_group).strip(),
            format_correct=True,
        )
    return ResponseSections(
        reasoning=response.strip(),
        answer=loose_section(response, ANSWER),
    )


def canonical_wm(sections: ResponseSections) -> str:
    return (
        f"<perception>{sections.perception}</perception>"
        f"<reasoning>{sections.reasoning}</reasoning>"
        f"<prediction>{sections.prediction}</prediction>"
        f"<answer>{sections.answer}</answer>"
    )


def canonical_free_think(sections: ResponseSections) -> str:
    return f"<think>{sections.reasoning}</think><answer>{sections.answer}</answer>"


def split_actions(text: str, separator: str, maximum: int, *, lower: bool = True) -> list[str]:
    actions = [part.strip() for part in text.split(separator) if part.strip()][:maximum]
    return [action.lower() for action in actions] if lower else actions


__all__ = [
    "ANSWER",
    "ANSWER_FORMAT",
    "FREE_THINK_FORMAT",
    "PERCEPTION",
    "PREDICTION",
    "REASONING",
    "ResponseSections",
    "THINK",
    "WM_FORMAT",
    "canonical_free_think",
    "canonical_wm",
    "loose_section",
    "parse_answer_sections",
    "parse_free_think_sections",
    "parse_wm_sections",
    "split_actions",
]
