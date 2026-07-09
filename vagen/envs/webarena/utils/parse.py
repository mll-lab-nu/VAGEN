"""Action parsing and URL mapping for webrl-format outputs.

Model output contract:
    <think> reasoning </think>
    <answer> do(action="Click", element="7") </answer>

or terminate with:
    <answer> exit(message="the answer") </answer>
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_EXIT_RE = re.compile(r'^\s*exit\s*\(\s*message\s*=\s*"(?P<msg>.*)"\s*\)\s*$', re.DOTALL)


def extract_answer(response: str) -> Optional[str]:
    """Extract the content of the last <answer>...</answer> block."""
    matches = _ANSWER_RE.findall(response)
    if not matches:
        return None
    return matches[-1].strip()


def parse_exit(answer: str) -> Optional[str]:
    """If answer is `exit(message="...")`, return the message. Else None."""
    m = _EXIT_RE.match(answer)
    if m is None:
        return None
    return m.group("msg")


def parse_response(response: str) -> Dict[str, Any]:
    """Parse a model response into structured fields.

    Returns a dict with:
      - `format_correct`: bool, whether <answer>...</answer> was found
      - `answer`: Optional[str], raw answer text (already stripped)
      - `is_exit`: bool
      - `exit_message`: Optional[str], if is_exit else None
    """
    answer = extract_answer(response)
    if answer is None:
        return {
            "format_correct": False,
            "answer": None,
            "is_exit": False,
            "exit_message": None,
        }
    exit_msg = parse_exit(answer)
    return {
        "format_correct": True,
        "answer": answer,
        "is_exit": exit_msg is not None,
        "exit_message": exit_msg,
    }


def map_url_to_local(text: str, url_mappings: Dict[str, str]) -> str:
    """Replace canonical URLs (e.g. http://reddit.com) in the text with the
    corresponding local deployment URL (e.g. http://localhost:9999).

    `url_mappings` maps local -> canonical, so we invert here.
    Also handles https variants.
    """
    if not text:
        return text
    for local, canonical in url_mappings.items():
        if canonical in text:
            text = text.replace(canonical, local)
        https_canonical = canonical.replace("http://", "https://")
        if https_canonical in text:
            text = text.replace(https_canonical, local)
    return text
