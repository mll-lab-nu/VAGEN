"""Parse XML-tag actions from agent output and execute them via Playwright."""

import re
from typing import Any, Dict, Optional

from playwright.async_api import Page


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

# Matches: click[123], type[42][hello world], scroll[down], goto[http://...], go_back, stop[answer]
_ACTION_RE = re.compile(
    r"(click|type|scroll|goto|go_back|stop)"  # action name
    r"(?:\[([^\]]*)\])?"                       # first arg (optional)
    r"(?:\[([^\]]*)\])?",                      # second arg (optional, for type)
    re.IGNORECASE,
)


def parse_action(action_str: str) -> Dict[str, Any]:
    """
    Parse agent output containing <think>...</think><action>...</action> tags.

    Returns dict with keys:
        - format_correct (bool)
        - think (str | None)
        - action_type (str | None): click | type | scroll | goto | go_back | stop
        - args (list[str]): parsed arguments
        - answer (str | None): for stop action
        - raw (str): original input
    """
    out: Dict[str, Any] = {
        "format_correct": False,
        "think": None,
        "action_type": None,
        "args": [],
        "answer": None,
        "raw": action_str,
    }

    s = action_str.strip()

    # Extract <think> (optional)
    m_think = re.search(r"<think>(.*?)</think>", s, re.DOTALL | re.IGNORECASE)
    if m_think:
        out["think"] = m_think.group(1).strip()

    # Extract <action> (required)
    m_action = re.search(r"<action>(.*?)</action>", s, re.DOTALL | re.IGNORECASE)
    if not m_action:
        return out

    action_body = m_action.group(1).strip()

    # Parse action body: action_name[arg1][arg2]
    m = _ACTION_RE.match(action_body)
    if not m:
        return out

    action_type = m.group(1).lower()
    arg1 = m.group(2)  # may be None
    arg2 = m.group(3)  # may be None

    out["format_correct"] = True
    out["action_type"] = action_type
    out["args"] = [a for a in (arg1, arg2) if a is not None]

    if action_type == "stop":
        out["answer"] = arg1

    return out


# ---------------------------------------------------------------------------
# Action execution via Playwright
# ---------------------------------------------------------------------------

async def execute_action(page: Page, parsed: Dict[str, Any]) -> Optional[str]:
    """
    Execute a parsed action on the Playwright page.

    Returns:
        Error message string if the action fails, None on success.
    """
    action_type = parsed.get("action_type")
    args = parsed.get("args", [])

    if not action_type or not parsed.get("format_correct"):
        return "invalid_action"

    try:
        if action_type == "click":
            if not args:
                return "click: missing element_id"
            eid = args[0].strip()
            locator = page.locator(f'[data-webarena-id="{eid}"]')
            await locator.click(timeout=5000)

        elif action_type == "type":
            if len(args) < 2:
                return "type: requires element_id and text"
            eid = args[0].strip()
            text = args[1]
            locator = page.locator(f'[data-webarena-id="{eid}"]')
            await locator.fill(text, timeout=5000)

        elif action_type == "scroll":
            direction = args[0].lower() if args else "down"
            delta = -500 if direction == "up" else 500
            await page.evaluate(f"window.scrollBy(0, {delta})")

        elif action_type == "goto":
            if not args:
                return "goto: missing url"
            url = args[0].strip()
            await page.goto(url, timeout=10000)

        elif action_type == "go_back":
            await page.go_back(timeout=10000)

        elif action_type == "stop":
            # No browser action needed; the env handles termination
            pass

        else:
            return f"unknown_action: {action_type}"

    except Exception as e:
        return f"action_error: {e}"

    return None
