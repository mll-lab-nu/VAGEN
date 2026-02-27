"""Task success evaluation for WebArena tasks.

WebArena tasks define evaluation criteria in their JSON config. Common types:
- string_match: Check if page content or agent answer matches expected string
- url_match: Check if current URL matches expected pattern
- program_html: Run a JS snippet on the page that returns pass/fail

This module wraps those evaluators for use in the VAGEN environment.
"""

import re
from typing import Any, Dict, Optional

from playwright.async_api import Page


async def evaluate_string_match(
    page: Page,
    eval_config: Dict[str, Any],
    agent_answer: Optional[str] = None,
) -> bool:
    """Check if agent answer or page content matches the expected string."""
    expected = eval_config.get("reference_answers", eval_config.get("ref", ""))
    if isinstance(expected, list):
        expected_list = expected
    else:
        expected_list = [expected]

    must_include = eval_config.get("must_include", [])
    if isinstance(must_include, str):
        must_include = [must_include]

    # Check agent's stop[answer] if available
    if agent_answer:
        answer_lower = agent_answer.lower().strip()
        for exp in expected_list:
            if exp.lower().strip() == answer_lower:
                return True
            if exp.lower().strip() in answer_lower:
                return True

    # Check must_include against agent answer
    if must_include and agent_answer:
        answer_lower = agent_answer.lower()
        return all(m.lower() in answer_lower for m in must_include)

    return False


async def evaluate_url_match(
    page: Page,
    eval_config: Dict[str, Any],
) -> bool:
    """Check if the current URL matches the expected pattern."""
    expected = eval_config.get("reference_url", eval_config.get("ref", ""))
    url_note = eval_config.get("url_note", "EXACT")
    current_url = page.url

    if url_note == "EXACT":
        return current_url.rstrip("/") == expected.rstrip("/")
    elif url_note == "GOLD in PRED":
        return expected.rstrip("/") in current_url
    elif url_note == "PRED in GOLD":
        return current_url.rstrip("/") in expected
    else:
        # Default: substring match
        return expected.rstrip("/") in current_url


async def evaluate_program_html(
    page: Page,
    eval_config: Dict[str, Any],
) -> bool:
    """Run a JS evaluation script on the page. Returns True if the script returns a truthy value."""
    js_code = eval_config.get("program_html", "")
    if not js_code:
        return False

    try:
        result = await page.evaluate(js_code)
        return bool(result)
    except Exception:
        return False


async def evaluate_task(
    page: Page,
    task_config: Dict[str, Any],
    agent_answer: Optional[str] = None,
) -> bool:
    """
    Evaluate whether the current task is completed successfully.

    Supports multiple evaluation types from WebArena task configs.
    If eval is a list, ALL conditions must pass.
    """
    eval_types = task_config.get("eval", {}).get("eval_types", [])
    eval_configs = task_config.get("eval", {}).get("configs", [])

    # Handle the case where eval is a flat dict
    if not eval_types and "eval_type" in task_config.get("eval", {}):
        eval_types = [task_config["eval"]["eval_type"]]
        eval_configs = [task_config["eval"]]

    # If no eval config, try simple reference_answers check
    if not eval_types:
        ref = task_config.get("eval", {}).get("reference_answers", None)
        if ref is not None and agent_answer:
            return await evaluate_string_match(page, task_config.get("eval", {}), agent_answer)
        return False

    results = []
    for eval_type, eval_config in zip(eval_types, eval_configs):
        eval_type = eval_type.lower()
        if eval_type == "string_match":
            results.append(await evaluate_string_match(page, eval_config, agent_answer))
        elif eval_type == "url_match":
            results.append(await evaluate_url_match(page, eval_config))
        elif eval_type == "program_html":
            results.append(await evaluate_program_html(page, eval_config))
        else:
            # Unknown eval type, skip
            results.append(False)

    return all(results) if results else False
