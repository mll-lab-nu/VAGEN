import re
from typing import Dict, List
from PIL import Image
import numpy as np

# <think>...</think> then, after any amount of visible prose, <answer>...</answer>.
#
# ★ The opening `<think>` is optional, because on a native-thinking family the chat
# template writes it into the *generation prompt* -- the response begins already inside
# the block and its first tag is the closing one (see the vagen-think-token notes:
# Qwen3.5 emits `</think>` only, Qwen2.5-VL emits both as ordinary text). Demanding the
# opening tag makes the format unsatisfiable on exactly the models it is meant for.
#
# ★ The answer must come *after* `</think>`. Thinking models draft candidate answers
# mid-reasoning -- measured on Qwen3.5, 8 of 40 truncated rambles contained an `<answer>`
# tag inside the unclosed block. A parser that takes the first one anywhere rewards a
# trace that never terminated, which is the opposite of the pressure this arm needs.
#
# ★ Split with `str.find` rather than one regex spanning both tags. The obvious pattern,
# `(?:<think>)?(.*?)</think>(.*?)<answer>(.*?)</answer>`, backtracks quadratically on the
# input that matters most: a response with no `</think>` at all makes the engine rescan to
# the end from every start position. These responses run to 90k characters, so that is
# ~8e9 steps -- it does not return. `find` is linear and the tags are unambiguous.
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_NATIVE_BOX_TAG = re.compile(
    r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", re.DOTALL
)


def parse_free_think(response: str, action_sep: str = ",", max_actions: int = 3) -> Dict:
    """
    Parse free_think format response: <think>...</think><answer>...</answer>

    Two-tier like ``parse_wm``: the shape above decides ``format_correct``, extraction
    falls back to a loose search so a malformed turn still yields its action instead of
    discarding the episode over punctuation.

    Args:
        response: Raw LLM response string
        action_sep: Separator between actions
        max_actions: Maximum number of actions to extract

    Returns:
        Dict containing parsed components and validation info
    """
    closed = response.find("</think>")
    opened = response.find("<think>")
    # Content before the close is the reasoning. If the model typed an opening tag of its
    # own before that point, start after it; on a native-thinking family there is none,
    # because the chat template already emitted it into the prompt.
    start = opened + len("<think>") if 0 <= opened < closed else 0

    after_close = _ANSWER_TAG.search(response, closed + len("</think>")) if closed >= 0 else None
    # GLM-4.6V's official post-training protocol uses a pair of native box tokens for
    # the final answer. Treat that pair as the model-family equivalent of <answer>, but
    # prefer a literal <answer> when both occur (it may itself be wrapped by the box).
    native_box = (
        _NATIVE_BOX_TAG.search(response, closed + len("</think>"))
        if closed >= 0 and after_close is None
        else None
    )
    action_match = after_close or native_box
    format_correct = action_match is not None

    if not format_correct:
        # Salvage what is there. format_correct stays False either way.
        think_content = (response[start:closed] if closed >= 0 else response[start:]).strip()
        action_content = _loose_section(response, "answer")
    else:
        think_content = response[start:closed].strip()
        action_content = action_match.group(1).strip()

    actions = [a.strip().lower() for a in action_content.split(action_sep) if a.strip()]
    if len(actions) > max_actions:
        actions = actions[:max_actions]
        action_content = action_sep.join(actions)

    # Keep GLM's model-native tokens intact. The parser may interpret its box as the
    # action marker, but PPO must train on exactly the sequence the model sampled.
    llm_response = (
        response
        if native_box is not None
        else f"<think>{think_content}</think><answer>{action_content}</answer>"
    )

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        # The other formats expose the model's reasoning under this name; keep the alias so
        # anything reading `reasoning_content` works across formats.
        "reasoning_content": think_content,
        "observation_content": "",
        "prediction_content": "",
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }

# A section, however the model chose to mark it: the tag it was asked for, or the
# plain-text label small models fall back to from ReAct-style pretraining.
_SECTION_ALIASES = {
    "observation": ("observation",),
    "think": ("think", "thought", "reasoning"),
    "answer": ("answer", "action"),
    "prediction": ("prediction",),
}


def _loose_section(response: str, name: str) -> str:
    """Pull one section out on its own, tag or label, without requiring the others."""
    for alias in _SECTION_ALIASES[name]:
        m = re.search(rf"<{alias}>(.*?)</{alias}>", response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    for alias in _SECTION_ALIASES[name]:
        # `Action: Up, Left` -- to the end of the line, since the label form is not closed.
        m = re.search(rf"^\s*{alias}\s*:\s*(.+)$", response, re.MULTILINE | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def parse_wm(response: str, action_sep: str = ",", max_actions: int = 3) -> Dict:
    """
    Parse wm_new format response:
    <observation>...</observation>
    <think>...</think>
    <answer>...</answer>
    <prediction>...</prediction>

    The strict pattern above decides ``format_correct``, which is what the format metric
    and any format reward read. Extraction is separate and forgiving: a response that
    reasoned correctly and labelled it ``Action:`` instead of ``<answer>`` still yields
    its action, so the episode continues instead of dying on syntax. Measured on the
    base model, half of all episodes reached zero usable actions this way -- half the
    batch discarded over punctuation, while the model was doing the task.

    The two are deliberately not the same test. Executing the action keeps the data;
    ``format_correct`` staying strict keeps the pressure to write the tags.
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
        # Salvage what is there. format_correct stays False either way.
        observation_content = _loose_section(response, "observation")
        think_content = _loose_section(response, "think")
        prediction_content = _loose_section(response, "prediction")
        action_content = _loose_section(response, "answer")
        actions = [a.strip().lower() for a in action_content.split(action_sep) if a.strip()]
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = action_sep.join(actions)
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

def parse_answer_only(response: str, action_sep: str = ",", max_actions: int = 3) -> Dict:
    """Read the action out of `<answer>...</answer>` and treat the rest as reasoning.

    For models whose thinking is native: everything before the tag is their own reasoning,
    recorded as reasoning_content but not required to follow any shape. ``format_correct``
    therefore asks only that a well-formed `<answer>` exists -- there is nothing else the
    format demands, so anything stricter would penalise a model that answered correctly.
    """
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    format_correct = match is not None

    action_content = match.group(1).strip() if match else _loose_section(response, "answer")
    reasoning_content = response[: match.start()].strip() if match else response.strip()

    actions = [a.strip().lower() for a in action_content.split(action_sep) if a.strip()]
    if len(actions) > max_actions:
        actions = actions[:max_actions]
        action_content = action_sep.join(actions)

    return {
        "llm_raw_response": response,
        "llm_response": response,
        "observation_content": "",
        "reasoning_content": reasoning_content,
        "prediction_content": "",
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }


def parse_free_wm(response: str, action_sep: str = ",", max_actions: int = 3) -> Dict:
    """
    Parse free_wm format response:
    <observation>...</observation> ... <answer>...</answer> ... <prediction>...</prediction>

    Like wm but without the <think> tag. Free-form reasoning text is allowed
    between the tags and is captured as reasoning_content.
    """
    pattern = (
        r'<observation>(.*?)</observation>'
        r'(.*?)'
        r'<answer>(.*?)</answer>'
        r'(.*?)'
        r'<prediction>(.*?)</prediction>'
    )

    match = re.search(pattern, response, re.DOTALL)
    format_correct = match is not None

    if not match:
        observation_content = ""
        prediction_content = ""
        action_content = ""
        reasoning_content = ""
        actions: List[str] = []
    else:
        observation_content = match.group(1).strip()
        reasoning_before_answer = match.group(2).strip()
        action_content = match.group(3).strip()
        reasoning_after_answer = match.group(4).strip()
        prediction_content = match.group(5).strip()

        # Combine any free-form reasoning between tags
        reasoning_parts = [p for p in [reasoning_before_answer, reasoning_after_answer] if p]
        reasoning_content = " ".join(reasoning_parts)

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
        f"<answer>{action_content}</answer>"
        f"<prediction>{prediction_content}</prediction>"
    )

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "observation_content": observation_content,
        "think_content": reasoning_content,
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
    elif prompt_format == "free_wm":
        return parse_free_wm(response, action_sep, max_actions)
    elif prompt_format == "answer":
        return parse_answer_only(response, action_sep, max_actions)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")
    
def numpy_to_pil(numpy_array: np.ndarray) -> Image.Image:
    """Convert numpy (H, W, 3) to PIL.Image in RGB."""
    if numpy_array.shape[-1] == 3:
        return Image.fromarray(numpy_array.astype(np.uint8), mode="RGB")
    raise ValueError(f"Unsupported channels: {numpy_array.shape[-1]}. Expected 3 (RGB).")
