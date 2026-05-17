"""WebArena system prompt (webrl format).

Loaded directly from upstream `agent/prompts/jsons/p_webrl_chat_think.json`
to guarantee character-for-character match with the prompt used during SFT.
Falls back to a copy of the JSON content if the upstream path is unavailable.
"""

import json
import os

# Try to load from -rl upstream JSON for exact match.
_UPSTREAM_JSON = "/work/nvme/bgig/ryu4/WebAgent-R1-rl/WebAgent-R1/Eval/agent/prompts/jsons/p_webrl_chat_think.json"
try:
    with open(_UPSTREAM_JSON) as f:
        WEBARENA_SYS_PROMPT = json.load(f)["intro"]
except (FileNotFoundError, KeyError):
    # Fallback: hard-coded copy (kept in sync with upstream).
    WEBARENA_SYS_PROMPT = (
        "You are a professional web browsing agent assistant that can fulfill user's high-level instructions. "
        "Given simplified html of the browsed webpage at each step, you plan operations in python-style pseudo "
        "code using provided functions. \nYou first think about the reasoning process as an internal monologue "
        "and then decide an action. The reasoning process and answer are enclosed within <think> </think> and "
        "<answer> </answer> tags, respectively, i.e., responding in the following format: <think>\n...\n</think>\n"
        "<answer>\n...\n</answer>. "
        # NOTE: this fallback is incomplete — install the upstream JSON for the full prompt.
    )


def format_task_prompt(intent: str, round_idx: int, observation: str) -> str:
    """Format a user turn as WebRL-style: Task Instruction: ...\\n\\nRound N\\n\\n<obs>."""
    if round_idx == 0:
        return f"Task Instruction: {intent}\n\nRound {round_idx}\n\n{observation}"
    return f"Round {round_idx}\n\n{observation}"
