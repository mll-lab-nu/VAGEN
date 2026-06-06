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


def compress_history(messages):
    """Mirror WebAgent-R1's WebRLChatPromptConstructor: replace HTML in
    historical user messages with `** Simplified html **`, keeping only the
    most recent user message's real observation. System and assistant
    messages are untouched. Without this compression, multi-turn chat
    accumulates full HTML each turn and quickly exceeds 32K context.

    Wired in via `vagen/envs/webarena/agent_loop.py:WebArenaGymAgentLoop`,
    which subclasses `GymAgentLoop` and overrides `_handle_generating_state`
    to re-tokenize from compressed messages each turn. The webarena training
    scripts in `examples/train/webarena/*.sh` point
    `actor_rollout_ref.rollout.agent.agent_loop_config_path` to
    `vagen/envs/webarena/configs/agent.yaml`, which maps the `gym_agent`
    name to the webarena subclass.
    """
    if not messages:
        return messages
    user_idxs = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_idxs) <= 1:
        return messages
    last_user_idx = user_idxs[-1]
    out = []
    for i, m in enumerate(messages):
        if m.get("role") != "user" or i == last_user_idx:
            out.append(m)
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            text = "".join(
                blk.get("text", "") for blk in content
                if isinstance(blk, dict) and blk.get("type") == "text"
            )
        else:
            text = content or ""
        # Preserve "Task Instruction: ...\n\nRound N" prefix (model needs intent)
        if "Task Instruction" in text.split("\n\n", 1)[0]:
            parts = text.split("\n\n", 2)
            prefix = "\n\n".join(parts[:2]) if len(parts) >= 2 else parts[0]
            new_content = f"{prefix}\n\n** Simplified html **"
        else:
            # "Round N\n\n<html...>" — keep "Round N"
            first_line = text.split("\n", 1)[0]
            new_content = f"{first_line}\n\n** Simplified html **"
        out.append({"role": "user", "content": new_content})
    return out
