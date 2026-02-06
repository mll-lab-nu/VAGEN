# All comments are in English.
from __future__ import annotations
import os
import json
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional

# ------------------------
# Helpers
# ------------------------

def _safe_jsonable(x: Any) -> Any:
    """Best-effort make data JSON-friendly."""
    try:
        json.dumps(x)
        return x
    except Exception:
        return repr(x)

def _success_from_info(info: Dict[str, Any], success_keys: List[str]) -> Optional[bool]:
    """
    Read per-turn success directly from info using provided keys.
    Return None if no explicit success key is present.
    """
    for k in success_keys:
        if k in info:
            try:
                return bool(info[k])
            except Exception:
                return False
    return None

def _build_per_turn_with_turn0(
    rewards: List[float],
    infos: List[Dict[str, Any]],
    success_keys: List[str],
) -> List[Dict[str, Any]]:
    """
    Build per_turn list starting with turn=0 for reset info (reward=None),
    then turns 1..T for each step aligned with rewards.
    Alignment rule:
      - If len(infos) == len(rewards) + 1 -> infos[0] is reset info; step i uses infos[i+1].
      - Else -> step i uses infos[i] (best-effort).
    Success rule:
      - If the selected info contains any success key, use it directly.
      - Otherwise, success=False (no inference).
    """
    per_turn: List[Dict[str, Any]] = []

    # 1) turn 0 (reset info), if present
    if len(infos) >= 1:
        s0 = _success_from_info(infos[0], success_keys)
        per_turn.append({
            "turn": 0,
            "reward": None,                  # reset has no reward
            "success": bool(s0) if s0 is not None else False,
            "info": _safe_jsonable(infos[0]),
        })

    # 2) turns 1..T
    T = len(rewards)
    if len(infos) == T + 1:
        info_offset = 1
    else:
        info_offset = 0

    for i, rew in enumerate(rewards):
        j = i + info_offset
        info_i = infos[j] if j < len(infos) else {}
        s_i = _success_from_info(info_i, success_keys)
        per_turn.append({
            "turn": i + 1,
            "reward": float(rew),
            "success": bool(s_i) if s_i is not None else False,
            "info": _safe_jsonable(info_i),
        })

    return per_turn


# ------------------------
# Legacy in-memory summary (kept)
# ------------------------

def write_rollouts_summary(
    results: List[Dict[str, Any]],
    dump_dir: str = "./rollouts",
    filename: str = "summary.json",
    success_keys: Optional[List[str]] = None,
) -> str:
    """
    Aggregate summary from in-memory results.
    Now outputs per_turn starting with turn=0 for reset info (reward=None).
    Success per turn is taken directly from info if present; otherwise False.
    """
    os.makedirs(dump_dir, exist_ok=True)
    success_keys = success_keys or ["success", "is_success", "solved"]

    episodes = []
    succ_cnt = 0
    sum_cum_reward = 0.0
    sum_turns = 0
    error_rollouts: List[str] = []
    per_tag_data: Dict[int, Dict[str, Any]] = {}

    for r in results:
        rewards = r.get("rewards") or []
        infos = r.get("infos") or []

        per_turn = _build_per_turn_with_turn0(rewards, infos, success_keys)

        ep_success = bool(r.get("success") or False)
        ep = {
            "rollout_id": r.get("rollout_id"),
            "seed": r.get("seed"),
            "num_turns": int(r.get("num_turns") or 0),
            "terminated": bool(r.get("terminated") or False),
            "finish_reason": r.get("finish_reason"),
            "success": ep_success,
            "cumulative_reward": float(r.get("cumulative_reward") or 0.0),
            "per_turn": per_turn,
        }
        if r.get("error_details") is not None:
            ep["error_details"] = _safe_jsonable(r.get("error_details"))
        tag_val = r.get("tag_id")
        env_name = r.get("env_name")
        split = r.get("split")
        max_turns_meta = r.get("max_turns")
        if tag_val is not None:
            ep["tag_id"] = tag_val  # Keep original type (int or str)
        if env_name is not None:
            ep["env_name"] = env_name
        if split is not None:
            ep["split"] = split
        if max_turns_meta is not None:
            ep["max_turns"] = int(max_turns_meta)
        episodes.append(ep)

        if ep["success"]:
            succ_cnt += 1
        sum_cum_reward += ep["cumulative_reward"]
        sum_turns += ep["num_turns"]

        # Count errors (exclude normal endings like done/max_turns/skipped resumes)
        if ep["finish_reason"] not in ("done", "max_turns", "skipped_resume"):
            rid = ep.get("rollout_id")
            if rid:
                error_rollouts.append(rid)

        if tag_val is not None:
            tag_id = tag_val  # Keep original type (int or str)
            tag_state = per_tag_data.setdefault(
                tag_id,
                {
                    "episodes": [],
                    "succ_cnt": 0,
                    "reward_sum": 0.0,
                    "turn_sum": 0,
                    "error_rollouts": set(),
                },
            )
            tag_state["episodes"].append(ep)
            if ep["success"]:
                tag_state["succ_cnt"] += 1
            tag_state["reward_sum"] += ep["cumulative_reward"]
            tag_state["turn_sum"] += ep["num_turns"]
            if ep["finish_reason"] not in ("done", "max_turns", "skipped_resume"):
                rid = ep.get("rollout_id")
                if rid:
                    tag_state["error_rollouts"].add(rid)

    n = len(results)
    summary = {
        "created_at": dt.datetime.now().isoformat(),
        "n_episodes": n,
        "success_rate": (succ_cnt / n) if n else 0.0,
        "avg_cumulative_reward": (sum_cum_reward / n) if n else 0.0,
        "avg_turns": (sum_turns / n) if n else 0.0,
        "error_rollouts": error_rollouts,
        "episodes": episodes,
    }

    if per_tag_data:
        per_tag_summary: Dict[str, Any] = {}
        for tag_id, state in per_tag_data.items():
            eps = state["episodes"]
            count = len(eps)
            key = f"tag_{tag_id}"
            per_tag_summary[key] = {
                "created_at": summary["created_at"],
                "n_episodes": count,
                "success_rate": (state["succ_cnt"] / count) if count else 0.0,
                "avg_cumulative_reward": (state["reward_sum"] / count) if count else 0.0,
                "avg_turns": (state["turn_sum"] / count) if count else 0.0,
                "error_rollouts": sorted(state["error_rollouts"]),
                "episodes": eps,
            }
        summary["per_tag"] = per_tag_summary

    out_path = os.path.join(dump_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return out_path


# ------------------------
# Preferred summary by scanning dump_dir
# ------------------------

def write_rollouts_summary_from_dump(
    dump_dir: str,
    filename: str = "summary.json",
    success_keys: Optional[List[str]] = None,
) -> str:
    """
    Aggregate summary by scanning dump_dir for metrics.json files.
    Outputs per_turn starting with turn=0 for reset info (reward=None).
    Success per turn is taken directly from info if present; otherwise False.
    """
    success_keys = success_keys or ["success", "is_success", "solved"]
    root = Path(dump_dir)
    os.makedirs(root, exist_ok=True)

    episodes: List[Dict[str, Any]] = []
    succ_cnt = 0
    sum_cum_reward = 0.0
    sum_turns = 0
    error_rollouts: List[str] = []
    per_tag_data: Dict[int, Dict[str, Any]] = {}

    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            # Skip corrupted metrics
            continue

        rewards = m.get("rewards") or []
        infos = m.get("infos") or []

        per_turn = _build_per_turn_with_turn0(rewards, infos, success_keys)

        rollout_id = m.get("rollout_id") or run_dir.name
        ep_success = bool(m.get("success") or False)
        ep = {
            "rollout_id": rollout_id,
            "seed": m.get("seed"),
            "num_turns": int(m.get("num_turns") or 0),
            "terminated": bool(m.get("terminated") or False),
            "finish_reason": m.get("finish_reason"),
            "success": ep_success,
            "cumulative_reward": float(m.get("cumulative_reward") or 0.0),
            "per_turn": per_turn,
        }
        if m.get("error_details") is not None:
            ep["error_details"] = _safe_jsonable(m.get("error_details"))
        tag_val = m.get("tag_id")
        env_name = m.get("env_name")
        split = m.get("split")
        max_turns_meta = m.get("max_turns")
        if tag_val is not None:
            ep["tag_id"] = tag_val  # Keep original type (int or str)
        if env_name is not None:
            ep["env_name"] = env_name
        if split is not None:
            ep["split"] = split
        if max_turns_meta is not None:
            try:
                ep["max_turns"] = int(max_turns_meta)
            except Exception:
                pass
        episodes.append(ep)

        if ep["success"]:
            succ_cnt += 1
        sum_cum_reward += ep["cumulative_reward"]
        sum_turns += ep["num_turns"]

        if ep["finish_reason"] not in ("done", "max_turns", "skipped_resume"):
            error_rollouts.append(rollout_id)

        if tag_val is not None:
            tag_id = tag_val  # Keep original type (int or str)
            tag_state = per_tag_data.setdefault(
                tag_id,
                {
                    "episodes": [],
                    "succ_cnt": 0,
                    "reward_sum": 0.0,
                    "turn_sum": 0,
                    "error_rollouts": set(),
                },
            )
            tag_state["episodes"].append(ep)
            if ep["success"]:
                tag_state["succ_cnt"] += 1
            tag_state["reward_sum"] += ep["cumulative_reward"]
            tag_state["turn_sum"] += ep["num_turns"]
            if ep["finish_reason"] not in ("done", "max_turns", "skipped_resume"):
                tag_state["error_rollouts"].add(rollout_id)

    n = len(episodes)
    summary = {
        "created_at": dt.datetime.now().isoformat(),
        "n_episodes": n,
        "success_rate": (succ_cnt / n) if n else 0.0,
        "avg_cumulative_reward": (sum_cum_reward / n) if n else 0.0,
        "avg_turns": (sum_turns / n) if n else 0.0,
        "error_rollouts": error_rollouts,
        "episodes": episodes,
    }

    if per_tag_data:
        per_tag_summary: Dict[str, Any] = {}
        for tag_id, state in per_tag_data.items():
            eps = state["episodes"]
            count = len(eps)
            key = f"tag_{tag_id}"
            per_tag_summary[key] = {
                "created_at": summary["created_at"],
                "n_episodes": count,
                "success_rate": (state["succ_cnt"] / count) if count else 0.0,
                "avg_cumulative_reward": (state["reward_sum"] / count) if count else 0.0,
                "avg_turns": (state["turn_sum"] / count) if count else 0.0,
                "error_rollouts": sorted(state["error_rollouts"]),
                "episodes": eps,
            }
        summary["per_tag"] = per_tag_summary

    out_path = str(root / filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return out_path
