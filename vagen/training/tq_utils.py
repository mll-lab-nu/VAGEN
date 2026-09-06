"""Small, dependency-light helpers shared by VAGEN's TransferQueue path.

The actual TQ worker and trainer import :mod:`transfer_queue`, which is an optional
runtime dependency for the legacy trainer.  Keeping the data-contract helpers here
makes the important parts independently testable without starting Ray or a TQ server.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import torch


ROLLOUT_SOURCE = "__vagen_rollout_index__"


TRAJECTORY_EXTRA_FIELDS = (
    "episode_id",
    "group_idx",
    "traj_idx",
    "turn_idx",
    "conversation_id",
    "episode_turns",
    "response_spans",
    "ends_with_summary",
    "last_turn",
    "image_data",
    "rollout_metadata",
    "reward_extra_info",
)


def unwrap_non_tensor(value: Any) -> Any:
    """Remove TensorDict's scalar non-tensor wrapper when one is present."""
    return getattr(value, "data", value)


def trajectory_identity(kwargs: dict[str, Any]) -> tuple[Any, Any]:
    """Return VAGEN's prompt-group and within-group trajectory identifiers.

    Legacy rollout managers publish ``group_idx`` / ``traj_idx``.  V1 TQ workers
    publish the same two axes as ``uid`` / ``session_id``.  The mapping is semantic,
    not positional, and is therefore safe when async completion reorders episodes.
    """
    group_idx = kwargs.get("group_idx", kwargs.get("uid"))
    traj_idx = kwargs.get("traj_idx", kwargs.get("session_id"))
    if group_idx is None or traj_idx is None:
        raise KeyError(
            "a rollout needs either (group_idx, traj_idx) or V1's "
            "(uid, session_id) identity fields"
        )
    return unwrap_non_tensor(group_idx), unwrap_non_tensor(traj_idx)


def parse_tq_trajectory_key(key: str) -> tuple[str, int, int]:
    """Parse V1's ``{uid}_{session_id}_{output_index}`` trajectory key."""
    fields = str(key).rsplit("_", 2)
    if len(fields) != 3:
        raise ValueError(f"unexpected TransferQueue trajectory key: {key!r}")
    return fields[0], int(fields[1]), int(fields[2])


def inflight_is_stale(
    *, current_step: int, start_step: int, max_inflight_steps: int, partition_id: str
) -> bool:
    """Whether a training rollout has exceeded its allowed policy-version span."""
    return (
        partition_id == "train"
        and max_inflight_steps > 0
        and current_step - start_step + 1 > max_inflight_steps
    )


def token_level_reward_tensor(output: Any) -> torch.Tensor | None:
    """Validate and materialize the token reward published by a VAGEN env.

    This function intentionally does no tokenization and no reward computation.  The
    environment already aligned the vector to the engine's sampled token ids; the TQ
    boundary only verifies that the alignment survived transport.
    """
    extra_fields = getattr(output, "extra_fields", None) or {}
    rewards = extra_fields.get("per_token_reward")
    if rewards is None:
        return None

    response_ids = list(getattr(output, "response_ids"))
    if len(rewards) != len(response_ids):
        raise ValueError(
            "per_token_reward must align with the engine response ids: "
            f"got {len(rewards)} rewards for {len(response_ids)} tokens"
        )

    tensor = torch.as_tensor(rewards, dtype=torch.float32)
    scalar = getattr(output, "reward_score", None)
    if scalar is not None and not math.isclose(
        float(tensor.sum().item()), float(scalar), rel_tol=1e-5, abs_tol=1e-6
    ):
        raise ValueError(
            "reward_score must equal the transported per_token_reward sum: "
            f"{float(scalar)} != {float(tensor.sum().item())}"
        )
    return tensor


def _object_array(values: Iterable[Any]) -> np.ndarray:
    values = list(values)
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def trajectory_columns(
    extra_fields: Iterable[Any],
    *,
    keys: list[str],
    uids: Iterable[Any],
    padding: Iterable[bool] | None = None,
) -> dict[str, np.ndarray]:
    """Expand VAGEN metadata dictionaries into DataProto non-tensor columns.

    V1 stores ``AgentLoopOutput.extra_fields`` as one TQ field.  VAGEN's advantage
    estimators operate on DataProto columns, so this is the explicit boundary between
    those representations.  Synthetic V1 padding rows receive unique identities rather
    than inheriting the first real episode's metadata (which would merge two different
    response masks into one trajectory).
    """
    extras = [unwrap_non_tensor(value) for value in extra_fields]
    uid_values = [unwrap_non_tensor(value) for value in uids]
    if len(extras) != len(keys) or len(uid_values) != len(keys):
        raise ValueError("TQ metadata columns do not have the same row count")
    padding_values = list(padding) if padding is not None else [False] * len(keys)
    if len(padding_values) != len(keys):
        raise ValueError("padding metadata does not have the same row count as TQ keys")

    rows: list[dict[str, Any]] = []
    for key, uid, value, is_padding in zip(keys, uid_values, extras, padding_values, strict=True):
        _, session_id, _ = parse_tq_trajectory_key(key)
        row = dict(value) if isinstance(value, dict) else {}
        if is_padding:
            # ``upsample_batch_to_divisible_size`` copies the first sample's
            # extra_fields.  Keeping those identifiers would make trajectory packing
            # treat the zero-mask filler as a duplicate of a real row.
            row.update(
                episode_id=key,
                group_idx=uid,
                traj_idx=session_id,
                turn_idx=0,
                conversation_id=0,
                episode_turns=0,
                response_spans=[],
                ends_with_summary=False,
                last_turn=True,
                image_data=[],
                rollout_metadata={},
                reward_extra_info={},
            )
        else:
            row.setdefault("group_idx", uid)
            row.setdefault("traj_idx", session_id)
        rows.append(row)

    columns: dict[str, np.ndarray] = {"uid": _object_array(uid_values)}
    for name in TRAJECTORY_EXTRA_FIELDS:
        values = [row.get(name) for row in rows]
        if any(value is not None for value in values):
            columns[name] = _object_array(values)
    return columns
