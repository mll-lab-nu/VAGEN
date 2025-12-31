from __future__ import annotations

from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto

PAD_TOKEN_ID = 0


def _as_1d_object_array(items: List[Any]) -> np.ndarray:
    """
    Force a 1D numpy object array, where each element is a Python object
    (e.g., list/dict), avoiding numpy auto-expanding list-of-lists to 2D.
    """
    arr = np.empty((len(items),), dtype=object)
    for i, v in enumerate(items):
        arr[i] = v
    return arr


def concat_val_multi_turn(
    test_output_gen_batch: DataProto,
    test_gen_batch: DataProto,
) -> DataProto:
    """
    Turn -> trajectory concatenation and STRICT reorder by test_gen_batch.uid,
    where uid == group_idx.

    STRICT CONSTRAINTS:
      - After concat, #trajectories MUST equal len(test_gen_batch)
      - Multiset(uid) MUST equal multiset(group_idx after concat)
      - Reorder follows gen_uid order; within same uid consume by traj_idx ascending
      - No placeholders; mismatch => assert

    reward_extra_info RULE:
      - Input MUST have nt["reward_extra_info"].
      - For each trajectory, reward_extra_info is taken from the MAX turn_idx (last turn).
      - Then all (k, v) in reward_extra_info are also copied to top-level non_tensor_entry.
      - reward_extra_info itself is kept as well.
    """
    n = len(test_output_gen_batch)
    if n == 0:
        return test_output_gen_batch[:0]

    assert "uid" in test_gen_batch.non_tensor_batch, (
        "concat_val_multi_turn: test_gen_batch.non_tensor_batch must contain key 'uid'"
    )

    nt = test_output_gen_batch.non_tensor_batch
    assert "reward_extra_info" in nt, (
        "concat_val_multi_turn: non_tensor_batch must contain key 'reward_extra_info' (no legacy traj_success support)"
    )

    group_arr = nt["group_idx"]
    traj_arr = nt["traj_idx"]
    turn_arr = nt.get("turn_idx", [0] * n)

    # ------------------------------------------------------------------
    # 1) Group turns by (group_idx, traj_idx)
    # ------------------------------------------------------------------
    trajectory_groups: Dict[Tuple[str, int], List[Tuple[int, int]]] = defaultdict(list)
    for i in range(n):
        g = str(group_arr[i])
        t = int(traj_arr[i])
        ti = int(turn_arr[i])
        trajectory_groups[(g, t)].append((ti, i))

    # Sort turns inside each trajectory by turn_idx
    for k in trajectory_groups:
        trajectory_groups[k].sort(key=lambda x: x[0])

    concatenated: List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]] = []

    # ------------------------------------------------------------------
    # 2) Concatenate per trajectory
    # ------------------------------------------------------------------
    for (group_idx_str, traj_idx), turns in sorted(trajectory_groups.items()):
        first_i = turns[0][1]
        concat_prompt = test_output_gen_batch.batch["prompts"][first_i]

        resp_parts: List[torch.Tensor] = []
        mask_parts: List[torch.Tensor] = []
        rm_parts: List[torch.Tensor] = []

        for j, (_, i) in enumerate(turns):
            resp = test_output_gen_batch.batch["responses"][i]

            if "response_mask" in test_output_gen_batch.batch:
                mask = test_output_gen_batch.batch["response_mask"][i]
            else:
                mask = torch.ones_like(resp)

            rm = test_output_gen_batch.batch["rm_scores"][i]

            resp_parts.append(resp)
            mask_parts.append(mask)
            rm_parts.append(rm)

            # insert next prompt segment (as-is)
            if j < len(turns) - 1:
                next_i = turns[j + 1][1]
                next_prompt = test_output_gen_batch.batch["prompts"][next_i]

                resp_parts.append(next_prompt)
                mask_parts.append(torch.zeros_like(next_prompt))
                rm_parts.append(torch.zeros_like(next_prompt, dtype=rm.dtype, device=rm.device))

        concat_response = torch.cat(resp_parts, dim=0)
        concat_response_mask = torch.cat(mask_parts, dim=0)
        concat_rm_scores = torch.cat(rm_parts, dim=0)

        # images: simply concatenate lists
        merged_images: List[Any] = []
        if "image_data" in nt:
            img_arr = nt["image_data"]
            for _, i in turns:
                v = img_arr[i]
                if v is None:
                    continue
                if isinstance(v, (list, tuple, np.ndarray)):
                    merged_images.extend(list(v))
                else:
                    merged_images.append(v)

        # reward_extra_info: take from LAST turn (max turn_idx)
        last_turn_i = turns[-1][1]
        rei = nt["reward_extra_info"][last_turn_i]
        assert rei is not None and isinstance(rei, dict), (
            "concat_val_multi_turn: reward_extra_info per row must be a dict (and not None)"
        )
        reward_extra_info: Dict[str, Any] = dict(rei)  # copy

        batch_entry = {
            "prompts": concat_prompt,
            "responses": concat_response,
            "response_mask": concat_response_mask,
            "rm_scores": concat_rm_scores,
        }

        non_tensor_entry: Dict[str, Any] = {
            "group_idx": group_idx_str,
            "traj_idx": int(traj_idx),
            "image_data": merged_images,
            "reward_extra_info": reward_extra_info,
        }

        # Copy all reward_extra_info kv to top-level
        for k, v in reward_extra_info.items():
            assert k not in non_tensor_entry, (
                f"concat_val_multi_turn: reward_extra_info key '{k}' conflicts with non_tensor_entry keys"
            )
            non_tensor_entry[k] = v

        concatenated.append((batch_entry, non_tensor_entry))

    if not concatenated:
        return test_output_gen_batch[:0]

    # ------------------------------------------------------------------
    # 2.5) STRICT alignment with test_gen_batch.uid (uid == group_idx)
    # ------------------------------------------------------------------
    gen_uid = [str(x) for x in test_gen_batch.non_tensor_batch["uid"]]
    target_n = len(test_gen_batch)

    bucket: Dict[str, List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]] = defaultdict(list)
    for be, nte in concatenated:
        bucket[str(nte["group_idx"])].append((be, nte))
    for uid in bucket:
        bucket[uid].sort(key=lambda x: int(x[1]["traj_idx"]))

    num_traj = sum(len(v) for v in bucket.values())
    assert num_traj == target_n, (
        "concat_val_multi_turn: trajectory-level count mismatch.\n"
        f"  num_traj_after_concat={num_traj}\n"
        f"  len(test_gen_batch)={target_n}\n"
        "Hint: test_gen_batch.uid must be TRAJECTORY-level, not TURN-level."
    )

    expected_uid_counter = Counter({uid: len(v) for uid, v in bucket.items()})
    actual_uid_counter = Counter(gen_uid)
    assert actual_uid_counter == expected_uid_counter, (
        "concat_val_multi_turn: uid multiset mismatch (uid == group_idx).\n"
        f"  expected(from concat)={dict(expected_uid_counter)}\n"
        f"  actual(test_gen_batch)={dict(actual_uid_counter)}\n"
        "Hint: test_gen_batch.uid must repeat per-trajectory under the same uid."
    )

    reordered: List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]] = []
    for i, uid in enumerate(gen_uid):
        assert uid in bucket and len(bucket[uid]) > 0, (
            f"concat_val_multi_turn: cannot find trajectory for uid={uid} (index={i})."
        )
        reordered.append(bucket[uid].pop(0))

    leftover = {uid: len(v) for uid, v in bucket.items() if len(v) > 0}
    assert not leftover, (
        "concat_val_multi_turn: extra trajectories not matched by test_gen_batch.uid.\n"
        f"  leftover={leftover}"
    )

    assert len(reordered) == target_n
    concatenated = reordered

    # ------------------------------------------------------------------
    # 3) Pad to same length per key, then stack
    # ------------------------------------------------------------------
    def _pad_1d(t: torch.Tensor, max_len: int, kind: str) -> torch.Tensor:
        cur = int(t.shape[0])
        if cur == max_len:
            return t
        pad = max_len - cur
        if kind in ("response_mask", "rm_scores"):
            padding = torch.zeros((pad,), dtype=t.dtype, device=t.device)
        else:
            padding = torch.full((pad,), PAD_TOKEN_ID, dtype=t.dtype, device=t.device)
        return torch.cat([t, padding], dim=0)

    keys = ["prompts", "responses", "response_mask", "rm_scores"]
    stacked_batch: Dict[str, torch.Tensor] = {}
    for k in keys:
        vals = [be[k] for be, _ in concatenated]
        max_len = max(int(v.shape[0]) for v in vals)
        vals = [_pad_1d(v, max_len, k) for v in vals]
        stacked_batch[k] = torch.stack(vals, dim=0)

    # Dynamic non-tensor keys copied from reward_extra_info
    base_nt_keys = {"group_idx", "traj_idx", "image_data", "reward_extra_info"}
    extra_keys: List[str] = []
    seen = set()
    for _, nte in concatenated:
        for k in nte.keys():
            if k in base_nt_keys:
                continue
            if k not in seen:
                seen.add(k)
                extra_keys.append(k)

    stacked_non_tensor: Dict[str, np.ndarray] = {
        "group_idx": _as_1d_object_array([nte["group_idx"] for _, nte in concatenated]),
        "traj_idx": _as_1d_object_array([nte["traj_idx"] for _, nte in concatenated]),
        "image_data": _as_1d_object_array([nte["image_data"] for _, nte in concatenated]),
        "reward_extra_info": _as_1d_object_array([nte["reward_extra_info"] for _, nte in concatenated]),
    }

    for k in extra_keys:
        stacked_non_tensor[k] = _as_1d_object_array([nte.get(k, None) for _, nte in concatenated])

    out = DataProto(
        batch=TensorDict(stacked_batch, batch_size=(len(concatenated),)),
        non_tensor_batch=stacked_non_tensor,
        meta_info=getattr(test_output_gen_batch, "meta_info", {}),
    )

    # final alignment: out.group_idx[i] == test_gen_batch.uid[i]
    for i in range(len(out)):
        assert str(out.non_tensor_batch["group_idx"][i]) == gen_uid[i], (
            f"concat_val_multi_turn: order mismatch at i={i}: "
            f"out.group_idx={out.non_tensor_batch['group_idx'][i]} != test_gen_batch.uid={gen_uid[i]}"
        )

    return out


# -----------------------------
# Test helpers
# -----------------------------
def _pad_1d_int(seqs: List[List[int]], pad_id: int = PAD_TOKEN_ID) -> torch.Tensor:
    max_len = max(len(x) for x in seqs) if seqs else 0
    out: List[torch.Tensor] = []
    for x in seqs:
        if len(x) < max_len:
            x = x + [pad_id] * (max_len - len(x))
        out.append(torch.tensor(x, dtype=torch.long))
    return torch.stack(out, dim=0)


def _pad_1d_float(seqs: List[List[float]], pad_value: float = 0.0) -> torch.Tensor:
    max_len = max(len(x) for x in seqs) if seqs else 0
    out: List[torch.Tensor] = []
    for x in seqs:
        if len(x) < max_len:
            x = x + [pad_value] * (max_len - len(x))
        out.append(torch.tensor(x, dtype=torch.float32))
    return torch.stack(out, dim=0)


def _make_dataproto(
    prompts: List[List[int]],
    responses: List[List[int]],
    response_mask: Optional[List[List[int]]] = None,
    rm_scores: Optional[List[List[float]]] = None,
    group_idx: Optional[List[Any]] = None,
    traj_idx: Optional[List[Any]] = None,
    turn_idx: Optional[List[Any]] = None,
    image_data: Optional[List[Any]] = None,
    reward_extra_info: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> DataProto:
    """
    Make a padded DataProto for tests.
    non_tensor fields are stored as 1D object arrays.
    """
    B = len(prompts)
    assert len(responses) == B

    if group_idx is None:
        group_idx = ["0"] * B
    if traj_idx is None:
        traj_idx = [0] * B

    batch_prompts = _pad_1d_int(prompts, PAD_TOKEN_ID)
    batch_resps = _pad_1d_int(responses, PAD_TOKEN_ID)

    if response_mask is None:
        response_mask = [[1] * len(responses[i]) for i in range(B)]
    batch_mask = _pad_1d_int(response_mask, 0)

    if rm_scores is None:
        rm_scores = [[0.0] * len(responses[i]) for i in range(B)]
    batch_rm = _pad_1d_float(rm_scores, 0.0)

    batch = {
        "prompts": batch_prompts,
        "responses": batch_resps,
        "response_mask": batch_mask,
        "rm_scores": batch_rm,
    }

    nt: Dict[str, Any] = {
        "group_idx": _as_1d_object_array(list(group_idx)),
        "traj_idx": _as_1d_object_array(list(traj_idx)),
    }
    if turn_idx is not None:
        nt["turn_idx"] = _as_1d_object_array(list(turn_idx))
    if image_data is not None:
        nt["image_data"] = _as_1d_object_array(list(image_data))
    if reward_extra_info is not None:
        nt["reward_extra_info"] = _as_1d_object_array(list(reward_extra_info))

    return DataProto(batch=TensorDict(batch, batch_size=(B,)), non_tensor_batch=nt, meta_info={})


def _make_test_gen_batch_uid_from_output(dp: DataProto) -> DataProto:
    """
    Build a TRAJECTORY-level test_gen_batch with only uid, where uid == group_idx.
    Trajectories are unique (group_idx, traj_idx), sorted by (group, traj).
    """
    nt = dp.non_tensor_batch
    group_arr = nt["group_idx"]
    traj_arr = nt["traj_idx"]

    keys = sorted({(str(group_arr[i]), int(traj_arr[i])) for i in range(len(dp))})
    uid = [g for (g, _t) in keys]

    dummy_batch = TensorDict({}, batch_size=(len(uid),))
    return DataProto(batch=dummy_batch, non_tensor_batch={"uid": _as_1d_object_array(uid)}, meta_info={})


# -----------------------------
# Tests (minimal, cover new rule)
# -----------------------------
def test_two_turn_reward_extra_info_uses_last_turn_and_is_copied_to_top_level():
    dp = _make_dataproto(
        prompts=[[10], [11]],
        responses=[[20], [30]],
        group_idx=["g", "g"],
        traj_idx=[1, 1],
        turn_idx=[0, 1],
        image_data=[["i0"], ["i1"]],
        reward_extra_info=[
            {"traj_success": 0.0, "foo": 1},
            {"traj_success": 1.0, "foo": 2, "bar": "x"},
        ],
    )
    tg = _make_test_gen_batch_uid_from_output(dp)
    out = concat_val_multi_turn(dp, tg)

    assert out.non_tensor_batch["reward_extra_info"][0] == {"traj_success": 1.0, "foo": 2, "bar": "x"}
    assert out.non_tensor_batch["traj_success"][0] == 1.0
    assert out.non_tensor_batch["foo"][0] == 2
    assert out.non_tensor_batch["bar"][0] == "x"


if __name__ == "__main__":
    test_two_turn_reward_extra_info_uses_last_turn_and_is_copied_to_top_level()
    print("All tests passed.")
