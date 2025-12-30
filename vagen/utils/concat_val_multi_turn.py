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
    Concatenate multi-turn trajectories, then STRICTLY reorder to match test_gen_batch.uid,
    where uid is equivalent to group_idx.

    STRICT CONSTRAINTS:
      - After turn->trajectory concatenation, number of trajectories MUST equal len(test_gen_batch).
      - The multiset of test_gen_batch.uid MUST equal the multiset of concatenated group_idx.
      - Reorder consumes, for each uid, the next trajectory in ascending traj_idx order.
      - No placeholder. Any mismatch => assert.
    """
    n = len(test_output_gen_batch)
    if n == 0:
        return test_output_gen_batch[:0]

    # test_gen_batch must provide uid
    assert "uid" in test_gen_batch.non_tensor_batch, (
        "concat_val_multi_turn: test_gen_batch.non_tensor_batch must contain key 'uid'"
    )

    nt = test_output_gen_batch.non_tensor_batch
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

    for k in trajectory_groups:
        trajectory_groups[k].sort(key=lambda x: x[0])  # by turn_idx

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
        traj_success_values: List[float] = []

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

            if "traj_success" in nt:
                v = nt["traj_success"][i]
                if v is not None:
                    traj_success_values.append(float(v))

            if j < len(turns) - 1:
                next_i = turns[j + 1][1]
                next_prompt = test_output_gen_batch.batch["prompts"][next_i]

                resp_parts.append(next_prompt)
                mask_parts.append(torch.zeros_like(next_prompt))
                rm_parts.append(torch.zeros_like(next_prompt, dtype=rm.dtype, device=rm.device))

        concat_response = torch.cat(resp_parts, dim=0)
        concat_response_mask = torch.cat(mask_parts, dim=0)
        concat_rm_scores = torch.cat(rm_parts, dim=0)

        merged_images: List[Any] = []
        if "images" in nt:
            img_arr = nt["images"]
            for _, i in turns:
                v = img_arr[i]
                if v is None:
                    continue
                if isinstance(v, (list, tuple, np.ndarray)):
                    merged_images.extend(list(v))
                else:
                    merged_images.append(v)

        traj_success = max(traj_success_values) if traj_success_values else None

        batch_entry = {
            "prompts": concat_prompt,
            "responses": concat_response,
            "response_mask": concat_response_mask,
            "rm_scores": concat_rm_scores,
        }
        non_tensor_entry = {
            "group_idx": group_idx_str,
            "traj_idx": int(traj_idx),
            "images": merged_images,
            "reward_extra_info": {"traj_success": traj_success},
        }
        concatenated.append((batch_entry, non_tensor_entry))

    if not concatenated:
        return test_output_gen_batch[:0]

    # ------------------------------------------------------------------
    # 2.5) STRICT alignment with test_gen_batch.uid (uid == group_idx)
    # ------------------------------------------------------------------
    gen_uid = [str(x) for x in test_gen_batch.non_tensor_batch["uid"]]
    target_n = len(test_gen_batch)

    # Build bucket: uid(group_idx) -> list of trajectories sorted by traj_idx asc
    bucket: Dict[str, List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]] = defaultdict(list)
    for be, nte in concatenated:
        bucket[str(nte["group_idx"])].append((be, nte))
    for uid in bucket:
        bucket[uid].sort(key=lambda x: int(x[1]["traj_idx"]))

    # --- HARD CHECK 1: trajectory count must match ---
    num_traj = sum(len(v) for v in bucket.values())
    assert num_traj == target_n, (
        "concat_val_multi_turn: trajectory-level count mismatch.\n"
        f"  num_traj_after_concat={num_traj}\n"
        f"  len(test_gen_batch)={target_n}\n"
        "Hint: test_gen_batch.uid must be TRAJECTORY-level, not TURN-level."
    )

    # --- HARD CHECK 2: uid multiset must match group_idx multiset ---
    expected_uid_counter = Counter(bucket.keys())
    # Counter(bucket.keys()) is wrong if same uid has >1 traj; need counts:
    expected_uid_counter = Counter({uid: len(v) for uid, v in bucket.items()})
    actual_uid_counter = Counter(gen_uid)
    assert actual_uid_counter == expected_uid_counter, (
        "concat_val_multi_turn: uid multiset mismatch (uid == group_idx).\n"
        f"  expected(from concat)={dict(expected_uid_counter)}\n"
        f"  actual(test_gen_batch)={dict(actual_uid_counter)}\n"
        "Hint: test_gen_batch.uid must repeat per-trajectory under the same uid."
    )

    # Reorder exactly following gen_uid; consume within uid by traj_idx asc
    reordered: List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]] = []
    for i, uid in enumerate(gen_uid):
        assert uid in bucket and len(bucket[uid]) > 0, (
            f"concat_val_multi_turn: cannot find trajectory for uid={uid} (index={i})."
        )
        reordered.append(bucket[uid].pop(0))

    # Must exhaust buckets
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

    stacked_non_tensor = {
        "group_idx": _as_1d_object_array([nte["group_idx"] for _, nte in concatenated]),
        "traj_idx": _as_1d_object_array([nte["traj_idx"] for _, nte in concatenated]),
        "images": _as_1d_object_array([nte["images"] for _, nte in concatenated]),
        "reward_extra_info": _as_1d_object_array([nte["reward_extra_info"] for _, nte in concatenated]),
    }

    out = DataProto(
        batch=TensorDict(stacked_batch, batch_size=(len(concatenated),)),
        non_tensor_batch=stacked_non_tensor,
        meta_info=getattr(test_output_gen_batch, "meta_info", {}),
    )

    # final alignment check: out.group_idx[i] == test_gen_batch.uid[i]
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
    traj_success: Optional[List[Any]] = None,
    images: Optional[List[Any]] = None,
) -> DataProto:
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
    if traj_success is not None:
        nt["traj_success"] = _as_1d_object_array(list(traj_success))
    if images is not None:
        nt["images"] = _as_1d_object_array(list(images))

    return DataProto(batch=TensorDict(batch, batch_size=(B,)), non_tensor_batch=nt, meta_info={})


def _make_test_gen_batch_uid_from_output(dp: DataProto) -> DataProto:
    """
    Build a TRAJECTORY-level test_gen_batch with only uid, where uid == group_idx.
    We derive trajectories by unique (group_idx, traj_idx), sorted by (group, traj).
    This matches the strict requirement in concat_val_multi_turn.
    """
    nt = dp.non_tensor_batch
    group_arr = nt["group_idx"]
    traj_arr = nt["traj_idx"]

    keys = sorted({(str(group_arr[i]), int(traj_arr[i])) for i in range(len(dp))})
    uid = [g for (g, _t) in keys]  # uid == group_idx, per trajectory

    # test_gen_batch only needs non_tensor_batch["uid"], batch content unused in concat
    dummy_batch = TensorDict({}, batch_size=(len(uid),))
    return DataProto(batch=dummy_batch, non_tensor_batch={"uid": _as_1d_object_array(uid)}, meta_info={})


# -----------------------------
# Tests
# -----------------------------
def test_single_turn_keeps_as_is_and_traj_success_max():
    dp = _make_dataproto(
        prompts=[[11, 12, 13]],
        responses=[[21, 22]],
        response_mask=[[1, 1]],
        rm_scores=[[0.3, 0.7]],
        group_idx=["g"],
        traj_idx=[5],
        turn_idx=[0],
        traj_success=[0.2],
        images=[["imgA"]],
    )
    tg = _make_test_gen_batch_uid_from_output(dp)
    out = concat_val_multi_turn(dp, tg)

    assert len(out) == len(tg) == 1
    assert out.non_tensor_batch["group_idx"][0] == "g"
    assert int(out.non_tensor_batch["traj_idx"][0]) == 5
    assert out.batch["responses"][0].tolist() == [21, 22]
    assert out.batch["response_mask"][0].tolist() == [1, 1]
    assert torch.allclose(out.batch["rm_scores"][0], torch.tensor([0.3, 0.7]))
    assert out.non_tensor_batch["images"][0] == ["imgA"]
    assert out.non_tensor_batch["reward_extra_info"][0] == {"traj_success": 0.2}


def test_two_turn_concat_inserts_next_prompt_as_is_and_mask_rm_zeros_for_prompt_segment():
    dp = _make_dataproto(
        prompts=[[10, 10], [11, 11, 11]],
        responses=[[20], [30, 31]],
        response_mask=[[1], [1, 0]],
        rm_scores=[[0.2], [0.9, 0.1]],
        group_idx=["g", "g"],
        traj_idx=[1, 1],
        turn_idx=[0, 1],
        traj_success=[0.0, 1.0],
        images=[["i0"], ["i1", "i2"]],
    )
    tg = _make_test_gen_batch_uid_from_output(dp)  # trajectory-level uid => ["g"]
    out = concat_val_multi_turn(dp, tg)

    assert len(out) == 1
    assert out.batch["responses"][0].tolist() == [20, 0, 11, 11, 11, 30, 31]
    assert out.batch["response_mask"][0].tolist() == [1, 0, 0, 0, 0, 1, 0]
    assert torch.allclose(
        out.batch["rm_scores"][0],
        torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1]),
    )
    assert list(out.non_tensor_batch["images"][0]) == ["i0", "i1", "i2"]
    assert out.non_tensor_batch["reward_extra_info"][0] == {"traj_success": 1.0}


def test_sort_by_turn_idx_and_traj_success_max_across_turns():
    dp = _make_dataproto(
        prompts=[[111], [100, 0, 0]],
        responses=[[300], [200]],
        response_mask=[[1], [1]],
        rm_scores=[[3.0], [2.0]],
        group_idx=["g", "g"],
        traj_idx=[7, 7],
        turn_idx=[1, 0],
        traj_success=[0.4, 0.9],
        images=[["b"], ["a"]],
    )
    tg = _make_test_gen_batch_uid_from_output(dp)  # ["g"]
    out = concat_val_multi_turn(dp, tg)

    assert len(out) == 1
    assert out.non_tensor_batch["images"][0] == ["a", "b"]
    assert out.non_tensor_batch["reward_extra_info"][0] == {"traj_success": 0.9}


def test_multiple_trajectories_padding_in_output_stack():
    dp = _make_dataproto(
        prompts=[[1, 1], [2], [9, 9, 9]],
        responses=[[10], [20, 21], [90]],
        response_mask=[[1], [1, 1], [1]],
        rm_scores=[[0.1], [0.2, 0.3], [9.0]],
        group_idx=["A", "A", "B"],
        traj_idx=[0, 0, 5],
        turn_idx=[0, 1, 0],
        traj_success=[0.0, 1.0, 1.0],
        images=[["a0"], ["a1"], []],
    )
    tg = _make_test_gen_batch_uid_from_output(dp)  # trajectory-level uid => ["A", "B"]
    out = concat_val_multi_turn(dp, tg)

    assert len(out) == 2
    L = out.batch["responses"].shape[1]
    assert out.batch["response_mask"].shape[1] == L
    assert out.batch["rm_scores"].shape[1] == L

    assert tuple(out.non_tensor_batch["group_idx"]) == ("A", "B")
    assert out.non_tensor_batch["reward_extra_info"][0] == {"traj_success": 1.0}
    assert out.non_tensor_batch["reward_extra_info"][1] == {"traj_success": 1.0}


def test_missing_images_and_traj_success_is_none():
    dp = _make_dataproto(
        prompts=[[1], [2]],
        responses=[[10], [20]],
        group_idx=["g", "g"],
        traj_idx=[1, 1],
        turn_idx=[0, 1],
    )
    tg = _make_test_gen_batch_uid_from_output(dp)  # ["g"]
    out = concat_val_multi_turn(dp, tg)

    assert len(out) == 1
    assert out.non_tensor_batch["images"][0] == []
    assert out.non_tensor_batch["reward_extra_info"][0] == {"traj_success": None}


if __name__ == "__main__":
    test_single_turn_keeps_as_is_and_traj_success_max()
    test_two_turn_concat_inserts_next_prompt_as_is_and_mask_rm_zeros_for_prompt_segment()
    test_sort_by_turn_idx_and_traj_success_max_across_turns()
    test_multiple_trajectories_padding_in_output_stack()
    test_missing_images_and_traj_success_is_none()
    print("All tests passed.")
