"""Tests for `concat_val_multi_turn` (no-concat validation turn->trajectory concat).

Focus: the padding-strip fix. Each per-turn tensor is individually padded
(response RIGHT-padded to response_length, prompts LEFT-padded to prompt_length).
Concatenating them as-is buries pad tokens *inside* the sequence, which then breaks
the trailing-pad assumption of the `s[:l]` decode in `_validate` -> later turns are
truncated from the logged text while `image_data` keeps every turn's image
("N images but only M<N visible turns"). The fix strips each segment before
concatenating so the result is contiguous (pad only at the very end).

Token id conventions used by these tests:
  - response real tokens live in [100, 199]  -> loss mask must be 1
  - prompt   real tokens live in [800, 999]  -> loss mask must be 0
  - a prompt's FIRST real token is a unique per-turn marker in [900, 949]
  - PAD = 0 (must never appear inside real content)

Run directly:  python -m vagen.tests.test_concat_val_multi_turn
Or with pytest: pytest vagen/tests/test_concat_val_multi_turn.py
"""
import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto

from vagen.utils.concat_val_multi_turn import (
    concat_val_multi_turn,
    _real_len_right,
    _real_start_left,
    _as_1d_object_array,
)

PAD = 0
PROMPT_W = 24   # stand-in for rollout.prompt_length
RESP_W = 16     # stand-in for rollout.response_length


class FakeTok:
    pad_token_id = PAD


# ----------------------------- builders -----------------------------
def _left_pad(seq, width=PROMPT_W):
    seq = list(seq)
    assert len(seq) <= width, (len(seq), width)
    return [PAD] * (width - len(seq)) + seq


def _right_pad(seq, width=RESP_W, pad=PAD):
    seq = list(seq)
    assert len(seq) <= width, (len(seq), width)
    return seq + [pad] * (width - len(seq))


def _turn_row(group, traj, turn, n_resp=2, n_prompt=3, success=0.0):
    """One per-turn output row. Prompt starts with a unique marker 900+turn."""
    prompt_real = [900 + turn] + [800 + turn] * (n_prompt - 1)   # all in prompt range
    resp_real = [100 + turn] * n_resp                            # all in response range
    return {
        "group": group, "traj": traj, "turn": turn,
        "prompt": prompt_real, "resp": resp_real,
        "img": [f"g{group}_t{turn}_img"],                        # exactly one image per turn
        "rei": {"traj_success": float(success)},
    }


def _make_output(rows):
    prompts, responses, masks, rms = [], [], [], []
    groups, trajs, turns, imgs, reis = [], [], [], [], []
    for r in rows:
        prompts.append(_left_pad(r["prompt"]))
        responses.append(_right_pad(r["resp"]))
        masks.append(_right_pad([1] * len(r["resp"])))
        rms.append(_right_pad([1.0] * len(r["resp"]), pad=0.0))
        groups.append(r["group"]); trajs.append(r["traj"]); turns.append(r["turn"])
        imgs.append(r["img"]); reis.append(r["rei"])
    batch = TensorDict(
        {
            "prompts": torch.tensor(prompts, dtype=torch.long),
            "responses": torch.tensor(responses, dtype=torch.long),
            "response_mask": torch.tensor(masks, dtype=torch.long),
            "rm_scores": torch.tensor(rms, dtype=torch.float32),
        },
        batch_size=(len(rows),),
    )
    nt = {
        "group_idx": _as_1d_object_array(groups),
        "traj_idx": _as_1d_object_array(trajs),
        "turn_idx": _as_1d_object_array(turns),
        "image_data": _as_1d_object_array(imgs),
        "reward_extra_info": _as_1d_object_array(reis),
    }
    return DataProto(batch=batch, non_tensor_batch=nt, meta_info={})


def _gen_batch(uids):
    return DataProto(
        batch=TensorDict({}, batch_size=(len(uids),)),
        non_tensor_batch={"uid": _as_1d_object_array(list(uids))},
        meta_info={},
    )


# -------------------- decode identical to _validate --------------------
def _decode_real_response(resp_row):
    """Mirror ray_trainer._validate: outputs -> s[:l], l = #non-pad tokens."""
    l = int((resp_row != PAD).sum().item())
    return resp_row[:l]


def _decode_real_prompt(prompt_row):
    """Mirror _validate inputs -> s[-l:], l = #non-pad tokens (left-padded)."""
    l = int((prompt_row != PAD).sum().item())
    return prompt_row[-l:] if l else prompt_row[:0]


def _count_turn_markers(t):
    return int(((t >= 900) & (t <= 949)).sum().item())


def _row_for_uid(out, uid):
    idx = [i for i, g in enumerate(out.non_tensor_batch["group_idx"]) if str(g) == str(uid)]
    assert len(idx) == 1, (uid, idx)
    return idx[0]


def _assert_trajectory_ok(out, uid, expected_turns):
    """The central invariant: after the fix, a trajectory's decoded text shows
    exactly `expected_turns` turns and that equals len(image_data)."""
    i = _row_for_uid(out, uid)
    resp = out.batch["responses"][i]
    real = _decode_real_response(resp)

    # (1) contiguous: the s[:l] slice contains NO interior pad -> nothing truncated
    assert bool((real != PAD).all()), f"uid={uid}: interior pad in response -> s[:l] would truncate"

    # (2) image_data count == expected turns
    imgs = out.non_tensor_batch["image_data"][i]
    assert len(imgs) == expected_turns, f"uid={uid}: image_data={len(imgs)} != turns={expected_turns}"

    # (3) THE user-reported property: #turns visible in text == #images
    prompt_real = _decode_real_prompt(out.batch["prompts"][i])
    markers = _count_turn_markers(prompt_real) + _count_turn_markers(real)
    assert markers == expected_turns, f"uid={uid}: visible turns={markers} != {expected_turns}"
    assert markers == len(imgs), f"uid={uid}: visible turns={markers} != images={len(imgs)}"

    # (4) response_mask aligns with the stripped content:
    #     1 for response tokens (100-199), 0 for inserted prompt tokens (800-999)
    mask = out.batch["response_mask"][i][: real.shape[0]]
    is_resp_tok = (real >= 100) & (real <= 199)
    is_prompt_tok = (real >= 800) & (real <= 999)
    assert bool((mask[is_resp_tok] == 1).all()), f"uid={uid}: response tokens not masked 1"
    assert bool((mask[is_prompt_tok] == 0).all()), f"uid={uid}: inserted prompt tokens not masked 0"
    return real


# ------------------------------ tests ------------------------------
def test_single_turn_consistent():
    rows = [_turn_row("A", 0, 0)]
    out = concat_val_multi_turn(_make_output(rows), _gen_batch(["A"]), FakeTok())
    _assert_trajectory_ok(out, "A", expected_turns=1)


def test_two_turn_contiguous_and_matches_images():
    rows = [_turn_row("A", 0, 0), _turn_row("A", 0, 1)]
    out = concat_val_multi_turn(_make_output(rows), _gen_batch(["A"]), FakeTok())
    real = _assert_trajectory_ok(out, "A", expected_turns=2)
    # exact contiguous layout: resp0 + prompt1 + resp1  (all real, in order)
    expected = [100, 100, 901, 801, 801, 101, 101]
    assert real.tolist() == expected, real.tolist()


def test_five_turn_no_truncation_regression():
    """The exact bug: a 5-turn trajectory must show 5 turns / 5 images (not 2/5)."""
    rows = [_turn_row("A", 0, t) for t in range(5)]
    out = concat_val_multi_turn(_make_output(rows), _gen_batch(["A"]), FakeTok())
    _assert_trajectory_ok(out, "A", expected_turns=5)


def test_varying_segment_lengths():
    rows = [
        _turn_row("A", 0, 0, n_resp=1, n_prompt=1),
        _turn_row("A", 0, 1, n_resp=3, n_prompt=5),
        _turn_row("A", 0, 2, n_resp=2, n_prompt=2),
    ]
    out = concat_val_multi_turn(_make_output(rows), _gen_batch(["A"]), FakeTok())
    _assert_trajectory_ok(out, "A", expected_turns=3)


def test_turn_order_is_sorted_by_turn_idx():
    # feed rows out of order; concat must sort by turn_idx
    rows = [_turn_row("A", 0, 2), _turn_row("A", 0, 0), _turn_row("A", 0, 1)]
    out = concat_val_multi_turn(_make_output(rows), _gen_batch(["A"]), FakeTok())
    real = _assert_trajectory_ok(out, "A", expected_turns=3)
    # markers must appear in ascending turn order: 901 then 902 in the response
    marker_seq = [int(x) for x in real.tolist() if 900 <= x <= 949]
    assert marker_seq == [901, 902], marker_seq


def test_multi_trajectory_batch_mixed_lengths():
    rows = []
    rows += [_turn_row("A", 0, 0)]                       # 1 turn
    rows += [_turn_row("B", 0, t) for t in range(2)]     # 2 turns
    rows += [_turn_row("C", 0, t) for t in range(5)]     # 5 turns
    out = concat_val_multi_turn(_make_output(rows), _gen_batch(["A", "B", "C"]), FakeTok())
    assert len(out) == 3
    _assert_trajectory_ok(out, "A", 1)
    _assert_trajectory_ok(out, "B", 2)
    _assert_trajectory_ok(out, "C", 5)
    # final batch padding must be trailing-only for every row
    for i in range(len(out)):
        resp = out.batch["responses"][i]
        l = int((resp != PAD).sum().item())
        assert bool((resp[:l] != PAD).all()) and bool((resp[l:] == PAD).all())


def test_reorder_follows_gen_uid():
    rows = [_turn_row("B", 0, 0), _turn_row("A", 0, 0), _turn_row("A", 0, 1)]
    out = concat_val_multi_turn(_make_output(rows), _gen_batch(["A", "B"]), FakeTok())
    assert [str(x) for x in out.non_tensor_batch["group_idx"]] == ["A", "B"]


def test_rm_scores_sum_preserved():
    rows = [_turn_row("A", 0, t, n_resp=2) for t in range(4)]
    out = concat_val_multi_turn(_make_output(rows), _gen_batch(["A"]), FakeTok())
    i = _row_for_uid(out, "A")
    # 4 turns * 2 response tokens * 1.0 each = 8.0; inserted prompts contribute 0
    assert abs(float(out.batch["rm_scores"][i].sum().item()) - 8.0) < 1e-6


def test_last_turn_reward_extra_info_used():
    rows = [_turn_row("A", 0, 0, success=0.0), _turn_row("A", 0, 1, success=1.0)]
    out = concat_val_multi_turn(_make_output(rows), _gen_batch(["A"]), FakeTok())
    i = _row_for_uid(out, "A")
    assert out.non_tensor_batch["reward_extra_info"][i]["traj_success"] == 1.0
    assert out.non_tensor_batch["traj_success"][i] == 1.0   # copied to top level


def test_stress_random_lengths_image_equals_turns():
    rng = np.random.RandomState(0)
    rows, uids = [], []
    for g in range(12):
        uid = f"U{g}"
        uids.append(uid)
        n_turns = int(rng.randint(1, 6))
        for t in range(n_turns):
            rows.append(_turn_row(uid, 0, t,
                                  n_resp=int(rng.randint(1, 4)),
                                  n_prompt=int(rng.randint(1, 6))))
    out = concat_val_multi_turn(_make_output(rows), _gen_batch(uids), FakeTok())
    # recompute expected turns per uid
    from collections import Counter
    exp = Counter(r["group"] for r in rows)
    for uid in uids:
        _assert_trajectory_ok(out, uid, exp[uid])


def test_helpers_are_robust_to_interior_pad():
    # right-padded, with an interior pad-valued token: keep up to last real token
    t = torch.tensor([5, 0, 7, 0, 0])
    assert _real_len_right(t, PAD) == 3          # keeps [5,0,7]
    # left-padded, interior pad: keep from first real token
    t = torch.tensor([0, 0, 5, 0, 7])
    assert _real_start_left(t, PAD) == 2         # keeps [5,0,7]
    # all pad
    assert _real_len_right(torch.tensor([0, 0]), PAD) == 0
    assert _real_start_left(torch.tensor([0, 0]), PAD) == 2


def test_empty_input_returns_empty():
    empty = _make_output([_turn_row("A", 0, 0)])[:0]
    out = concat_val_multi_turn(empty, _gen_batch([]), FakeTok())
    assert len(out) == 0


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    passed = 0
    for fn in ALL:
        fn()
        print(f"[ok] {fn.__name__}")
        passed += 1
    print(f"\nALL {passed} TESTS PASSED")
