from __future__ import annotations

import logging
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto

from vagen.models import (
    ImagePlaceholderMismatch, count_placeholder_runs, image_token_ids,
    split_on_images, vision_sentinel_ids,
)

PAD_TOKEN_ID = 0


logger = logging.getLogger(__name__)

def _as_1d_object_array(items: List[Any]) -> np.ndarray:
    """
    Force a 1D numpy object array, where each element is a Python object
    (e.g., list/dict), avoiding numpy auto-expanding list-of-lists to 2D.
    """
    arr = np.empty((len(items),), dtype=object)
    for i, v in enumerate(items):
        arr[i] = v
    return arr


def _real_len_right(t: torch.Tensor, pad_id: int) -> int:
    """Length of the real (non-pad) prefix of a RIGHT-padded 1D tensor.

    Returns ``last_non_pad_index + 1`` (0 if all pad). Using the last non-pad
    index rather than a non-pad *count* is robust to a pad-valued token appearing
    inside the real content (e.g. an interior eos == pad_id): only the trailing
    pad run is trimmed.
    """
    nonpad = (t != pad_id).nonzero()
    if nonpad.numel() == 0:
        return 0
    return int(nonpad.max().item()) + 1


def _real_start_left(t: torch.Tensor, pad_id: int) -> int:
    """Start index of the real span of a LEFT-padded 1D tensor.

    Returns the first non-pad index (``len(t)`` if all pad), so ``t[start:]`` is
    the real content. Robust to interior pad-valued tokens (only the leading pad
    run is trimmed).
    """
    nonpad = (t != pad_id).nonzero()
    if nonpad.numel() == 0:
        return t.shape[0]
    return int(nonpad.min().item())


def concat_val_multi_turn(
    test_output_gen_batch: DataProto,
    test_gen_batch: DataProto,
    tokenizer,
    processor=None,
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
    # ★ What identifies a trajectory here: `episode_id` when the loop published one,
    # falling back to `(group_idx, traj_idx)`. Same rule as `TrajectoryView.build`, and
    # for the same reason -- that pair identifies a *rollout*, not an episode.
    #
    # On this path the two provably differ. `_validate` pads the generation batch to a
    # multiple of the worker count by duplicating rows from the front; the duplicates run
    # as genuinely separate episodes, and `_vagen_assign_indices` re-tiles traj_idx
    # positionally so a copy lands on the same pair as its original. Grouping by the pair
    # then merges two distinct episodes into one, which shows up as the strict count check
    # below firing -- 126 trajectories against 256 prompts on the compact run that found
    # this, i.e. half the validation set silently folded into the other half. It had
    # already been reported on val_navigation (n_envs 30 and 60) and val_spatial_gym (50).
    #
    # `group_idx`/`traj_idx` are still carried on every merged entry; only the grouping
    # key changes, so the uid bucketing below is unaffected.

    # ------------------------------------------------------------------
    # 1) Group turns by (group_idx, traj_idx)
    # ------------------------------------------------------------------
    episode_arr = nt.get("episode_id")
    if episode_arr is None:
        logger.warning(
            "concat_val_multi_turn: no `episode_id` column; grouping by "
            "(group_idx, traj_idx) instead. That pair identifies a rollout, and on the "
            "validation path padding duplicates make one rollout hold several episodes -- "
            "they will be merged into a single row."
        )

    trajectory_groups: Dict[Tuple[str, int], List[Tuple[int, int]]] = defaultdict(list)
    # Keeps the (group_idx, traj_idx) each key came from, so the merged entry can still
    # report them even when the key is an episode_id.
    key_identity: Dict[Tuple[str, int], Tuple[str, int]] = {}
    for i in range(n):
        g = str(group_arr[i])
        t = int(traj_arr[i])
        ti = int(turn_arr[i])
        key = (str(episode_arr[i]), 0) if episode_arr is not None else (g, t)
        trajectory_groups[key].append((ti, i))
        key_identity[key] = (g, t)

    # Sort turns inside each trajectory by turn_idx
    for k in trajectory_groups:
        trajectory_groups[k].sort(key=lambda x: x[0])

    concatenated: List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]] = []

    # ------------------------------------------------------------------
    # 2) Concatenate per trajectory
    # ------------------------------------------------------------------
    for key, turns in sorted(trajectory_groups.items()):
        group_idx_str, traj_idx = key_identity[key]
        first_i = turns[0][1]
        concat_prompt = test_output_gen_batch.batch["prompts"][first_i]

        resp_parts: List[torch.Tensor] = []
        mask_parts: List[torch.Tensor] = []
        rm_parts: List[torch.Tensor] = []


        pad_id = tokenizer.pad_token_id
        # One batch row is one conversation. Rebuild each as it was actually spoken:
        #
        #   conversation 0   system + first observation, then response / observation ...
        #   conversation 1   system + observation carrying the summary, then ...
        #
        # A conversation's own prompt belongs at its *start*. Attaching it to the
        # previous turn as an "observation" -- which is where the next row's prompt
        # naturally falls when you walk the batch -- put each new system prompt after
        # the response before it, so the boundary marker landed a whole turn early.
        conversations: List[Dict[str, Any]] = []
        # Where the pictures actually sit in the sequence, so they can replace their
        # placeholders instead of being appended near them.
        source = processor if processor is not None else tokenizer
        placeholders = image_token_ids(source)
        sentinels = vision_sentinel_ids(source)

        for j, (_, i) in enumerate(turns):
            resp = test_output_gen_batch.batch["responses"][i]

            if "response_mask" in test_output_gen_batch.batch:
                mask = test_output_gen_batch.batch["response_mask"][i]
            else:
                mask = torch.ones_like(resp)

            rm = test_output_gen_batch.batch["rm_scores"][i]
            # Length from the mask when there is one, not from token values. The pad id
            # is a stop token on this model family -- Qwen2.5-VL lists <|endoftext|>,
            # which is also pad, among its eos ids -- so a turn that stops on it has a
            # real final token indistinguishable by value from padding. Trimming by
            # value drops that token, and with it the turn's reward: verl writes the
            # whole row's score at the last real position, computed from the attention
            # mask, so the score lands exactly on the token the value-based trim cuts.
            r_len = _real_len_right(resp, pad_id)
            if "attention_mask" in test_output_gen_batch.batch:
                plen = test_output_gen_batch.batch["prompts"].shape[1]
                r_len = int(test_output_gen_batch.batch["attention_mask"][i, plen:].sum().item())
            elif "response_mask" in test_output_gen_batch.batch:
                nz = (mask != 0).nonzero()
                if nz.numel():
                    r_len = max(r_len, int(nz.max().item()) + 1)

            this_prompt = test_output_gen_batch.batch["prompts"][i]
            p_start = _real_start_left(this_prompt, pad_id)

            # A turn's prompt comes BEFORE the response it elicited. Emitting the
            # response first and the prompt after produces the same tokens in the wrong
            # order -- r0 r1 p1 r2 p2 instead of r0 p1 r1 p2 r2 -- which nothing
            # downstream shape-checks, because it is a permutation of the same length.
            # Every log-prob on such a row is then conditioned on context the model
            # never saw in that position.
            if j:
                prompt_seg = this_prompt[p_start:]
                resp_parts.append(prompt_seg)
                mask_parts.append(torch.zeros_like(prompt_seg))
                rm_parts.append(torch.zeros(prompt_seg.shape[0], dtype=rm.dtype, device=rm.device))

            resp_parts.append(resp[:r_len])
            mask_parts.append(mask[:r_len])
            rm_parts.append(rm[:r_len])

            frames = []
            if "image_data" in nt and nt["image_data"][i] is not None:
                v = nt["image_data"][i]
                frames = list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v]

            # Split the conversation into its turns using the spans the tape recorded.
            # What sits between two responses is the observation that came back.
            spans = nt["response_spans"][i] if nt.get("response_spans") is not None else None
            spans = [(int(x), int(y)) for x, y in spans] if spans else [(0, r_len)]
            spans = [(x, y) for x, y in spans if 0 <= x < y <= r_len] or [(0, r_len)]

            # Frames are consumed in sequence order across the whole conversation: the
            # prompt's placeholders first, then each observation's.
            remaining = list(frames)
            # Every frame has to be accounted for by a run somewhere in this conversation.
            # Fewer runs than frames means two pictures rendered as one indistinguishable
            # placeholder block -- which happens on a family that declares no vision
            # sentinels -- and the frames then slide: a span takes the wrong picture and
            # the last is dropped, with nothing raised. This is detectable; adjacency
            # itself is not.
            total_runs = (count_placeholder_runs(this_prompt[p_start:], placeholders)
                          + count_placeholder_runs(resp[:r_len], placeholders))
            if remaining and total_runs < len(remaining):
                raise ImagePlaceholderMismatch(
                    f"{total_runs} placeholder run(s) for {len(remaining)} frame(s) in one "
                    f"conversation"
                    + ("" if sentinels else
                       " -- this model declares no vision sentinels, so adjacent pictures "
                       "render as a single run and cannot be told apart. Register the "
                       "family in IMAGE_TOKEN_ADAPTERS.")
                )

            def take(span_ids):
                # Hand this span only the frames it has room for. Passing the whole
                # remaining list made every span look like it had frames left over, and
                # the leftover check -- which exists to catch a real placeholder/feature
                # mismatch -- fired on the first span of any conversation whose pictures
                # were not all in its prompt. That is every multimodal concat and compact
                # episode, since their later frames sit in the response region.
                span_ids = list(span_ids)
                runs = count_placeholder_runs(span_ids, placeholders)
                mine, del_ = remaining[:runs], remaining[runs:]
                remaining[:] = del_
                return split_on_images(span_ids, placeholders, tokenizer, mine)

            prompt_parts = take(this_prompt[p_start:])

            turn_list = []
            for k, (start, end) in enumerate(spans):
                nxt = spans[k + 1][0] if k + 1 < len(spans) else r_len
                turn_list.append({
                    "turn_id": k,
                    "response": take(resp[start:end]),
                    "observation": take(resp[end:nxt]) if nxt > end else [],
                })

            conversations.append({
                "conversation_id": j,
                "prompt": prompt_parts,
                "turns": turn_list,
            })

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

        # The identity chain, carried across the merge. Validation folds an episode's
        # per-turn rows into one, so afterwards a row *is* an episode -- but only if the
        # ids come with it. Dropping them left the episode log grouping merged rows by
        # the dataset's axis, which is unique per row, so every episode read as one turn.
        def _first(key, default=None):
            arr = nt.get(key)
            if arr is None:
                return default
            for _, i in turns:
                if arr[i] is not None:
                    return arr[i]
            return default

        non_tensor_entry: Dict[str, Any] = {
            "group_idx": group_idx_str,
            "traj_idx": int(traj_idx),
            "image_data": merged_images,
            "reward_extra_info": reward_extra_info,
            "episode_id": _first("episode_id"),
            # The episode as it was spoken, conversation by conversation. The
            # concatenated response above is what gets trained on; this is what gets read.
            "conversations": conversations,
            # Turns the episode ran. Prefer what the loop counted; fall back to the rows
            # merged here, which is the same number under no_concat.
            # The runner's count, which excludes summaries. Counting response spans
            # instead adds one per compaction: a summary is a model output like any
            # other, but the environment never acted on it.
            "episode_turns": _first(
                "episode_turns", sum(len(c["turns"]) for c in conversations) or len(turns)
            ),
            # How many conversations the episode spanned: 1 for concat, one per turn for
            # no_concat, and however many compactions happened plus one for compact.
            "n_conversations": len(conversations),
            # data_source is deliberately absent: the input batch already carries it, and
            # _validate unions the two. union asserts that a key present on both sides is
            # the same object, so supplying it here fails the whole validation pass.
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
            padding = torch.full((pad,), tokenizer.pad_token_id, dtype=t.dtype, device=t.device)
        return torch.cat([t, padding], dim=0)

    keys = ["prompts", "responses", "response_mask", "rm_scores"]
    stacked_batch: Dict[str, torch.Tensor] = {}
    for k in keys:
        vals = [be[k] for be, _ in concatenated]
        max_len = max(int(v.shape[0]) for v in vals)
        vals = [_pad_1d(v, max_len, k) for v in vals]
        stacked_batch[k] = torch.stack(vals, dim=0)

    # Keys the stacking below writes explicitly. This is a *skip* list for the generic
    # copy that follows, so adding a key here without also stacking it drops the column
    # -- which is how the identity chain went missing twice.
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

    class _Tok:
        pad_token_id = PAD_TOKEN_ID

    out = concat_val_multi_turn(dp, tg, _Tok())

    assert out.non_tensor_batch["reward_extra_info"][0] == {"traj_success": 1.0, "foo": 2, "bar": "x"}
    assert out.non_tensor_batch["traj_success"][0] == 1.0
    assert out.non_tensor_batch["foo"][0] == 2
    assert out.non_tensor_batch["bar"][0] == "x"


if __name__ == "__main__":
    test_two_turn_reward_extra_info_uses_last_turn_and_is_copied_to_top_level()
    print("All tests passed.")
