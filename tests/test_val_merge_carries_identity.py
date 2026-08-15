"""The validation merge folds an episode's turns into one row -- with its identity.

After the merge a row *is* an episode, so everything describing its shape has to come
across: which episode it is, how many turns it ran, how many conversations it spanned.
Dropping them is silent -- the episode log then groups merged rows by the dataset's axis,
which is unique per row, and every episode reads as a single turn.

This has now broken twice for opposite reasons: the keys were absent from the merge, and
then present but listed in the skip-list for the copy that would have carried them.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vagen.utils.concat_val_multi_turn import concat_val_multi_turn  # noqa: E402


class _Tok:
    pad_token_id = 0

    def decode(self, ids, **kw):
        return " ".join(str(int(i)) for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [int(t) for t in text.split()] if text.strip() else []


def _batch(n_turns=3, uid="ep-1"):
    from tensordict import TensorDict
    from verl.protocol import DataProto

    L = 4
    b = TensorDict(
        {
            "prompts": torch.ones(n_turns, L, dtype=torch.long),
            "responses": torch.ones(n_turns, L, dtype=torch.long) * 2,
            "response_mask": torch.ones(n_turns, L, dtype=torch.long),
            "rm_scores": torch.zeros(n_turns, L),
        },
        batch_size=(n_turns,),
    )

    def arr(vals):
        a = np.empty(n_turns, dtype=object)
        a[:] = vals
        return a

    nt = {
        "uid": arr([uid] * n_turns),
        "group_idx": arr([uid] * n_turns),
        "traj_idx": arr([0] * n_turns),
        "turn_idx": arr(list(range(n_turns))),
        "image_data": arr([[] for _ in range(n_turns)]),
        "reward_extra_info": arr([{"traj_success": 1.0} for _ in range(n_turns)]),
        "episode_id": arr(["EP-XYZ"] * n_turns),
        "episode_turns": arr([n_turns] * n_turns),
        "conversation_id": arr([f"c{i}" for i in range(n_turns)]),
        "data_source": arr(["sokoban"] * n_turns),
    }
    out = DataProto(batch=b, non_tensor_batch=nt, meta_info={})

    pb = TensorDict({"input_ids": torch.ones(1, L, dtype=torch.long)}, batch_size=(1,))
    pa = np.empty(1, dtype=object)
    pa[:] = [uid]
    prompts = DataProto(batch=pb, non_tensor_batch={"uid": pa}, meta_info={})
    return out, prompts


def test_the_merge_yields_one_row_per_episode():
    merged = concat_val_multi_turn(*_batch(3), _Tok())
    assert len(merged.batch["prompts"]) == 1


@pytest.mark.parametrize("key", ["episode_id", "episode_turns", "n_conversations"])
def test_identity_survives_the_merge(key):
    merged = concat_val_multi_turn(*_batch(3), _Tok())
    assert key in merged.non_tensor_batch, f"{key} dropped by the merge"


def test_the_values_are_right_not_just_present():
    merged = concat_val_multi_turn(*_batch(3), _Tok())
    nt = merged.non_tensor_batch
    assert nt["episode_id"][0] == "EP-XYZ"
    assert nt["episode_turns"][0] == 3, "turn count lost; every episode reads as one turn"
    assert nt["n_conversations"][0] == 3, "conversation count lost; compaction invisible"


def test_the_merge_does_not_supply_keys_the_input_batch_already_has():
    """_validate unions the merged output with the input batch, and union asserts that a
    key on both sides is the *same object*. Adding data_source here -- which the dataset
    already provides -- failed every validation pass with an assertion naming only the
    key, several frames from anything suggesting the merge."""
    merged = concat_val_multi_turn(*_batch(3), _Tok())
    assert "data_source" not in merged.non_tensor_batch, (
        "the merge re-supplies a key the input batch owns; union will assert"
    )


# ------------------------------------------------- two episodes under one (group, traj)


def _two_episodes_sharing_a_pair(n_turns=2):
    """The padding-duplicate case, reproduced.

    ``_validate`` pads the generation batch to a multiple of the worker count by
    repeating rows from the front. Those copies run as genuinely separate episodes, but
    ``_vagen_assign_indices`` re-tiles ``traj_idx`` positionally, so a copy lands on the
    same ``(group_idx, traj_idx)`` as its original. Only ``episode_id`` tells them apart.
    """
    from tensordict import TensorDict
    from verl.protocol import DataProto

    L, rows = 4, n_turns * 2
    b = TensorDict(
        {
            "prompts": torch.ones(rows, L, dtype=torch.long),
            "responses": torch.ones(rows, L, dtype=torch.long) * 2,
            "response_mask": torch.ones(rows, L, dtype=torch.long),
            "rm_scores": torch.zeros(rows, L),
        },
        batch_size=(rows,),
    )

    def arr(vals):
        a = np.empty(rows, dtype=object)
        a[:] = vals
        return a

    nt = {
        # ★ Same uid, same group_idx, same traj_idx for both episodes -- exactly what the
        # padding path produces. Distinct only in episode_id.
        "uid": arr(["ep-1"] * rows),
        "group_idx": arr(["ep-1"] * rows),
        "traj_idx": arr([0] * rows),
        "turn_idx": arr(list(range(n_turns)) * 2),
        "image_data": arr([[] for _ in range(rows)]),
        "reward_extra_info": arr([{"traj_success": 1.0} for _ in range(rows)]),
        "episode_id": arr(["EP-A"] * n_turns + ["EP-B"] * n_turns),
        "episode_turns": arr([n_turns] * rows),
        "conversation_id": arr([f"c{i}" for i in range(rows)]),
        "data_source": arr(["sokoban"] * rows),
    }
    out = DataProto(batch=b, non_tensor_batch=nt, meta_info={})

    # Two prompts, because two episodes were run.
    pb = TensorDict({"input_ids": torch.ones(2, L, dtype=torch.long)}, batch_size=(2,))
    pa = np.empty(2, dtype=object)
    pa[:] = ["ep-1", "ep-1"]
    prompts = DataProto(batch=pb, non_tensor_batch={"uid": pa}, meta_info={})
    return out, prompts


def test_two_episodes_sharing_a_pair_do_not_merge():
    """★ The bug that took down the compact run at step 299 with
    ``trajectory-level count mismatch: num_traj=126, len(test_gen_batch)=256`` -- half the
    validation set folded into the other half.

    Grouping by ``(group_idx, traj_idx)`` merges these two episodes into one row, which
    both loses an episode and trips the strict count check. Grouping by ``episode_id``
    keeps them apart. The count check is the *only* thing that made this loud; merging is
    otherwise silent, and one episode is simply credited with another's turns.
    """
    merged = concat_val_multi_turn(*_two_episodes_sharing_a_pair(2), _Tok())
    assert len(merged.batch["prompts"]) == 2, "the two episodes were merged into one row"
    assert sorted(merged.non_tensor_batch["episode_id"]) == ["EP-A", "EP-B"]


def test_the_merged_rows_keep_the_pair_they_came_from():
    """Changing the grouping key must not change what the row reports about itself --
    `group_idx` is what the uid bucketing downstream aligns on."""
    merged = concat_val_multi_turn(*_two_episodes_sharing_a_pair(2), _Tok())
    assert list(merged.non_tensor_batch["group_idx"]) == ["ep-1", "ep-1"]
    assert [int(t) for t in merged.non_tensor_batch["traj_idx"]] == [0, 0]
