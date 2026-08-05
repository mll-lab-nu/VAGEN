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
