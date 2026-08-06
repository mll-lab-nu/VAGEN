"""A concat episode is one conversation whose turns must still be recoverable.

concat puts a whole episode in ONE batch row: [resp0][obs1][resp1][obs2][resp2]. The
turns are only distinguishable by the spans the tape recorded, and what sits *between*
two spans is the observation the environment sent back.

Getting this wrong is invisible: the transcript renders as a stack of responses with no
observations at all, which reads as a model talking to itself.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vagen.utils.concat_val_multi_turn import concat_val_multi_turn  # noqa: E402


class _Tok:
    """Decodes each id to a word, so the split is legible in assertions."""

    pad_token_id = 0
    WORDS = {1: "RESP0", 2: "OBS1", 3: "RESP1", 4: "OBS2", 5: "RESP2"}

    def decode(self, ids, **kw):
        return " ".join(self.WORDS.get(int(i), "?") for i in ids if int(i) != 0)


def _concat_batch():
    """One row = one conversation with three turns and two observations between them."""
    from tensordict import TensorDict
    from verl.protocol import DataProto

    # response region: RESP0 OBS1 RESP1 OBS2 RESP2
    resp = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    L = resp.shape[1]
    b = TensorDict(
        {
            "prompts": torch.ones(1, 4, dtype=torch.long),
            "responses": resp,
            "response_mask": torch.tensor([[1, 0, 1, 0, 1]], dtype=torch.long),
            "rm_scores": torch.zeros(1, L),
        },
        batch_size=(1,),
    )

    def arr(v):
        a = np.empty(1, dtype=object)
        a[:] = [v]
        return a

    nt = {
        "uid": arr("ep"), "group_idx": arr("ep"), "traj_idx": arr(0), "turn_idx": arr(0),
        "image_data": arr(["f0", "f1", "f2"]),
        "reward_extra_info": arr({"traj_success": 0.0}),
        "episode_id": arr("EP"), "conversation_id": arr(0),
        # the three model outputs inside this one conversation
        "response_spans": arr([(0, 1), (2, 3), (4, 5)]),
    }
    out = DataProto(batch=b, non_tensor_batch=nt, meta_info={})

    pb = TensorDict({"input_ids": torch.ones(1, 4, dtype=torch.long)}, batch_size=(1,))
    pa = np.empty(1, dtype=object)
    pa[:] = ["ep"]
    return out, DataProto(batch=pb, non_tensor_batch={"uid": pa}, meta_info={})


def _conversations():
    merged = concat_val_multi_turn(*_concat_batch(), _Tok())
    return merged.non_tensor_batch["conversations"][0]


def test_one_conversation_with_three_turns():
    convs = _conversations()
    assert len(convs) == 1, "concat is one conversation per episode"
    assert len(convs[0]["turns"]) == 3, f"turns not recovered: {convs[0]['turns']}"


def test_each_turn_holds_only_its_own_response():
    turns = _conversations()[0]["turns"]
    assert [t["response"] for t in turns] == ["RESP0", "RESP1", "RESP2"]


def test_the_observation_between_two_turns_is_recovered():
    """The gap between two response spans is what the environment sent back. Reading it
    from the *next row's* prompt -- which is what walking the batch gives you -- yields
    nothing at all here, because concat has no next row."""
    turns = _conversations()[0]["turns"]
    assert [t["observation"] for t in turns] == ["OBS1", "OBS2", ""], (
        "observations between turns were dropped; the transcript reads as a model "
        "talking to itself"
    )


def test_turn_ids_run_from_zero_within_the_conversation():
    assert [t["turn_id"] for t in _conversations()[0]["turns"]] == [0, 1, 2]


def test_frames_are_spread_over_the_turns_not_stacked_at_the_top():
    conv = _conversations()[0]
    assert conv["prompt_image"] == "f0"
    assert [t["observation_image"] for t in conv["turns"]] == ["f1", "f2", None]


def test_the_training_tensors_are_untouched_by_any_of_this():
    """The concatenated response is what gets optimised; the split is only for reading."""
    merged = concat_val_multi_turn(*_concat_batch(), _Tok())
    assert merged.batch["responses"][0].tolist() == [1, 2, 3, 4, 5]
    assert merged.batch["response_mask"][0].tolist() == [1, 0, 1, 0, 1], (
        "the mask must still be 1 exactly on generated tokens"
    )
