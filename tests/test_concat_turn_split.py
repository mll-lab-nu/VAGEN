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
    """Decodes each id to a word, so the split is legible in assertions.

    Declares an image placeholder the way a real processor does, so the merge can find
    where a picture sits instead of guessing.
    """

    pad_token_id = 0
    image_token_id = 9
    WORDS = {1: "RESP0", 2: "OBS1", 3: "RESP1", 4: "OBS2", 5: "RESP2"}

    def decode(self, ids, **kw):
        return " ".join(self.WORDS.get(int(i), "?") for i in ids if int(i) not in (0, 9))


def _concat_batch():
    """One row = one conversation with three turns and two observations between them."""
    from tensordict import TensorDict
    from verl.protocol import DataProto

    # response region: RESP0 OBS1 RESP1 OBS2 RESP2
    resp = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    L = resp.shape[1]
    b = TensorDict(
        {
            "prompts": torch.tensor([[1, 9, 9, 1]], dtype=torch.long),   # a placeholder run
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
        "image_data": arr(["f0"]),
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


def _joined(parts):
    """The text of a span, ignoring where its pictures sit."""
    return " ".join(p["text"] for p in parts if "text" in p).strip()


def test_each_turn_holds_only_its_own_response():
    turns = _conversations()[0]["turns"]
    assert [_joined(t["response"]) for t in turns] == ["RESP0", "RESP1", "RESP2"]


def test_the_observation_between_two_turns_is_recovered():
    """The gap between two response spans is what the environment sent back. Reading it
    from the *next row's* prompt -- which is what walking the batch gives you -- yields
    nothing at all here, because concat has no next row."""
    turns = _conversations()[0]["turns"]
    assert [_joined(t["observation"]) for t in turns] == ["OBS1", "OBS2", ""], (
        "observations between turns were dropped; the transcript reads as a model "
        "talking to itself"
    )


def test_turn_ids_run_from_zero_within_the_conversation():
    assert [t["turn_id"] for t in _conversations()[0]["turns"]] == [0, 1, 2]


def test_a_frame_replaces_the_placeholder_run_that_stands_for_it():
    """The picture's position is in the token sequence, not a guess about which turn it
    belongs to. Frame f0 stands where the prompt's placeholder is."""
    conv = _conversations()[0]
    assert [p.get("image") for p in conv["prompt"] if "image" in p] == ["f0"]


def test_the_training_tensors_are_untouched_by_any_of_this():
    """The concatenated response is what gets optimised; the split is only for reading."""
    merged = concat_val_multi_turn(*_concat_batch(), _Tok())
    assert merged.batch["responses"][0].tolist() == [1, 2, 3, 4, 5]
    assert merged.batch["response_mask"][0].tolist() == [1, 0, 1, 0, 1], (
        "the mask must still be 1 exactly on generated tokens"
    )


def test_frames_in_the_response_region_do_not_trip_the_leftover_check():
    """The merge deals each span its own frames, one per placeholder run.

    Handed the whole list instead, every span but the last looked like it had frames
    left over and split_on_images refused -- a mismatch the merge had invented. Because
    concat and compact keep every observation after the first in the *response* region,
    that was every multimodal validation pass in both modes. The single existing case
    missed it by having exactly one frame, and having it in the prompt.
    """
    import numpy as np
    import torch
    from tensordict import TensorDict
    from verl import DataProto

    from vagen.utils.concat_val_multi_turn import concat_val_multi_turn, _as_1d_object_array

    IMG, PAD = 700, 0

    class _Tok:
        pad_token_id = PAD
        image_token_id = IMG

        def decode(self, ids, **kw):
            return " ".join(str(int(i)) for i in ids)

    # One frame in the prompt and one in each of the two observations between responses,
    # which is the layout concat produces.
    prompt = [900, IMG, 901]
    resp = [100, 101, 800, IMG, 801, 102, 103]
    spans = [(0, 2), (5, 7)]

    batch = DataProto(
        batch=TensorDict({
            "prompts": torch.tensor([prompt]),
            "responses": torch.tensor([resp]),
            "input_ids": torch.tensor([prompt + resp]),
            "attention_mask": torch.ones(1, len(prompt) + len(resp), dtype=torch.long),
            "position_ids": torch.arange(len(prompt) + len(resp)).unsqueeze(0),
            "rm_scores": torch.zeros(1, len(resp)),
            "loss_mask": torch.tensor([[1, 1, 0, 0, 0, 1, 1]]),
        }, batch_size=[1]),
        non_tensor_batch={
            "group_idx": np.array(["g"], dtype=object),
            "traj_idx": np.array([0], dtype=object),
            "turn_idx": np.array([0], dtype=object),
            "conversation_id": np.array([0], dtype=object),
            "episode_id": np.array(["e"], dtype=object),
            "response_spans": _as_1d_object_array([spans]),
            "image_data": _as_1d_object_array([["frame-prompt", "frame-observation"]]),
            "reward_extra_info": _as_1d_object_array([{"traj_success": 1.0}]),
        },
        meta_info={},
    )
    gen = DataProto(
        batch=TensorDict({}, batch_size=[1]),
        non_tensor_batch={"group_idx": np.array(["g"], dtype=object),
                          "traj_idx": np.array([0], dtype=object),
                          "uid": np.array(["g"], dtype=object)},
        meta_info={},
    )

    out = concat_val_multi_turn(batch, gen, _Tok())
    conv = out.non_tensor_batch["conversations"][0][0]
    assert [p for p in conv["prompt"] if "image" in p] == [{"image": "frame-prompt"}]
    observation = conv["turns"][0]["observation"]
    assert [p for p in observation if "image" in p] == [{"image": "frame-observation"}], (
        f"the response-region frame did not land in the observation: {observation}"
    )


def test_two_pictures_rendering_as_one_run_are_refused_not_slid():
    """Without vision sentinels, adjacent pictures are one indistinguishable block.

    The frames then slide: this span takes the wrong picture and the last is dropped,
    with nothing raised. Adjacency itself is undetectable, but "fewer runs than frames"
    is, and it is the same condition.
    """
    import numpy as np
    import pytest
    import torch
    from tensordict import TensorDict
    from verl import DataProto

    from vagen.utils.concat_val_multi_turn import concat_val_multi_turn, _as_1d_object_array
    from vagen.models import ImagePlaceholderMismatch

    IMG, PAD = 700, 0

    class _NoSentinels:
        pad_token_id = PAD
        image_token_id = IMG          # declared; sentinels are not

        def decode(self, ids, **kw):
            return " ".join(str(int(i)) for i in ids)

    prompt = [900, IMG, IMG, 901]     # two pictures, one indistinguishable run
    resp = [100, 101]
    batch = DataProto(
        batch=TensorDict({
            "prompts": torch.tensor([prompt]),
            "responses": torch.tensor([resp]),
            "input_ids": torch.tensor([prompt + resp]),
            "attention_mask": torch.ones(1, len(prompt) + len(resp), dtype=torch.long),
            "position_ids": torch.arange(len(prompt) + len(resp)).unsqueeze(0),
            "rm_scores": torch.zeros(1, len(resp)),
            "loss_mask": torch.tensor([[1, 1]]),
        }, batch_size=[1]),
        non_tensor_batch={
            "group_idx": np.array(["g"], dtype=object),
            "traj_idx": np.array([0], dtype=object),
            "turn_idx": np.array([0], dtype=object),
            "conversation_id": np.array([0], dtype=object),
            "episode_id": np.array(["e"], dtype=object),
            "response_spans": _as_1d_object_array([[(0, 2)]]),
            "image_data": _as_1d_object_array([["first", "second"]]),
            "reward_extra_info": _as_1d_object_array([{"traj_success": 1.0}]),
        },
        meta_info={},
    )
    gen = DataProto(
        batch=TensorDict({}, batch_size=[1]),
        non_tensor_batch={"group_idx": np.array(["g"], dtype=object),
                          "traj_idx": np.array([0], dtype=object),
                          "uid": np.array(["g"], dtype=object)},
        meta_info={},
    )
    with pytest.raises(ImagePlaceholderMismatch, match="no vision sentinels"):
        concat_val_multi_turn(batch, gen, _NoSentinels())
