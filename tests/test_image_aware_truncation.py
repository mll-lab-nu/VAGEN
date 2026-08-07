"""Cutting a multimodal sequence without lying to the model about its pictures.

The rule that looks right -- "if the cut lands inside a run of placeholders, drop the
whole run" -- is not enough, and the case it misses is the silent one. ``get_rope_index``
counts images by reading the token *after* every ``vision_start``:

    vision_tokens = input_ids[vision_start_indices + 1]      # modeling_qwen2_5_vl.py

so a run that has lost its opening sentinel is not counted as an image at all. It is laid
out as text, the grid entry meant for it is consumed by a later run, and every position
after it shifts. Nothing raises -- the placeholder and feature counts still agree, so
``masked_scatter`` runs. The atomic unit is therefore ``vision_start .. vision_end``.

These tests push every result through verl's real position-id path, because that is the
only thing that distinguishes the silent failure from a correct cut.
"""

from __future__ import annotations

import pytest

from vagen.utils.image_token_utils import (
    ImagePlaceholderMismatch,
    NoValidTruncation,
    placeholder_blocks,
    truncate_keeping_images_whole,
    vision_sentinel_ids,
)

MODEL = ("$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct"
         "/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3")

PAD, VS, VE = 9, 8, 7          # placeholder, vision_start, vision_end
PH, SENT = {PAD}, {VS, VE}


def _seq(n_images=2):
    """A sequence with `n_images` bracketed pictures and text between them."""
    ids = [1, 2]
    for i in range(n_images):
        ids += [VS, PAD, PAD, PAD, VE, 3 + i]
    return ids


# ------------------------------------------------------------------ the block boundary
def test_a_block_is_the_sentinels_too_not_just_the_run():
    ids = _seq(2)
    runs_only = placeholder_blocks(ids, PH, sentinels=frozenset())
    blocks = placeholder_blocks(ids, PH, SENT)
    assert runs_only == [(3, 6), (9, 12)]
    assert blocks == [(2, 7), (8, 13)], (
        "the sentinels have to be inside the block, or a cut at the run's first token "
        "leaves the sentinel dangling and rope stops counting that image"
    )


@pytest.mark.parametrize("budget", range(1, 15))
def test_a_head_cut_never_leaves_a_dangling_vision_start(budget):
    """The loud half: a sequence ending on vision_start makes get_rope_index index past
    the end of its own input."""
    ids = _seq(2)
    kept, _ = truncate_keeping_images_whole(
        ids, budget, keep="head", placeholders=PH, frames=["a", "b"], sentinels=SENT,
        min_kept=0)
    assert not kept or kept[-1] != VS, f"budget={budget} left a dangling vision_start"


@pytest.mark.parametrize("budget", range(1, 15))
def test_a_tail_cut_never_orphans_a_run_from_its_vision_start(budget):
    """The silent half: the run survives without its sentinel, every count still agrees,
    and rope lays the picture out as text."""
    ids = _seq(2)
    kept, frames = truncate_keeping_images_whole(
        ids, budget, keep="tail", placeholders=PH, frames=["a", "b"], sentinels=SENT,
        min_kept=0)
    runs = placeholder_blocks(kept, PH, sentinels=frozenset())
    for start, _ in runs:
        assert start > 0 and kept[start - 1] == VS, (
            f"budget={budget}: a run at {start} lost its vision_start -- "
            f"rope will count one image fewer and shift every position after it"
        )
    assert len(runs) == len(frames), "frames and surviving runs disagree"


# ------------------------------------------------------------------- frames in lockstep
@pytest.mark.parametrize("keep", ["head", "tail"])
@pytest.mark.parametrize("budget", range(1, 15))
def test_frames_and_blocks_agree_after_every_cut(keep, budget):
    """multi_modal_inputs is built from the frames list alone -- the token sequence is
    decoded with skip_special_tokens=True first, which erases every placeholder. So a
    frames list that disagrees with the blocks is handed to the model unchallenged."""
    ids = _seq(2)
    kept, frames = truncate_keeping_images_whole(
        ids, budget, keep=keep, placeholders=PH, frames=["a", "b"], sentinels=SENT,
        min_kept=0)
    assert len(placeholder_blocks(kept, PH, SENT)) == len(frames)


def test_a_frames_list_that_already_disagrees_is_refused():
    with pytest.raises(ImagePlaceholderMismatch, match="before truncating"):
        truncate_keeping_images_whole(_seq(2), 5, keep="head", placeholders=PH,
                                      frames=["only-one"], sentinels=SENT)


def test_under_budget_is_returned_untouched():
    ids = _seq(2)
    kept, frames = truncate_keeping_images_whole(ids, 999, keep="head", placeholders=PH,
                                                 frames=["a", "b"], sentinels=SENT)
    assert kept == ids and frames == ["a", "b"]


def test_a_budget_too_small_for_anything_is_refused_rather_than_emitting_a_stub():
    """A picture that does not fit takes the sequence with it, and a four-token row is a
    well-formed batch row that trains on nothing."""
    ids = [1] + [VS] + [PAD] * 300 + [VE]
    with pytest.raises(NoValidTruncation):
        truncate_keeping_images_whole(ids, 200, keep="head", placeholders=PH,
                                      frames=["a"], sentinels=SENT, min_kept=50)


# ------------------------------------------------- against the real model's rope index
@pytest.mark.parametrize("keep", ["head", "tail"])
def test_every_cut_survives_the_real_position_id_path(keep):
    """The only check that distinguishes the silent failure from a correct cut.

    Sweeps every budget over a real Qwen2.5-VL sequence and pushes each result through
    the same get_rope_index the training path uses. A naive slice fails most budgets here.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from verl.utils.tokenizer import hf_processor

    processor = hf_processor(MODEL)
    if processor is None or not hasattr(processor, "get_rope_index"):
        pytest.skip("no processor with get_rope_index available")

    from PIL import Image

    from vagen.utils.image_token_utils import image_token_ids

    placeholders = {i for i in image_token_ids(processor)}
    sentinels = vision_sentinel_ids(processor)
    assert sentinels, "the sentinels must be declared, or the cut cannot be safe"

    frames = [Image.new("RGB", (56, 56), (i * 40, 0, 0)) for i in range(3)]
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "a"},
                                      {"type": "image"}, {"type": "text", "text": "b"},
                                      {"type": "image"}, {"type": "text", "text": "c"}]}],
        add_generation_prompt=True, tokenize=False)
    ids = processor(text=[text], images=frames, return_tensors="pt")["input_ids"][0].tolist()

    for budget in range(1, len(ids) + 1):
        try:
            kept, kept_frames = truncate_keeping_images_whole(
                ids, budget, keep=keep, placeholders=placeholders, frames=frames,
                sentinels=sentinels, min_kept=0)
        except NoValidTruncation:
            continue
        if not kept:
            continue
        # Rebuild the vision inputs from the frames that survived, as the training path
        # does, then ask for position ids. A disagreement shows up here and nowhere else.
        grid = None
        if kept_frames:
            sub = processor.image_processor(images=kept_frames, return_tensors="pt")
            grid = sub["image_grid_thw"]
        t = torch.tensor([kept])
        try:
            processor.get_rope_index(input_ids=t, image_grid_thw=grid,
                                     attention_mask=torch.ones_like(t))
        except Exception as exc:  # noqa: BLE001 - the failure is the finding
            pytest.fail(f"{keep} budget={budget}: position ids rejected the cut: "
                        f"{type(exc).__name__}: {exc}")


def test_the_naive_rule_really_is_insufficient():
    """A canary. Without it, every test above could be passing for the wrong reason.

    Same cut, same sequence, sentinels withheld -- which is exactly the rule "drop the
    whole run if the cut lands inside it". A run survives without its vision_start, every
    count still agrees, and rope silently lays that picture out as text.
    """
    ids = _seq(2)
    orphaned = []
    for budget in range(1, len(ids) + 1):
        kept, _ = truncate_keeping_images_whole(
            ids, budget, keep="tail", placeholders=PH, frames=["a", "b"],
            sentinels=frozenset(), min_kept=0)
        for start, _ in placeholder_blocks(kept, PH, sentinels=frozenset()):
            if start == 0 or kept[start - 1] != VS:
                orphaned.append(budget)
                break
    assert orphaned, (
        "the naive rule produced no orphaned run, so the sentinel handling above is "
        "not being tested by anything"
    )
