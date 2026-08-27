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

import inspect
import pytest

from vagen.models import (
    ImagePlaceholderMismatch,
    NoValidTruncation,
    placeholder_blocks,
    truncate_keeping_images_whole,
    vision_sentinel_ids,
)

from model_path import local_snapshot

class _NoCompaction:
    """A harness that never summarised. `_outputs` asks it which conversations ended at a
    compaction seam rather than because the environment stepped; only CompactHarness ever
    answers non-empty."""

    summarised_conversations: set = set()



MODEL = local_snapshot()

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

    from vagen.models import image_token_ids

    placeholders = {i for i in image_token_ids(processor)}
    sentinels = vision_sentinel_ids(processor)
    assert sentinels, "the sentinels must be declared, or the cut cannot be safe"
    vstart = getattr(processor.config, "vision_start_token_id", None)
    assert vstart is not None, "no vision_start declared, so rope's count cannot be checked"

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
            # transformers>=5.3 requires mm_token_type_ids; the real processor emits it,
            # so mark the image-pad positions the way the training path does.
            rope_kwargs = {}
            if "mm_token_type_ids" in inspect.signature(processor.get_rope_index).parameters:
                mm = torch.zeros_like(t)
                pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                mm[t == pad_id] = 1
                rope_kwargs["mm_token_type_ids"] = mm
            processor.get_rope_index(input_ids=t, image_grid_thw=grid,
                                     attention_mask=torch.ones_like(t), **rope_kwargs)
        except Exception as exc:  # noqa: BLE001 - the failure is the finding
            pytest.fail(f"{keep} budget={budget}: position ids rejected the cut: "
                        f"{type(exc).__name__}: {exc}")

        # And the silent half. Rope counts an image by reading the token after each
        # vision_start, so a run that lost its sentinel is counted as text and raises
        # nothing at all -- asserting "no exception" would pass straight over the failure
        # this whole file exists for. Measured on the naive rule: 0 of 40 tail budgets
        # raise, 3 of 40 are silently wrong.
        # Only vision_START opens an image; vision_end closes one, and counting from it
        # reads whatever text follows. rope itself indexes `vision_start_indices + 1`.
        counted = sum(1 for i, tok in enumerate(kept)
                      if tok == vstart and i + 1 < len(kept) and kept[i + 1] in placeholders)
        assert counted == len(kept_frames), (
            f"{keep} budget={budget}: rope counts {counted} image(s) but {len(kept_frames)} "
            f"frame(s) were kept -- an orphaned run is being laid out as text"
        )


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


# ------------------------------------------------- that the loop actually uses all this
def test_the_batch_boundary_cuts_with_the_image_aware_helper():
    """The helper being correct is worth nothing if `_outputs` slices around it.

    Replacing the call in `_truncate_response` with a plain `ids[:budget]` that keeps
    every frame broke no test before this one: the truncation tests all exercise the
    helper directly.
    """
    from omegaconf import OmegaConf

    from vagen.training.agent_loop.gym_loop import GymLoop

    #  model turn | observation with a picture | model turn | observation with a picture
    ids = [1, 2] + [VS, PAD, PAD, PAD, VE] + [3] * 10 + [VS, PAD, PAD, PAD, VE] + [4]
    frames = ["first", "second"]
    # Spans cover the model's tokens only -- the pictures arrive in observations, which
    # sit between them. A span covering a placeholder would mean the policy emitted one.
    spans = [(0, 2), (7, 17)]

    class _Row:
        ordinal = 0
        conversation_id = "c"
        prompt_ids = [1]
        response_ids = ids
        response_mask = [1] * len(ids)
        logprobs = [0.0] * len(ids)
        scores = [0.0] * len(ids)
        response_spans = spans

    class _Client:
        def rows(self): return [_Row()]
        def images(self, cid): return list(frames)

    class _Env:
        success = False
        state_scores = {}

    class _Result:
        turns = 1

    loop = GymLoop.__new__(GymLoop)
    loop.prompt_length, loop.response_length = 100, 16      # cuts inside the second image
    loop.processor = loop.tokenizer = None
    loop._ph_cache = (PH, SENT)
    loop.config = OmegaConf.create({"trainer": {"harness": "concat", "compact_budget": 400}})

    out = GymLoop._outputs(loop, _Client(), _Env(), _Result(),
                           {"group_idx": "g", "traj_idx": 0}, "ep", _NoCompaction())[0]
    kept = out.response_ids
    kept_frames = out.multi_modal_data.get("images", [])
    blocks = placeholder_blocks(kept, PH, SENT)
    assert len(blocks) == len(kept_frames), (
        f"{len(blocks)} block(s) survived but {len(kept_frames)} frame(s) were published"
    )
    assert not kept or kept[-1] != VS, "a dangling vision_start reached the batch"
    assert kept_frames == ["first"], f"the wrong frame survived: {kept_frames}"


def test_a_vision_token_the_policy_invented_is_refused():
    """Nothing bans the vision vocabulary from a generation -- they are ordinary ids.

    Sampled, there is no frame behind them, the placeholder count exceeds the frame count
    and the row dies inside get_rope_index with a bare `IndexError: index 2 is out of
    bounds`, several layers from anything naming a cause.
    """
    from omegaconf import OmegaConf

    from vagen.training.agent_loop.gym_loop import GymLoop, SampledVisionToken
    from vagen.rollout.client import EpisodeUnusable

    assert issubclass(SampledVisionToken, EpisodeUnusable), (
        "the policy's output is not a configuration error; it should cost one rollout"
    )

    class _Row:
        ordinal = 0
        conversation_id = "c"
        prompt_ids = [1, 2]
        response_ids = [5, VS, PAD, 6]          # the model emitted a picture opener
        response_mask = [1, 1, 1, 1]
        logprobs = [0.0] * 4
        scores = [0.0] * 4
        response_spans = [(0, 4)]

    class _Client:
        def rows(self): return [_Row()]
        def images(self, cid): return []

    class _Env:
        success = False
        state_scores = {}

    class _Result:
        turns = 1

    loop = GymLoop.__new__(GymLoop)
    loop.prompt_length, loop.response_length = 100, 100
    loop.processor = loop.tokenizer = None
    loop._ph_cache = (PH, SENT)
    loop.config = OmegaConf.create({"trainer": {"harness": "concat", "compact_budget": 400}})

    with pytest.raises(SampledVisionToken, match="generated vision token"):
        GymLoop._outputs(loop, _Client(), _Env(), _Result(),
                         {"group_idx": "g", "traj_idx": 0}, "ep", _NoCompaction())


# ------------------------------- cutting an observation must not cut the turn boundary
#
# ★ The regression this file exists to pin. The obvious way to enforce
# `max_env_response_per_turn` is to render the span and slice the token list -- and it is
# wrong, because `render` tokenizes with add_generation_prompt=True, so the span ends
# `<|im_end|>\n<|im_start|>assistant\n` and a head-keeping slice throws that away FIRST.
# The engine then gets a prompt that stops mid-observation with no role boundary; the model
# continues the user's sentence, `add_response` records it at mask 1, and `accept()` hands
# it to env.step as an action. Nothing raises. One warning per client covers a whole run.
#
# So the cut is on the message text, before rendering: the template rebuilds the boundary
# either way, and a trimmed observation is still a well-formed turn.


def test_a_trimmed_continuation_still_ends_with_the_generation_prompt():
    """★ Driven through `send`, not through `_fit_messages` directly.

    The bug was an ORDERING inside `InferenceClient.send`: it encoded first and cut the
    rendered span, which takes the trailing generation prompt with it. A test that calls
    `_fit_messages` and then re-renders proves nothing -- re-rendering puts the prompt back
    by construction, and the earlier version of this test passed against the buggy `send`.
    So this asserts on the tokens the *conversation* ends up holding.
    """
    import asyncio

    from transformers import AutoTokenizer

    from vagen.rollout.client import BackendOutput, InferenceClient
    from model_path import local_snapshot

    tok = AutoTokenizer.from_pretrained(local_snapshot("Qwen/Qwen2.5-VL-3B-Instruct"))

    class _C(InferenceClient):
        tokenizer = tok

        def __init__(self):
            super().__init__()
            self.ran = None

        def encode(self, messages):
            return tok.apply_chat_template(
                [{"role": m["role"], "content": m["content"]} for m in messages],
                add_generation_prompt=True, tokenize=True, return_dict=False)

        async def generate(self, prompt_ids, **kw):
            self.ran = list(prompt_ids)          # what the engine would have been given
            return BackendOutput(text="x", token_ids=[1])

    c = _C()
    c.opening_limit, c.continuation_limit = 10_000, 60
    # open the conversation, then continue it with an oversized observation
    cid = asyncio.run(c.send([{"role": "user", "content": "start"}])).conversation_id
    asyncio.run(c.send([{"role": "user", "content": "the board is unchanged. " * 200}], cid))

    tail = tok.decode(c.ran[-8:])
    assert "assistant" in tail, (
        f"the generation prompt was cut off the prompt the engine ran; tail={tail!r}. "
        f"A continuation ending mid-observation with no role boundary makes the model "
        f"continue the user's sentence, and that gets recorded at mask 1 as an action.")
    assert "board" in tok.decode(c.ran), "the observation was dropped entirely"


def test_the_cut_happens_before_rendering_not_after():
    """The same property, stated as the invariant rather than the symptom: whatever `send`
    stores for a conversation must be something `encode` could have produced."""
    import asyncio

    from vagen.rollout.client import BackendOutput, InferenceClient

    SENTINEL = 999_999          # stands in for the generation prompt

    class _C(InferenceClient):
        tokenizer = object()

        def encode(self, messages):
            n = sum(len(m["content"]) for m in messages)
            return [1] * n + [SENTINEL]          # every render ends with the boundary

        async def generate(self, prompt_ids, **kw):
            return BackendOutput(text="x", token_ids=[2])

    c = _C()
    c.opening_limit, c.continuation_limit = 10_000, 50
    cid = asyncio.run(c.send([{"role": "user", "content": "hi"}])).conversation_id
    asyncio.run(c.send([{"role": "user", "content": "o" * 400}], cid))

    stored = c._conversations[cid].token_ids
    assert stored[-1] == SENTINEL or SENTINEL in stored[-3:], (
        "the stored conversation does not end with the boundary encode always emits, so "
        "it was cut after rendering")
