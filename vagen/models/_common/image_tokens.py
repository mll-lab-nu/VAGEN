"""Shared model-family image-token handling.

A rendered VLM prompt repeats one placeholder token once per image patch, which for a
single frame can be hundreds of tokens. Dumped verbatim it buries the actual prompt.

The token is read off the processor rather than matched against a per-family regex: a
table of patterns has to be extended for every new model, and when it is not the miss is
silent -- the log simply looks wrong. Asking the processor works for any VLM that
declares its placeholder, which is the normal case.
"""

from __future__ import annotations

import re
import warnings
from typing import Optional, Union

class ImagePlaceholderMismatch(ValueError):
    """Frames and placeholder runs disagree.

    Raised rather than rendered around, because the same two things build
    ``multi_modal_inputs`` for the forward pass. A count that is wrong here is wrong
    there, and there it is silent -- the model attends to a picture it was never given,
    or to none, and the loss stays finite.
    """


# Attributes a processor may expose its image placeholder under, most specific first.
_TOKEN_ATTRS = ("image_token", "image_placeholder_token", "boi_token")


def get_image_token(processor) -> Optional[str]:
    """The placeholder string this processor repeats once per image patch."""
    if processor is None:
        return None
    for attr in _TOKEN_ATTRS:
        token = getattr(processor, attr, None)
        if isinstance(token, str) and token:
            return token
    # Some processors carry only the id, with the tokenizer holding the string.
    token_id = getattr(processor, "image_token_id", None)
    tokenizer = getattr(processor, "tokenizer", None)
    if token_id is not None and tokenizer is not None:
        try:
            return tokenizer.convert_ids_to_tokens(int(token_id))
        except Exception:  # noqa: BLE001 - a processor that cannot answer is not an error
            return None
    return None


def replace_image_tokens_for_logging(
    texts: Union[str, list[str]],
    processor=None,
    replacement: str = "<image>",
) -> Union[str, list[str]]:
    """Replace each run of the image placeholder with a single readable marker.

    Returns the text unchanged when the processor declares no placeholder: the log is
    then merely long, which must not be worth failing a training step over.
    """
    single = isinstance(texts, str)
    items = [texts] if single else list(texts)

    token = get_image_token(processor)
    if token is None:
        if processor is not None:
            warnings.warn(
                f"{type(processor).__name__} declares no image token; logged prompts keep the raw "
                "placeholder run",
                stacklevel=2,
            )
        return texts

    # Surrounding markers such as Qwen's <|vision_start|>/<|vision_end|> are left alone:
    # they are one token each, and dropping them would misrepresent what the model saw.
    pattern = re.compile(f"(?:{re.escape(token)})+")
    replaced = [pattern.sub(replacement, item) for item in items]
    return replaced[0] if single else replaced


#: Families that declare their image placeholder nowhere we can read it. Keyed on the
#: class name of whatever is passed in; the value returns that family's placeholder ids.
#:
#: Empty, and that is the point: every family checked so far declares them, and an entry
#: here should be a deliberate statement that we support a family which does not. The
#: alternative -- matching a list of likely spellings -- is a table that grows with every
#: model and whose misses are silent.
IMAGE_TOKEN_ADAPTERS: dict = {}


def register_image_tokens(*tokenizer_classes: str):
    """Declare how a family spells its image placeholder."""

    def wrap(fn):
        for name in tokenizer_classes:
            IMAGE_TOKEN_ADAPTERS[name] = fn
        return fn

    return wrap



def image_token_ids(source) -> set:
    """Ids that stand in for image content. A run of them is where a picture sits.

    Read off whatever declares them. For Qwen2.5-VL that is the *processor* --
    ``image_token_id`` / ``video_token_id`` -- and its tokenizer declares nothing at all,
    which is why asking the tokenizer produced an empty set and sent this looking for
    spellings to match. A list of spellings is a table that has to grow with every model
    and whose misses are silent, so there is none here.

    ``IMAGE_TOKEN_ADAPTERS`` is the extension point for a family that declares its
    placeholder nowhere. An entry there is a deliberate statement that we support that
    family and know how it marks an image.
    """
    ids = set()
    for holder in (source, getattr(source, "processor", None), getattr(source, "tokenizer", None)):
        if holder is None:
            continue
        for attr in ("image_token_id", "video_token_id"):
            value = getattr(holder, attr, None)
            if isinstance(value, int):
                ids.add(value)
    if ids:
        return ids

    adapter = IMAGE_TOKEN_ADAPTERS.get(type(source).__name__)
    if adapter is not None:
        try:
            return adapter(source) or set()
        except Exception as exc:  # noqa: BLE001 - a broken adapter must not stop a run
            warnings.warn(f"image-token adapter for {type(source).__name__} failed: {exc}")
    return ids


class NoValidTruncation(ValueError):
    # Not EpisodeUnusable: reaching it means a single row could not be made trainable,
    # which is about the row, and _outputs is where that is decided.
    """The budget cannot hold anything worth training on.

    A truncation always exists -- drop every picture -- but a 400-token budget on a
    sequence whose first image alone is 300 leaves four tokens, which is a well-formed
    batch row and a worthless one. Better to drop the row and count it.
    """


def vision_sentinel_ids(source) -> set:
    """The ids that bracket a picture, if the model declares them.

    Not decoration. ``get_rope_index`` counts images by reading the token *after* every
    ``vision_start`` -- ``vision_tokens = input_ids[vision_start_indices + 1]``, in
    transformers and in verl's own copy -- so a run that has lost its opening sentinel is
    not counted as an image at all. It is laid out as text, the grid entry meant for it is
    consumed by a later run, and every position after it shifts. Nothing raises: the
    placeholder and feature counts still agree, so ``masked_scatter`` runs happily.

    That makes ``vision_start .. vision_end`` the atomic unit for cutting, not the run.

    Declared on the config for Qwen2.5-VL, which verl attaches to the processor. Read the
    same way as the placeholders themselves: no spelling table.
    """
    ids = set()
    for holder in (source, getattr(source, "config", None), getattr(source, "processor", None),
                   getattr(getattr(source, "processor", None), "config", None)):
        if holder is None:
            continue
        for attr in ("vision_start_token_id", "vision_end_token_id"):
            value = getattr(holder, attr, None)
            if isinstance(value, int):
                ids.add(value)
    return ids


def placeholder_blocks(token_ids, placeholders: set, sentinels: set = frozenset()) -> list:
    """``(start, end)`` of each picture, sentinels included.

    ``count_placeholder_runs`` answers "how many"; cutting needs "where", and it needs the
    boundary to be the *block* rather than the run -- see ``vision_sentinel_ids``.
    """
    ids = [int(t) for t in token_ids]
    blocks, i, n = [], 0, len(ids)
    while i < n:
        if ids[i] in placeholders:
            start, end = i, i
            while end < n and ids[end] in placeholders:
                end += 1
            # Swallow the sentinels that bracket this run, when the model has them.
            if sentinels and start > 0 and ids[start - 1] in sentinels:
                start -= 1
            if sentinels and end < n and ids[end] in sentinels:
                end += 1
            blocks.append((start, end))
            i = end
        else:
            i += 1
    return blocks


def truncate_keeping_images_whole(token_ids, budget: int, *, keep: str, placeholders: set,
                                  frames=None, sentinels: set = frozenset(), min_kept: int = 1):
    """Cut to ``budget``, never through a picture. Returns ``(token_ids, frames)``.

    A cut that lands inside a picture takes the whole picture, and the frame that goes
    with it -- placeholder blocks and ``multi_modal_inputs`` are strictly 1:1, and
    ``multi_modal_inputs`` is built from the frames list *alone* (the token sequence is
    decoded with ``skip_special_tokens=True`` first, which erases every placeholder). So
    a frames list that disagrees with the blocks is not caught by anything downstream; it
    is handed to the model as a picture it was never shown.

    ``keep="head"`` cuts the end, ``keep="tail"`` drops the beginning.
    """
    ids = list(token_ids)
    frames = list(frames or [])
    if len(ids) <= budget:
        return ids, frames

    blocks = placeholder_blocks(ids, placeholders, sentinels)
    if blocks and len(blocks) != len(frames):
        raise ImagePlaceholderMismatch(
            f"{len(blocks)} placeholder block(s) but {len(frames)} frame(s) before "
            f"truncating; the two have to agree or the cut cannot keep them in step"
        )

    if keep == "head":
        cut = budget
        for start, end in blocks:
            if start < cut < end:
                cut = start          # the cut fell inside this picture, so it goes too
                break
        kept_ids = ids[:cut]
        kept_frames = [f for (s, e), f in zip(blocks, frames) if e <= cut]
    else:
        cut = len(ids) - budget
        for start, end in blocks:
            if start < cut < end:
                cut = end
                break
        kept_ids = ids[cut:]
        kept_frames = [f for (s, e), f in zip(blocks, frames) if s >= cut]

    if len(kept_ids) < min_kept:
        raise NoValidTruncation(
            f"cutting to {budget} leaves {len(kept_ids)} token(s), under the {min_kept} "
            f"worth keeping. A picture that does not fit takes the whole sequence with it."
        )
    return kept_ids, kept_frames


def count_placeholder_runs(token_ids, placeholders: set) -> int:
    """How many frames this span has room for: one per *run* of placeholder ids.

    A caller slicing a conversation into spans needs this to hand each span its own
    frames. Given the whole list instead, every span but the last looks like it has
    frames left over, and ``split_on_images`` refuses -- correctly, by its own rule, for
    a mismatch the caller invented.
    """
    runs, inside = 0, False
    for tid in token_ids:
        if int(tid) in placeholders:
            if not inside:
                runs += 1
            inside = True
        else:
            inside = False
    return runs


def split_on_images(token_ids, placeholders: set, tokenizer, frames) -> list[dict]:
    """A token span as alternating text and image parts, in sequence order.

    The placeholder run *is* the picture's position, so the frame replaces it rather than
    being appended near it. Decoding with ``skip_special_tokens`` erases those tokens
    entirely, which is why the position had to be guessed before -- and guessed wrong,
    since a frame then lands after the marker that starts the model's own reply.

    Frames are consumed in order, and the counts must agree exactly. A mismatch is not a
    display problem: the same frames and the same placeholders build ``multi_modal_inputs``
    for the forward pass, so if they disagree here they disagree there -- which is how a
    model ends up attending to a picture it was not given, or to none at all. It raises.
    """
    if frames and not placeholders:
        raise ImagePlaceholderMismatch(
            f"{len(frames)} frame(s) to place but no image placeholder is declared for "
            f"this model. Add an entry to IMAGE_TOKEN_ADAPTERS saying how this family "
            f"marks an image."
        )
    parts: list[dict] = []
    pending = list(frames or [])
    buffer: list[int] = []

    def flush():
        if buffer:
            text = tokenizer.decode(buffer, skip_special_tokens=True)
            if text:
                parts.append({"text": text})
            buffer.clear()

    i, n = 0, len(token_ids)
    while i < n:
        if placeholders and int(token_ids[i]) in placeholders:
            flush()
            while i < n and int(token_ids[i]) in placeholders:
                i += 1
            if not pending:
                raise ImagePlaceholderMismatch(
                    f"a placeholder run at token {i} has no frame to put in it; "
                    f"{len(frames or [])} frame(s) were captured for this span"
                )
            parts.append({"image": pending.pop(0)})
            continue
        buffer.append(int(token_ids[i]))
        i += 1
    flush()
    if pending:
        raise ImagePlaceholderMismatch(
            f"{len(pending)} frame(s) left over with no placeholder run to occupy; "
            f"the span has {len(frames or []) - len(pending)} run(s)"
        )
    return parts
