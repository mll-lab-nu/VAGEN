"""Collapse image-placeholder runs so logged text stays readable.

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


def split_on_images(token_ids, placeholders: set, tokenizer, frames) -> list[dict]:
    """A token span as alternating text and image parts, in sequence order.

    The placeholder run *is* the picture's position, so the frame replaces it rather than
    being appended near it. Decoding with ``skip_special_tokens`` erases those tokens
    entirely, which is why the position had to be guessed before -- and guessed wrong,
    since a frame then lands after the marker that starts the model's own reply.

    Frames are consumed in order. A run with no frame left, and a frame with no run to
    replace, both become visible markers: a mismatch here means the transcript no longer
    shows what the model was given, which is worse than an ugly line saying so.
    """
    if frames and not placeholders:
        # No adapter for this family. Say so where it will be read, once per span, rather
        # than appending the pictures somewhere plausible and looking correct.
        return [
            {"text": f"[no image placeholder declared for this model; "
                     f"see IMAGE_TOKEN_ADAPTERS. {len(frames)} frame(s) shown out of place]"},
            *({"image": f} for f in frames),
            {"text": tokenizer.decode(list(token_ids), skip_special_tokens=True)},
        ]
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
            parts.append({"image": pending.pop(0)} if pending else {"text": "[image, no frame captured]"})
            continue
        buffer.append(int(token_ids[i]))
        i += 1
    flush()
    if pending:
        # More frames than placeholder runs: show them, and say they are out of place.
        parts.append({"text": f"[{len(pending)} frame(s) with no placeholder in this span]"})
        parts.extend({"image": f} for f in pending)
    return parts
