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
