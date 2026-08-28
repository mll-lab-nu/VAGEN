"""Locate a rewarded response span among the tokens the model produced.

A reward computed on ``<observation>...</observation>`` should land on the tokens that
carry it, not on the end of the turn. That needs a map from character offsets to token
positions.

★ Built by decoding prefixes of the ids the model actually emitted, never by re-encoding
the text. BPE is not compositional, so re-encoding a substring can split it differently
from how it was generated, and a reward vector built on that would be misaligned against
the sequence being trained -- silently, since both are well-formed.
"""

from __future__ import annotations

import re


def token_offsets(token_ids: list[int], tokenizer) -> list[int]:
    """Character offset at which each token ends, decoding prefixes of what was emitted.

    ★ Decoded the same way the text being searched was decoded -- ``skip_special_tokens``
    on. The spans come from matching tags in the action text, which the client produced
    with special tokens skipped; measuring offsets with them rendered shifts every
    position after the first special token by that token's printed length. One
    ``<|box_start|>`` before the description was enough to move a reward off
    ``box left of player`` and onto ``<observation>box left``.

    Monotone by construction. O(n) decodes; at a few hundred response tokens that is
    negligible next to the rollout itself.
    """
    offsets, text = [], ""
    for k in range(1, len(token_ids) + 1):
        text = tokenizer.decode(token_ids[:k], skip_special_tokens=True)
        offsets.append(len(text))
    return offsets


def tokens_covering(span: tuple[int, int], offsets: list[int]) -> list[int]:
    """Indices of the tokens overlapping a character span.

    A token counts if any of its characters fall inside the span, so a token straddling
    the boundary is included rather than dropped -- crediting one token too many is
    preferable to leaving the first or last word of a description unrewarded.
    """
    start, end = span
    covering, previous_end = [], 0
    for i, offset in enumerate(offsets):
        if previous_end < end and offset > start:
            covering.append(i)
        previous_end = offset
    return covering


def tagged_span(text: str, tag: str) -> tuple[int, int] | None:
    """Character span of the content inside ``<tag>...</tag>``, if present."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.span(1) if match else None


def spread(value: float, indices: list[int], length: int) -> list[float]:
    """A reward vector giving ``value`` to the named tokens, split evenly.

    Split rather than repeated: the sum is what the return sees, so repeating it would
    make a longer description worth more, which is a length-hacking channel.
    """
    scores = [0.0] * length
    if not indices:
        return scores
    share = value / len(indices)
    for i in indices:
        scores[i] = share
    return scores
