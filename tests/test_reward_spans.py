"""Tests for placing a reward on the tokens that earned it.

The whole point is alignment, and misalignment is invisible: the reward still sums to
the same total and the loss stays finite, so it can only be caught by checking positions.
"""

import pytest

from vagen.envs._common.rewards import spread, tagged_span, token_offsets, tokens_covering


class CharTokenizer:
    """One character per token, so offsets are known by inspection."""

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def _ids(text):
    return [ord(c) for c in text]


def test_offsets_are_monotone_prefix_lengths():
    offsets = token_offsets(_ids("abc"), CharTokenizer())

    assert offsets == [1, 2, 3]


def test_a_span_maps_to_the_tokens_that_spell_it():
    text = "xx<perception>hi</perception>yy"
    offsets = token_offsets(_ids(text), CharTokenizer())
    span = tagged_span(text, "perception")

    covering = tokens_covering(span, offsets)

    assert "".join(text[i] for i in covering) == "hi"


def test_a_token_straddling_the_boundary_is_included():
    """Crediting one token too many beats leaving the first word of a description
    unrewarded, which is what dropping it would do."""

    class TwoCharTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(i) for i in ids)

    offsets = [2, 4, 6]          # tokens spanning [0,2), [2,4), [4,6)
    assert tokens_covering((1, 5), offsets) == [0, 1, 2]


def test_an_absent_tag_is_none_not_an_empty_span():
    """None means 'the agent did not describe anything', which earns no reward; an empty
    span would silently credit token zero."""
    assert tagged_span("no tags here", "perception") is None


def test_reward_is_split_across_the_span_not_repeated():
    """★ The return sees the sum, so repeating the value would pay more for a longer
    description -- a length-hacking channel."""
    scores = spread(1.0, [1, 2, 3], length=5)

    assert sum(scores) == pytest.approx(1.0)
    assert scores == pytest.approx([0.0, 1 / 3, 1 / 3, 1 / 3, 0.0])


def test_spreading_over_nothing_is_all_zero():
    assert spread(1.0, [], length=3) == [0.0, 0.0, 0.0]


def test_offsets_come_from_the_emitted_ids_not_a_re_encoding():
    """★ Re-encoding a substring can split it differently from how it was generated, and
    a reward vector built on that is misaligned against the sequence being trained."""
    import inspect

    from vagen.envs._common.rewards import spans

    source = inspect.getsource(spans)
    for encoder in ("encode(", "apply_chat_template", "__call__"):
        assert encoder not in source, f"the span map reaches for {encoder}"
