"""The continuation seam: a turn boundary must match what the template would produce.

Qwen closes every message with ``<|im_end|>\n``. The model stops at ``<|im_end|>``, so
the newline is template output -- and stripping the placeholder turn whole took it away,
leaving every continuation one token short. Rollout and training saw the same seam, so
nothing downstream could tell: off-distribution input, not a train/infer mismatch.

The fix derives the separator from the tokenizer's declared terminator and then *verifies*
it against a real two-turn render, falling back to the old behaviour on any mismatch.
That matters more than the derivation: a family whose template closes a message with
something other than its ``eos_token`` would derive the wrong answer, and splicing wrong
tokens into every turn boundary is far worse than being one token short.
"""

from __future__ import annotations

import pytest

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def _client(tok):
    from vagen.models.qwen import QwenModelAdapter

    return QwenModelAdapter(tok)


def test_an_incrementally_built_conversation_matches_the_canonical_render():
    """What the model is given should be what the chat template would have produced.

    Nothing downstream can tell when this is wrong -- the loss is unaffected and both
    sides of the run agree -- so it has to be asserted directly against the template.
    """
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    c = _client(tok)

    opening, _ = c.render(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"}],
        opening=True,
    )
    response = tok.encode("A1<|im_end|>", add_special_tokens=False)
    continuation, _ = c.render([{"role": "user", "content": "U2"}], opening=False)

    incremental = tok.decode(opening + response + continuation)
    canonical = tok.apply_chat_template(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"},
         {"role": "assistant", "content": "A1"}, {"role": "user", "content": "U2"}],
        add_generation_prompt=True, tokenize=False)

    assert incremental == canonical


def test_a_family_whose_separator_cannot_be_derived_falls_back_rather_than_guessing():
    """The verification is the point, not the derivation.

    Splicing the wrong tokens into every turn boundary is far worse than being one token
    short, so a derivation that does not reproduce the template must yield nothing.
    """
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    c = _client(tok)
    assert c.message_separator(), "the separator should be derivable for Qwen2.5-VL"

    c2 = _client(tok)
    # A terminator the template does not actually use.
    class _Wrong:
        def __getattr__(self, k): return getattr(tok, k)
        eos_token_id = 999999
    c2.tokenizer = _Wrong()
    assert c2.message_separator() == [], (
        "a terminator the template does not use produced a separator anyway"
    )
