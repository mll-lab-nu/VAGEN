"""The continuation seam: every turn boundary is one separator token short.

Known defect, pinned rather than fixed. See logs/template-seam.md for why the fix is not
a one-liner and what the options are.
"""

from __future__ import annotations

import pytest

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def _client(tok):
    from vagen.agent_loop.verl_client import VerlClient

    c = VerlClient.__new__(VerlClient)
    c.tokenizer, c.processor = tok, None
    c.apply_chat_template_kwargs, c.mm_processor_kwargs = {}, {}
    c._images, c._active, c._prefix_cache, c._conversations = {}, None, None, {}
    return c


@pytest.mark.xfail(strict=True, reason=(
    "the continuation strip removes the separator that belongs to the preceding "
    "assistant turn -- see logs/template-seam.md. Remove this marker when fixed."
))
def test_an_incrementally_built_conversation_matches_the_canonical_render():
    """What the model is given should be what the chat template would have produced.

    It is one token short at every turn boundary. Rollout and training see the same
    sequence, so this is off-distribution input rather than a train/infer mismatch, which
    is why it has survived: nothing downstream can tell, and the loss is unaffected.
    """
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    c = _client(tok)

    opening = c.encode([{"role": "system", "content": "S"}, {"role": "user", "content": "U1"}])

    class _Conv:
        prompt_len = 1                       # so the next encode reads as a continuation

    c._conversations["x"] = _Conv()
    c._active = "x"
    response = tok.encode("A1<|im_end|>", add_special_tokens=False)
    continuation = c.encode([{"role": "user", "content": "U2"}])

    incremental = tok.decode(opening + response + continuation)
    canonical = tok.apply_chat_template(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"},
         {"role": "assistant", "content": "A1"}, {"role": "user", "content": "U2"}],
        add_generation_prompt=True, tokenize=False)

    assert incremental == canonical
