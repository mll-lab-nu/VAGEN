"""A continuation span must contain the observation and nothing else.

A chat template given a message list with no system message of its own injects a default
one. Rendering the delta alone and then dropping a placeholder turn's worth of tokens
removes the wrong number: the leftover tail of that injected block ends up spliced into
the training sequence ahead of every observation after the first, at mask 0 -- context
the model conditions on that no one wrote.

Uses the real Qwen2.5-VL tokenizer, because the bug is entirely a property of the
template and a stub can be made to agree with any implementation.
"""

from __future__ import annotations

import glob

import pytest

MODEL = glob.glob(
    "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/*/"
)
pytestmark = pytest.mark.skipif(not MODEL, reason="Qwen2.5-VL snapshot not present")


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(MODEL[0], trust_remote_code=True)


class _Started:
    """A conversation that has already had a model output, so the next span is a delta."""

    prompt_len = 7


def _client(tok, *, mid_conversation=True):
    from vagen.agent_loop.verl_client import VerlClient

    c = VerlClient.__new__(VerlClient)
    c.tokenizer, c.processor = tok, None
    c.apply_chat_template_kwargs, c.mm_processor_kwargs = {}, {}
    c._prefix_cache = None
    c._images = {}
    if mid_conversation:
        c._active, c._conversations = "c1", {"c1": _Started()}
    else:
        c._active, c._conversations = None, {}
    return c


def test_the_template_really_does_inject_a_system_block(tok):
    """The premise. If this stops being true the strip is unnecessary, not wrong."""
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "X"}], add_generation_prompt=True, tokenize=False
    )
    assert "You are a helpful assistant." in rendered


def test_a_continuation_span_carries_the_separator_the_model_does_not_generate(tok):
    """It leads with the newline, and that is the fix rather than a leak.

    Qwen closes a message with `<|im_end|>\n`. The model stops at `<|im_end|>`, so the
    newline is template output that belongs to the *preceding* assistant turn -- and
    stripping the placeholder whole took it away, leaving every turn boundary one token
    short of what the template would have produced.
    """
    c = _client(tok)
    ids = c.encode([{"role": "user", "content": "OBSERVATION-ONE"}])
    text = tok.decode(ids)
    sep = tok.decode(c._message_separator())
    assert text.startswith(sep + "<|im_start|>user"), f"span starts with: {text[:60]!r}"
    assert "helpful assistant" not in text, (
        "the injected system block leaked into the continuation span"
    )
    assert "OBSERVATION-ONE" in text


def test_nothing_is_eaten_off_the_front_of_the_observation(tok):
    """Over-stripping is the other direction of the same bug."""
    c = _client(tok)
    text = tok.decode(c.encode([{"role": "user", "content": "DO-NOT-TRUNCATE-ME"}]))
    assert "DO-NOT-TRUNCATE-ME" in text


def test_the_opening_span_is_left_alone(tok):
    """The first call is the whole prompt; it has its own system message and no strip."""
    c = _client(tok, mid_conversation=False)
    ids = c.encode([{"role": "system", "content": "REAL-SYSTEM"},
                    {"role": "user", "content": "OBS-ZERO"}])
    text = tok.decode(ids)
    assert "REAL-SYSTEM" in text and "OBS-ZERO" in text
    assert "helpful assistant" not in text


def test_the_placeholder_used_to_strip_is_the_one_prepended():
    """Two copies of this turn is two chances for the strip to be the wrong length."""
    import inspect

    import pathlib

    # Scoped to the package, not one module: the duplicate this guards against lived in
    # a *different* file, so scanning only verl_client could never have seen it.
    root = pathlib.Path(verl_client_module().__file__).parent
    hits = sum(
        p.read_text().count('"content": "placeholder"')
        for p in root.glob("*.py")
    )
    assert hits == 1, f"the placeholder turn is written out {hits} times across the package"


def verl_client_module():
    from vagen.agent_loop import verl_client

    return verl_client


def test_the_prefix_is_rendered_the_same_way_it_is_prepended(tok):
    """A template may tokenize a plain string differently from a one-element parts list.
    If the prepend and the measurement disagree, the strip is measured against a render
    that never happened -- and the guard turns that into a loud failure, not a quiet one."""
    c = _client(tok)
    ids = c.encode([{"role": "user", "content": "OBS"}])   # must not raise
    sep = tok.decode(c._message_separator())
    assert tok.decode(ids).startswith(sep + "<|im_start|>user")


def test_the_guard_refuses_to_strip_a_span_that_does_not_start_with_the_prefix(tok, monkeypatch):
    c = _client(tok)
    monkeypatch.setattr(c, "_template_prefix", lambda: [999999, 999998])
    with pytest.raises(ValueError, match="did not begin the continuation"):
        c.encode([{"role": "user", "content": "OBS"}])
