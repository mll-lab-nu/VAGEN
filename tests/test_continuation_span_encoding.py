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

from model_path import local_snapshot

#: The snapshot itself, not a glob of it. `glob.glob((None or "") + "/")` is
#: `glob.glob("/")`, which returns `["/"]` -- truthy -- so with no snapshot cached the
#: skip never fired and the fixture called `from_pretrained("/")`. On a machine without
#: the model that is six errors where the file promises six skips.
MODEL = local_snapshot()
pytestmark = pytest.mark.skipif(MODEL is None, reason="Qwen2.5-VL snapshot not present")


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def _client(tok, *, mid_conversation=True):
    from vagen.models.qwen import QwenModelAdapter

    adapter = QwenModelAdapter(tok)

    class _Render:
        def encode(self, messages):
            return adapter.render(messages, opening=not mid_conversation)[0]

        def _message_separator(self):
            return adapter.message_separator()

        def _template_prefix(self):
            return adapter._template_prefix()

    out = _Render()
    out.model = adapter
    return out


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
    """A second copy of the placeholder is a second chance for the strip to be wrong.

    The placeholder is more than one turn now -- a system turn to suppress the template's
    default system block, plus a user turn because Qwen3.5's template raises
    ``No user query found in messages.`` without one -- so a bare count of the literal no
    longer expresses the invariant. What has to hold is that every render path reads the
    same single definition, which is checked three ways below.
    """
    import pathlib

    module = model_adapter_module()
    root = pathlib.Path(module.__file__).parent

    # Scoped to the package, not one module: the duplicate this guards against lived in
    # a *different* file, so scanning only verl_client could never have seen it.
    elsewhere = {
        p.name: n
        for p in root.glob("*.py")
        if p.name != "qwen.py" and (n := p.read_text().count('"content": "placeholder"'))
    }
    assert not elsewhere, f"the placeholder turn is restated outside verl_client: {elsewhere}"

    src = (root / "qwen.py").read_text()
    assert src.count("_PLACEHOLDER_TURNS = [") == 1, "more than one placeholder definition"
    # Tied to the constant rather than to a fixed number, so adding or dropping a turn
    # keeps this honest instead of quietly permitting a restated copy alongside it.
    assert src.count('"content": "placeholder"') == len(module._PLACEHOLDER_TURNS)


def model_adapter_module():
    from vagen.models.qwen import qwen

    return qwen


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
    monkeypatch.setattr(c.model, "_template_prefix", lambda: [999999, 999998])
    with pytest.raises(ValueError, match="did not begin the continuation"):
        c.encode([{"role": "user", "content": "OBS"}])


#: Qwen3.5's rule, reduced to the two clauses that interact with the placeholder: inject a
#: default system block when the caller supplied none, and refuse a message list with no
#: user turn. Written out rather than pulled from the Hub so the regression is pinned
#: without a 4B download -- the real template's own wording is quoted in verl_client.
_DEMANDS_A_USER_TURN = (
    "{%- if messages[0].role != 'system' %}"
    "{{- '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
    "{%- endif %}"
    "{%- set ns = namespace(has_user=false) %}"
    "{%- for m in messages %}{%- if m.role == 'user' %}{%- set ns.has_user = true %}"
    "{%- endif %}{%- endfor %}"
    "{%- if not ns.has_user %}"
    "{{- raise_exception('No user query found in messages.') }}"
    "{%- endif %}"
    "{%- for m in messages %}"
    "{{- '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"
)


@pytest.fixture()
def strict_tok(tok):
    """The same tokenizer, wearing a template that demands a user turn."""
    import copy

    t = copy.deepcopy(tok)
    t.chat_template = _DEMANDS_A_USER_TURN
    return t


def test_a_template_that_demands_a_user_turn_can_still_be_measured(strict_tok):
    """Qwen3.5 raises on a message list with no user turn.

    A system-only placeholder cannot even be rendered on its own under that rule, so
    `_template_prefix` -- which renders exactly that -- died before any continuation was
    reached, and the whole family was unrunnable. The placeholder carries a user turn for
    this reason; the assertion is simply that measuring it no longer raises.
    """
    c = _client(strict_tok)
    assert c._template_prefix(), "the prefix must be measurable under Qwen3.5's rule"


def test_an_assistant_only_span_survives_a_template_that_demands_a_user_turn(strict_tok):
    """The span that actually broke: a delta carrying no user message of its own."""
    c = _client(strict_tok)
    ids = c.encode([{"role": "assistant", "content": "REPLY"}])
    text = strict_tok.decode(ids)
    assert "REPLY" in text
    # The placeholder is measurement scaffolding; none of it may reach the sequence.
    assert "placeholder" not in text
    # And the suppressed default block must not survive the strip either.
    assert "helpful assistant" not in text
