"""A reply cut off at ``max_tokens`` must not look like a reply that never came.

A reasoning model spends its output budget on hidden thinking before it writes
anything, so a budget that is merely too small returns ``finish_reason=length``
with empty content. Dropping the reason makes that indistinguishable from a
refusal, and a harness re-asks against the same budget -- which cannot help --
and then scores the episode as a failure the model never had a chance to avoid.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vagen.evaluation.backends.openai.openai import OpenAIAdapter, OutputBudgetExceeded


def _response(content, finish_reason):
    message = SimpleNamespace(content=content)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=None,
    )


class _Client:
    """Stands in for AsyncOpenAI, returning one canned response."""

    def __init__(self, response):
        self._response = response
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **_kwargs):
        return self._response


def _call(response):
    adapter = OpenAIAdapter(client=_Client(response), model="a-reasoning-model")
    return asyncio.run(adapter.acompletion([{"role": "user", "content": "hi"}]))


def test_truncated_with_no_content_raises():
    """The whole budget went to hidden reasoning: loud, not an empty string."""
    with pytest.raises(OutputBudgetExceeded) as excinfo:
        _call(_response("", "length"))
    assert "max_tokens" in str(excinfo.value)


def test_truncated_whitespace_only_raises():
    """Whitespace is not content; it would fail to parse as an action either way."""
    with pytest.raises(OutputBudgetExceeded):
        _call(_response("  \n ", "length"))


def test_truncated_with_partial_content_is_kept():
    """Cut off but it wrote something -- that is the caller's to judge, not ours."""
    assert _call(_response("partial answer", "length")) == "partial answer"


def test_normal_completion_is_unchanged():
    assert _call(_response("done", "stop")) == "done"


def test_empty_reply_without_length_is_still_empty():
    """A refusal or a content filter is a different failure and keeps its old shape."""
    assert _call(_response(None, "content_filter")) == ""


def test_missing_finish_reason_is_tolerated():
    """Not every OpenAI-compatible server sets it."""
    message = SimpleNamespace(content="ok")
    resp = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)
    assert _call(resp) == "ok"
