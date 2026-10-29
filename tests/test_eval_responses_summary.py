"""The Responses backend asks for a reasoning summary, but only until it is refused.

A model that does not reason rejects the extra `reasoning` block. Retrying without it
is right once; repeating the failed request before every call would halve throughput
against a per-minute quota.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vagen.evaluation.backends.openai_responses.openai_responses import OpenAIResponsesAdapter


class _Client:
    def __init__(self, reject_reasoning: bool):
        self.calls = []
        self._reject = reject_reasoning
        self.responses = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._reject and "reasoning" in kwargs:
            raise RuntimeError("Unsupported parameter: 'reasoning' is not supported with this model.")
        return SimpleNamespace(output_text="ok", model_dump=lambda mode: {"output_text": "ok"})


def _run(adapter, **chat_config):
    return asyncio.run(adapter.acompletion([{"role": "user", "content": "hi"}], **chat_config))


def test_refused_summary_is_not_asked_for_again():
    client = _Client(reject_reasoning=True)
    adapter = OpenAIResponsesAdapter(client=client, model="m")
    assert _run(adapter) == "ok"
    assert _run(adapter) == "ok"
    # first call: rejected + retried; second call: straight through
    assert [("reasoning" in c) for c in client.calls] == [True, False, False]


def test_summary_added_to_the_callers_reasoning():
    client = _Client(reject_reasoning=False)
    adapter = OpenAIResponsesAdapter(client=client, model="m")
    _run(adapter, reasoning={"effort": "high"})
    assert client.calls[0]["reasoning"] == {"effort": "high", "summary": "auto"}
    assert adapter.last_raw == {"output_text": "ok"}


def test_unrelated_error_is_not_retried_without_summary():
    client = _Client(reject_reasoning=False)

    async def boom(**kwargs):
        client.calls.append(kwargs)
        raise RuntimeError("rate limited")

    client.responses = SimpleNamespace(create=boom)
    adapter = OpenAIResponsesAdapter(client=client, model="m")
    with pytest.raises(RuntimeError, match="rate limited"):
        _run(adapter)
    assert len(client.calls) == 1
