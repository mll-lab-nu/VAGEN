"""chat_config is an OpenRouter request body, and that body is what goes over the wire.

The OpenAI SDK raises TypeError on an argument it does not have, so OpenRouter's own
parameters (`reasoning`, `top_k`, `provider`, ...) must travel in `extra_body`. These
tests go through the real SDK with only the HTTP transport mocked, so a parameter the
SDK would refuse fails here as it would against the endpoint.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from vagen.evaluation.backends._common.retry import _is_retryable
from vagen.evaluation.backends.openrouter.openrouter import ProviderError, OpenRouterAdapter
from vagen.evaluation.client import ChatClient

_OK = {"id": "gen-1", "object": "chat.completion", "created": 0, "model": "m",
       "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"}}]}


def _adapter(reply=_OK):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=reply)

    client = AsyncOpenAI(api_key="k", base_url="http://openrouter.test/api/v1", max_retries=0,
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return OpenRouterAdapter(client=client, model="m"), bodies


def _run(adapter, **chat_config):
    return asyncio.run(adapter.acompletion([{"role": "user", "content": "hi"}], **chat_config))


def test_openrouter_parameters_reach_the_body_as_written():
    adapter, bodies = _adapter()
    assert _run(adapter, temperature=0, max_tokens=256, top_k=20,
                reasoning={"effort": "high"}, provider={"sort": "throughput"}) == "ok"
    body = bodies[0]
    assert body["temperature"] == 0 and body["max_tokens"] == 256
    assert body["top_k"] == 20
    assert body["reasoning"] == {"effort": "high"}
    assert body["provider"] == {"sort": "throughput"}


def test_unset_values_are_not_sent_and_extra_body_wins():
    adapter, bodies = _adapter()
    _run(adapter, top_p=None, reasoning={"effort": "low"}, extra_body={"reasoning": {"effort": "high"}})
    assert "top_p" not in bodies[0]
    assert bodies[0]["reasoning"] == {"effort": "high"}


def test_chat_client_limit_is_sent_as_max_tokens():
    adapter, bodies = _adapter()
    asyncio.run(ChatClient(adapter, chat_config={"max_tokens": 1024}, response_limit=256).generate([]))
    assert bodies[0]["max_tokens"] == 256


_CODES = [(502, True), ("502", True), (408, True), (429, True), (400, False), (None, True)]


def _error(code):
    return {"message": "provider returned error"} | ({"code": code} if code is not None else {})


@pytest.mark.parametrize("code, retried", _CODES)
def test_an_upstream_error_without_choices_is_retried_by_its_status(code, retried):
    adapter, _ = _adapter({"id": "gen-1", "object": "chat.completion", "created": 0, "model": "m",
                           "choices": [], "error": _error(code)})
    with pytest.raises(ProviderError) as info:
        _run(adapter)
    assert info.value.raw_response["error"]["message"] == "provider returned error"
    assert _is_retryable(info.value, ()) is retried


@pytest.mark.parametrize("code, retried", _CODES)
def test_a_choice_that_ended_in_error_is_not_a_reply(code, retried):
    choice = {"index": 0, "finish_reason": "error", "error": _error(code),
              "message": {"role": "assistant", "content": "half an ans"}}
    adapter, _ = _adapter({"id": "gen-1", "object": "chat.completion", "created": 0, "model": "m",
                           "choices": [choice]})
    with pytest.raises(ProviderError) as info:
        _run(adapter)
    assert _is_retryable(info.value, ()) is retried
