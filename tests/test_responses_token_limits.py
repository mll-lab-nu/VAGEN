"""Responses backends preserve the evaluation client's generation ceiling."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from openai import AsyncAzureOpenAI, AsyncOpenAI
from PIL import Image

from vagen.evaluation.backends.openai_responses import OpenAIResponsesAdapter
from vagen.evaluation.client import ChatClient


@pytest_asyncio.fixture(params=["openai", "azure"])
async def sdk_client(request, monkeypatch):
    if request.param == "azure":
        client = AsyncAzureOpenAI(
            api_key="test-local-not-a-credential",
            azure_endpoint="https://example.invalid",
            api_version="2025-04-01-preview",
        )
    else:
        client = AsyncOpenAI(api_key="test-local-not-a-credential")
    # Keep the real SDK's Responses argument checking and body transformation;
    # replace only the HTTP dispatch so no provider request can leave the test.
    post = AsyncMock(return_value=SimpleNamespace(output_text="move left"))
    monkeypatch.setattr(client, "post", post)
    try:
        yield client, post
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "expected_limit"),
    [
        ({"max_tokens": 128}, 128),
        ({"max_output_tokens": 64}, 64),
        ({"max_tokens": 128, "max_output_tokens": 64}, 64),
        ({"max_tokens": 32, "max_output_tokens": 128}, 32),
        ({"max_tokens": None, "max_output_tokens": 64}, 64),
        ({"max_tokens": 128, "max_output_tokens": None}, 128),
    ],
)
async def test_responses_sdk_receives_the_smaller_token_limit(
    sdk_client, config, expected_limit
):
    client, post = sdk_client
    adapter = OpenAIResponsesAdapter(client, "test-model")
    messages = [adapter.format_user_turn("Inspect <image>", [Image.new("RGB", (2, 2))])]

    result = await adapter.acompletion(messages, temperature=0.2, **config)

    assert result == "move left"
    post.assert_awaited_once()
    assert post.call_args.args == ("/responses",)
    body = post.call_args.kwargs["body"]
    assert "max_tokens" not in body
    assert body["max_output_tokens"] == expected_limit
    assert body["temperature"] == 0.2
    assert body["input"] == messages
    assert body["input"][0]["content"][1]["type"] == "input_image"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        {},
        {"max_tokens": None},
        {"max_output_tokens": None},
        {"max_tokens": None, "max_output_tokens": None},
    ],
)
async def test_no_configured_limit_keeps_the_api_default(config):
    create = AsyncMock(return_value=SimpleNamespace(output_text="done"))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))

    result = await OpenAIResponsesAdapter(client, "test-model").acompletion(
        [], **config
    )

    assert result == "done"
    create.assert_awaited_once_with(model="test-model", input=[])


@pytest.mark.asyncio
async def test_evaluation_response_limit_reaches_the_responses_sdk(sdk_client):
    client, post = sdk_client
    adapter = OpenAIResponsesAdapter(client, "test-model")
    evaluation_client = ChatClient(
        adapter,
        chat_config={"max_tokens": 256, "max_output_tokens": 512},
        response_limit=77,
    )

    result = await evaluation_client.create([{"role": "user", "content": "act"}])

    assert result.text == "move left"
    assert post.call_args.kwargs["body"]["max_output_tokens"] == 77
