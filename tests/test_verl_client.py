"""Tests for the verl-backed client.

Driven by a fake server and a fake processor: what is under test is which tokens the
client sends and how it accounts for them, and a real engine would only hide that. The
engine-facing half is thin by design and covered by the GPU runs.
"""

import types

import pytest

from vagen.agent_loop.verl_client import VerlClient


class Proc:
    """One token per character of the rendered text; images add a placeholder run."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kw):
        def rendered(message):
            content = message["content"]
            if isinstance(content, str):
                return content
            return "".join(str(part.get("text", "<I>")) for part in content)

        return "|" + "".join(rendered(m) for m in messages)

    def __call__(self, text=None, images=None, return_tensors=None, **kw):
        ids = [ord(c) for c in text[0]] + [7] * (len(images or []) * 3)
        return {"input_ids": types.SimpleNamespace(squeeze=lambda _: types.SimpleNamespace(tolist=lambda: ids))}


class Tok:
    def decode(self, ids, skip_special_tokens=True):
        return f"text{len(ids)}"


class Server:
    def __init__(self, engine_prompt=None):
        self.calls = []
        self.engine_prompt = engine_prompt

    async def generate(self, request_id, prompt_ids, sampling_params, image_data=None, **kw):
        self.calls.append({"prompt_ids": list(prompt_ids), "images": list(image_data or []),
                           "sampling": dict(sampling_params)})
        extra = {}
        if self.engine_prompt is not None:
            extra["prompt_token_ids"] = self.engine_prompt(prompt_ids)
        return types.SimpleNamespace(token_ids=[900, 901], log_probs=[0.1, 0.2], extra_fields=extra)


def _client(server=None, **kw):
    return VerlClient(server or Server(), Tok(), Proc(), **kw)


def _msg(text, images=()):
    return {"role": "user", "content": [{"type": "text", "text": text}], "images": list(images)}


@pytest.mark.asyncio
async def test_images_accumulate_across_a_conversation():
    """★ The engine re-processes every image on each call, so a later turn must resend
    the earlier ones -- otherwise its placeholders outnumber the features."""
    server = Server()
    c = _client(server)

    r = await c.send([_msg("a", images=["img1"])])
    await c.send([_msg("a", images=["img1"]), _msg("b", images=["img2"])], r.conversation_id)

    assert server.calls[0]["images"] == ["img1"]
    assert server.calls[1]["images"] == ["img1", "img2"]


@pytest.mark.asyncio
async def test_a_new_conversation_starts_its_own_image_list():
    server = Server()
    c = _client(server)

    await c.send([_msg("a", images=["img1"])])
    await c.send([_msg("b", images=["img2"])])          # no id: a new conversation

    assert server.calls[1]["images"] == ["img2"]


@pytest.mark.asyncio
async def test_the_per_turn_response_limit_is_applied():
    server = Server()
    c = _client(server, response_limit=16, sampling_params={"max_new_tokens": 999})
    await c.send([_msg("a")])

    assert server.calls[0]["sampling"]["max_new_tokens"] == 16


@pytest.mark.asyncio
async def test_the_engines_prompt_is_adopted():
    """★ Multimodal placeholders are expanded by the engine, so the prompt it ran is not
    the one it was handed. Training on ours would score a sequence the model never saw."""
    server = Server(engine_prompt=lambda ids: list(ids) + [42, 42])
    c = _client(server)
    await c.send([_msg("a")])

    conversation = c.conversations()[0]
    prompt = conversation.token_ids[: conversation.prompt_len]
    assert prompt[-2:] == [42, 42], "the adopted prompt is what precedes the response"
    assert conversation.token_ids[conversation.prompt_len :] == [900, 901]


@pytest.mark.asyncio
async def test_a_continuing_span_drops_the_template_preamble():
    """★ Chat templates prepend a system block. Tokenizing a mid-conversation span as if
    it began the prompt would splice that preamble into the middle of the sequence."""
    server = Server()
    c = _client(server)

    r = await c.send([_msg("a")])
    await c.send([_msg("a"), _msg("b")], r.conversation_id)

    second_span = server.calls[1]["prompt_ids"][len(server.calls[0]["prompt_ids"]) + 2 :]
    assert ord("|") not in second_span, "the template preamble leaked into a continuation"


@pytest.mark.asyncio
async def test_rows_carry_the_mask_and_the_scores():
    c = _client()
    r = await c.send([_msg("a")])
    c.reward(r.conversation_id, 1.0)

    row = c.rows()[0]
    assert row.response_mask == [1, 1]
    assert row.scores == [0.0, 1.0]
    assert row.logprobs == [0.1, 0.2]
