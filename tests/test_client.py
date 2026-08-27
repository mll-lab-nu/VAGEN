"""Tests for the conversation bookkeeping shared by every backend.

Exercised against a fake backend: what is under test is which tokens end up where, and a
real engine would only make that harder to see. The backend-specific parts (verl's
LLMServerClient, a chat API) are thin by design and covered by the GPU runs.
"""

import pytest

from vagen.rollout.client import BackendOutput, InferenceClient, Response
from vagen.rollout.trajectory import MaskMisaligned


class FakeClient(InferenceClient):
    """One token per character; the backend echoes a fixed reply."""

    tokenizer = object()

    def __init__(self, reply=(90, 91), engine_prompt=None):
        super().__init__()
        self.reply = list(reply)
        self.engine_prompt = engine_prompt      # what the backend claims it ran
        self.calls = []

    def encode(self, messages):
        return [ord(c) for m in messages for c in m]

    async def generate(self, prompt_ids, **kwargs):
        self.calls.append(list(prompt_ids))
        prompt = self.engine_prompt(prompt_ids) if callable(self.engine_prompt) else self.engine_prompt
        return BackendOutput(text="ok", token_ids=self.reply, prompt_token_ids=prompt)


# ------------------------------------------------------------------ the protocol


@pytest.mark.asyncio
async def test_no_id_opens_a_conversation():
    c = FakeClient()
    r = await c.send(["ab"])

    assert isinstance(r, Response) and r.conversation_id
    assert len(c.conversations()) == 1


@pytest.mark.asyncio
async def test_the_same_id_continues_one_conversation():
    """★ concat: one conversation for the whole episode, so one training row."""
    c = FakeClient()
    first = await c.send(["ab"])
    await c.send(["cd"], first.conversation_id)

    assert len(c.conversations()) == 1
    assert len(c.rows()) == 1


@pytest.mark.asyncio
async def test_dropping_the_id_starts_another():
    """★ no-concat: a conversation per turn, so a row per turn. Same code path as
    concat -- the only difference is whether the id is passed back."""
    c = FakeClient()
    await c.send(["ab"])
    await c.send(["cd"])

    assert len(c.rows()) == 2


@pytest.mark.asyncio
async def test_an_unknown_id_is_rejected():
    """Silently opening a new conversation would split one episode's row in two."""
    c = FakeClient()
    with pytest.raises(KeyError, match="unknown conversation"):
        await c.send(["ab"], "nope")


# --------------------------------------------------------------- what gets sent


@pytest.mark.asyncio
async def test_what_the_caller_hands_over_is_what_gets_encoded():
    """★ The harness already sends only what is new. Deduplicating again here dropped
    every observation after the first -- and each side's unit tests passed, because each
    was written against its own idea of who deduplicates."""
    c = FakeClient()
    first = await c.send(["ab"])
    await c.send(["cd"], first.conversation_id)     # the delta, as a harness sends it

    assert c.calls[1] == [ord("a"), ord("b")] + c.reply + [ord("c"), ord("d")]


@pytest.mark.asyncio
async def test_every_observation_reaches_the_context():
    """The failure this guards against is quiet: the conversation keeps generating, the
    rows stay well-formed, and the model simply never sees what the environment said."""
    c = FakeClient()
    first = await c.send(["ab"])
    await c.send(["cd"], first.conversation_id)
    await c.send(["ef"], first.conversation_id)

    ids = c.conversations()[0].token_ids
    for text in ("ab", "cd", "ef"):
        assert all(ord(ch) in ids for ch in text), f"{text!r} never entered the context"


@pytest.mark.asyncio
async def test_the_model_output_is_the_masked_in_part():
    c = FakeClient()
    first = await c.send(["ab"])
    await c.send(["cd"], first.conversation_id)

    row = c.rows()[0]
    assert row.prompt_ids == [ord("a"), ord("b")]
    assert row.response_mask == [1, 1, 0, 0, 1, 1]


# -------------------------------------------------------------------- adoption


@pytest.mark.asyncio
async def test_the_backends_prompt_is_adopted():
    """★ The backend expands multimodal placeholders its own way, so the prompt it runs
    is not always the one it was handed. Training on ours would compute log-probs over a
    sequence the model never saw."""
    c = FakeClient(engine_prompt=lambda ids: list(ids) + [999])
    await c.send(["ab"])

    assert c.conversations()[0].token_ids[:3] == [ord("a"), ord("b"), 999]


@pytest.mark.asyncio
async def test_adoption_repairs_the_seam_left_by_incremental_encoding():
    """Rendering messages one turn at a time can tokenize the join differently from
    rendering them together; the correction lands on the span just added."""
    c = FakeClient(engine_prompt=lambda ids: list(ids) + [777])
    first = await c.send(["ab"])
    await c.send(["cd"], first.conversation_id)

    conversation = c.conversations()[0]
    assert len(conversation.token_ids) - len(conversation.mask) == conversation.prompt_len
    assert sum(conversation.mask) == 4, "adoption must not change how many tokens are trained on"


@pytest.mark.asyncio
async def test_a_backend_that_reports_no_prompt_is_tolerated():
    """Non-vLLM backends do not carry the field; absence is not evidence of a problem."""
    c = FakeClient(engine_prompt=None)
    await c.send(["ab"])

    assert c.rows()[0].prompt_ids == [ord("a"), ord("b")]


# ---------------------------------------------------------------------- output


@pytest.mark.asyncio
async def test_a_silent_conversation_is_dropped_not_padded():
    """★ Opening a conversation immediately before a terminal step leaves a row with no
    gradient; padding it into the batch would dilute the loss with an empty sequence."""
    c = FakeClient()
    await c.send(["ab"])
    c._open(None)   # opened, never spoken in

    assert len(c.conversations()) == 2
    assert len(c.rows()) == 1


@pytest.mark.asyncio
async def test_reward_lands_on_the_conversation_it_belongs_to():
    c = FakeClient()
    first = await c.send(["ab"])
    second = await c.send(["cd"])

    c.reward(second.conversation_id, 1.0)
    rows = {conv.conversation_id: conv for conv in c.conversations()}
    assert rows[second.conversation_id].scores == [0.0, 1.0]
    assert rows[first.conversation_id].scores == [0.0, 0.0]


@pytest.mark.asyncio
async def test_a_misaligned_reward_vector_raises():
    c = FakeClient()
    r = await c.send(["ab"])

    with pytest.raises(MaskMisaligned, match="align them"):
        c.reward(r.conversation_id, [0.1, 0.2, 0.3])


def test_a_client_without_a_tokenizer_cannot_train():
    """A chat API returns text only. Rejecting it at construction beats discovering it
    halfway through an episode."""

    class TextOnly(FakeClient):
        tokenizer = None

    assert TextOnly().returns_token_ids is False
    assert FakeClient().returns_token_ids is True
