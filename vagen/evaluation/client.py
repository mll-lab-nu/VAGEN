"""Rollout ``InferenceClient`` over a hosted chat endpoint.

The backend half of evaluation, so that evaluation can run on the same episode loop and
the same harnesses as training. ``core/runner.py`` says it drives "a verl rollout and a
closed chat API"; this is the closed chat API, and until now nothing took it up on that --
``evaluate/`` carried its own turn loop, which hardcoded concat and could not express
compaction at all.

Two things differ from ``VerlClient`` and both are consequences of talking to a chat API
rather than to an engine:

**Messages, not token ids.** ``InferenceClient.send`` hands ``generate`` the conversation's
token ids, which a chat endpoint has no use for. So ``encode`` records the rendered API
messages against the active conversation and ``generate`` reads them back -- the same
pattern ``VerlClient`` uses for image frames.

**Sizes are measured when they can be.** Set ``model:`` in the eval config -- normally the
same id you are serving -- and this loads that model's *processor* and prices text and
frames exactly as training does. Only a closed API, where there is no processor to load,
falls back to estimating: 4 characters to the token, and ``tokens_per_image`` for a frame.

That fallback is a last resort, not a knob to tune. Getting it wrong is not cosmetic: the
same number drives the compaction trigger and the observation ceiling, so an estimate far
from the truth makes `compact` fire on conversations that are not full, or makes a ceiling
copied from a training config cut every observation to nothing.
"""

from __future__ import annotations

import logging
from typing import Any

from vagen.rollout import BackendOutput, InferenceClient

logger = logging.getLogger(__name__)

#: The marker an environment puts where a frame goes.
IMAGE_PLACEHOLDER = "<image>"

#: Characters per token when no tokenizer is available. Only the *ratio* matters, and only
#: for deciding when a conversation is full: 4 is the usual English rule of thumb, and an
#: image placeholder is charged separately below because it is worth hundreds of tokens
#: and would otherwise count as the ~20 characters of its data URL.
CHARS_PER_TOKEN = 4

#: What one image costs when no tokenizer can be asked. There is no good default: a 96x96
#: sokoban frame is 96 tokens under Qwen2.5-VL's processor and a 512x512 one is over a
#: thousand, so this is a per-environment number wearing a constant's clothes. 800 errs
#: high because under-counting overflows the model's real context, which fails at the API
#: rather than anywhere this layer can recover.
#:
#: ★ It is not only an overflow guard -- it feeds the compaction trigger. Set eight times
#: too high, as it is for sokoban, and `compact` decides a conversation is full before it
#: has bought a turn: measured, CompactionMakesNoProgress on every episode, reporting an
#: "832-token observation" that really costs 96. Set `tokens_per_image` in the eval config
#: to what your environment actually returns, or pass a processor.
DEFAULT_TOKENS_PER_IMAGE = 800


class ChatClient(InferenceClient):
    """Talks to a hosted endpoint through an ``EvaluationBackend``.

    ★ One per episode. ``generate`` writes the reply back to ``_api_messages[_active]``
    *after* awaiting the endpoint, so a client shared between two concurrent episodes would
    append one episode's reply to the other's conversation. ``arun_episode`` constructs it,
    and nothing should hand it around.
    """

    #: ★ Zero, unlike the engine-backed client. The base class retries an empty generation
    #: because an engine returning nothing is an interruption -- but a chat API returning
    #: "" is a refusal or a content filter, and asking again three times just pays for it
    #: four times. Measured: 4 calls per refusal, then a 0-turn episode the summary files
    #: as normal.
    empty_generation_retries = 0

    def __init__(self, adapter, chat_config: dict | None = None, tokenizer=None,
                 response_limit: int | None = None,
                 tokens_per_image: int = DEFAULT_TOKENS_PER_IMAGE,
                 processor=None):
        super().__init__()
        self.adapter = adapter
        self.chat_config = dict(chat_config or {})
        #: response_length_per_turn. The harness asks for the room it has left, which on
        #: the first turn is the whole region -- so without this clamp a turn could spend
        #: the entire episode's budget. VerlClient does the same.
        self.response_limit = response_limit
        #: Optional. Present, sizes are exact; absent, they are estimated from characters.
        self.tokenizer = tokenizer
        #: Optional, and better: a processor prices frames too, which a tokenizer cannot.
        self.processor = processor
        self.tokens_per_image = tokens_per_image
        self._active: str | None = None
        #: conversation id -> the API messages sent so far. The harness decides *which*
        #: messages a call carries; this only remembers what they rendered to.
        self._api_messages: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------ bookkeeping
    def _open(self, conversation_id: str | None) -> str:
        resolved = super()._open(conversation_id)
        self._active = resolved
        return resolved

    def messages(self, conversation_id: str) -> list[dict]:
        """What was actually sent, for the transcript dump."""
        return list(self._api_messages.get(conversation_id) or [])

    # ------------------------------------------------------------------ encoding
    def encode(self, messages: list[dict]) -> list[int]:
        """Render to API messages, record them, and report their size.

        The return value is a list of the right *length* rather than real ids: nothing in
        the evaluation path consumes evaluation token ids -- there is no row to train --
        and the harness only ever asks how long a conversation is.
        """
        rendered, size = self._render(messages)
        if self._active is not None:
            self._api_messages.setdefault(self._active, []).extend(rendered)
        return [0] * size

    def render(self, messages: list[dict]):
        """Size these messages without recording them. See ``InferenceClient.measure``:
        asking how big an observation is must not also send it."""
        _, size = self._render(messages)
        return [0] * size

    def _render(self, messages: list[dict]) -> tuple[list[dict], int]:
        rendered, size = [], 0
        for m in messages:
            text, images = _text_and_images(m)
            role = m.get("role", "user")
            if role == "system":
                api = self.adapter.format_system(text, images)
            elif role == "assistant":
                api = self.adapter.format_assistant_turn(text)
            else:
                api = self.adapter.format_user_turn(text, images)
            rendered.append(api)
            size += self._size(text, images)
        return rendered, size

    def _size(self, text: str, images: list) -> int:
        if self.processor is not None:
            # Exactly what training measures: the processor expands each placeholder into
            # however many tokens that frame really costs at this resolution.
            try:
                expanded = text.replace(
                    IMAGE_PLACEHOLDER, "<|vision_start|><|image_pad|><|vision_end|>")
                out = self.processor(text=[expanded], images=list(images) or None,
                                     return_tensors="pt")
                return int(out["input_ids"].shape[-1])
            except Exception:   # noqa: BLE001 - a processor that cannot render this
                pass            # falls through to the estimate rather than killing the run
        if self.tokenizer is not None:
            n = len(self.tokenizer.encode(text))
        else:
            n = -(-len(text) // CHARS_PER_TOKEN)
        return n + len(images) * self.tokens_per_image

    # ------------------------------------------------------------------ generation
    async def generate(self, prompt_ids: list[int], **kwargs) -> BackendOutput:
        """Send the conversation as messages. ``prompt_ids`` is the length, not the input.

        ``max_tokens`` follows the harness's per-call limit when it set one, so that
        ``response_length_per_turn`` means the same thing under evaluation as it does in
        training instead of being whatever the yaml's chat_config happened to say.
        """
        del prompt_ids
        params = dict(self.chat_config)
        params.update(kwargs.pop("sampling_params", {}) or {})
        limit = params.pop("max_new_tokens", None)
        for candidate in (self.response_limit, params.get("max_tokens")):
            if candidate:
                limit = min(limit, candidate) if limit else candidate
        if limit:
            params["max_tokens"] = limit

        messages = self._api_messages.get(self._active) or []
        text = await self.adapter.acompletion(messages, **params)
        text = text if isinstance(text, str) else ""
        # Recorded so the transcript shows the exchange and not just what we sent.
        if self._active is not None and text:
            self._api_messages[self._active].append(self.adapter.format_assistant_turn(text))
        return BackendOutput(text=text, token_ids=[0] * self._size(text, []))


def _text_and_images(message: dict) -> tuple[str, list]:
    """Split one harness message into the pair the adapter formats from."""
    images = list(message.get("images") or [])
    content = message.get("content", "")
    if isinstance(content, str):
        return content, images
    parts = []
    for part in content:
        if part.get("type") == "text":
            parts.append(part.get("text", ""))
        elif part.get("type") == "image":
            parts.append("<image>")
    return "".join(parts), images
