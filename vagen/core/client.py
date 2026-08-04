"""Inference client — the only layer that knows about tokens.

The harness works in text and the env works in text; everything token-level is here, and
it is written once. What varies between experiments is the harness and the env, and
neither can reach a token through this interface.

A conversation id is the whole protocol. Passing one continues that conversation; passing
``None`` starts a new one. Concat keeps the same id for an episode, no-concat drops it
every turn, and compaction drops it when a budget is hit — three points on one axis
rather than three mechanisms. One conversation becomes one training row.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from vagen.core.tape import Conversation, Row


@dataclass
class Response:
    """What the harness gets back. Text plus the id needed to continue."""

    text: str
    conversation_id: str
    token_ids: Optional[list[int]] = None
    logprobs: Optional[list[float]] = None


@dataclass
class BackendOutput:
    """What a backend reports for one generation."""

    text: str
    token_ids: list[int]
    logprobs: Optional[list[float]] = None
    # The prompt the backend actually ran. Multimodal placeholders are expanded by the
    # backend, so this is not always the prompt it was handed -- see Conversation.
    prompt_token_ids: Optional[list[int]] = None


class InferenceClient(ABC):
    """Conversation bookkeeping, shared by every backend."""

    #: ``None`` for closed APIs that only return text.
    tokenizer: Any = None

    def __init__(self):
        self._conversations: dict[str, Conversation] = {}
        self._counter = 0

    @property
    def returns_token_ids(self) -> bool:
        """Training needs the ids. Checked at construction, not mid-episode."""
        return self.tokenizer is not None

    # ------------------------------------------------------------------ backend
    @abstractmethod
    def encode(self, messages: list[Any]) -> list[int]:
        """Render messages to tokens. Called only on messages not yet sent."""

    @abstractmethod
    async def generate(self, prompt_ids: list[int], **kwargs) -> BackendOutput:
        """Run the model."""

    # -------------------------------------------------------------------- send
    async def send(self, messages: list[Any], conversation_id: str | None = None, **kwargs) -> Response:
        conversation_id = self._open(conversation_id)
        conversation = self._conversations[conversation_id]

        # Encode exactly what the harness handed over. The harness already sends only
        # what is new -- deduplicating again here silently dropped every observation
        # after the first, since it sliced a one-message delta against a count of the
        # messages already sent.
        conversation.add_context(self.encode(messages))

        output = await self.generate(conversation.token_ids, **kwargs)

        # Adopt what the backend ran, so the sequence trained on is the sequence sampled
        # from. This is also what repairs any seam left by rendering messages
        # incrementally, since the correction lands on the span just added.
        if output.prompt_token_ids is not None:
            conversation.adopt_prompt(output.prompt_token_ids)
        conversation.add_response(output.token_ids, output.logprobs)

        return Response(
            text=output.text,
            conversation_id=conversation_id,
            token_ids=output.token_ids,
            logprobs=output.logprobs,
        )

    def _open(self, conversation_id: str | None) -> str:
        if conversation_id is not None:
            if conversation_id not in self._conversations:
                raise KeyError(f"unknown conversation {conversation_id!r}; pass None to start one")
            return conversation_id
        self._counter += 1
        new_id = f"c{self._counter}"
        self._conversations[new_id] = Conversation(conversation_id=new_id)
        return new_id

    # ------------------------------------------------------------------ reading
    def reward(self, conversation_id: str, value: float | list[float]) -> None:
        """Credit the turn that just happened in this conversation."""
        self._conversations[conversation_id].add_reward(value)

    def rows(self) -> list[Row]:
        """One row per conversation the model spoke in.

        A conversation with no model output — a new one opened immediately before a
        terminal step — carries no gradient and is dropped rather than padded into the
        batch as an empty sequence.
        """
        return [c.row() for c in self._conversations.values() if c.is_trainable()]

    def usage(self, conversation_id: str) -> int:
        """How large this conversation has grown, in whatever unit the backend counts.

        Tokens here; a closed API would report the prompt size from its own ``usage``,
        counted by its own tokenizer, so a budget will trigger at slightly different
        points than in training. Log where it actually fires.
        """
        return len(self._conversations[conversation_id].token_ids)

    def conversations(self) -> list[Conversation]:
        return list(self._conversations.values())
