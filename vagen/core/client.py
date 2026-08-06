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

import logging

from vagen.core.tape import Conversation, Row

logger = logging.getLogger(__name__)


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


class ContextTooLarge(ValueError):
    """One call handed the model more context than the mode has room for.

    Context is everything it did not generate: the system prompt, an observation, a
    summary. Nothing bounded it before -- an environment returns what it returns, and a
    frame costs hundreds of tokens once the processor expands its placeholder -- so an
    observation that did not fit was found at the end of the episode by ``cap_token_ids``,
    if at all, by which point it is a truncated row and not an oversized observation.
    """


class InferenceClient(ABC):
    """Conversation bookkeeping, shared by every backend."""

    #: ``None`` for closed APIs that only return text.
    tokenizer: Any = None

    #: The most context one call may add, opening a conversation and continuing one. See
    #: ``vagen.harness.budget.context_limits``: the two differ because the calls do -- an
    #: opening call becomes a row's prompt region, a continuation appends to its response
    #: region. ``None`` disables the check, which is what evaluation against a closed API
    #: wants, since there is no row to fit.
    opening_limit: int | None = None
    continuation_limit: int | None = None

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
        #
        # Measured on this encode rather than a second one: encoding runs the processor,
        # which is expensive, and it records the message's images against the
        # conversation -- so measuring separately would both cost twice and ship every
        # frame twice.
        opening = conversation.prompt_len is None
        context = self.encode(messages)
        self._check_context(context, opening=opening)
        conversation.add_context(context)

        output = await self._generate_nonempty(conversation, **kwargs)

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

    #: How many times to re-ask when a generation comes back with no tokens. See
    #: ``_generate_nonempty``. Zero disables the retry and lets the empty result through.
    empty_generation_retries: int = 3

    async def _generate_nonempty(self, conversation, **kwargs) -> BackendOutput:
        """Generate, re-asking if the engine returns nothing.

        A generation with no tokens is an interruption, not an answer -- an aborted or
        pre-empted request. Retrying is safe *because* it is empty: the environment is
        stepped on the action this call returns, so if there is no action there was no
        step, and the state being re-asked about is the state that was asked about. In
        compaction the retry re-sends the summary that opened this conversation, for the
        same reason: nothing downstream of it happened.

        That safety is a property of the caller's order, not of this function, and it
        only holds while an empty response cannot reach ``env.step``. It could: ``accept``
        forwards ``response.text``, which is ``""`` and not ``None``, so the episode used
        to advance a turn on an empty action and the environment did move. Retrying here
        is what keeps the premise true.

        verl's fully-async client does the same thing a layer below (resuming from
        ``prompt_ids + token_ids`` rather than re-asking), so under that configuration
        this never fires. It is for every other configuration, where nothing does.
        """
        for attempt in range(self.empty_generation_retries + 1):
            output = await self.generate(conversation.token_ids, **kwargs)
            if output.token_ids or attempt == self.empty_generation_retries:
                return output
            logger.warning(
                "generation %d/%d returned no tokens (interrupted); re-asking. The "
                "environment has not been stepped, so the state is unchanged.",
                attempt + 1, self.empty_generation_retries,
            )
        raise AssertionError("unreachable")

    def _open(self, conversation_id: str | None) -> str:
        if conversation_id is not None:
            if conversation_id not in self._conversations:
                raise KeyError(f"unknown conversation {conversation_id!r}; pass None to start one")
            return conversation_id
        new_id = f"c{self._counter + 1}"
        # Numbered here, where the order is what actually happened. Numbering them at the
        # far end by position in ``rows()`` would be a different thing: a conversation the
        # model never spoke in is dropped there, and the survivors after the gap would
        # each move down one -- with no hole to notice, since the ids stay contiguous.
        self._conversations[new_id] = Conversation(conversation_id=new_id, ordinal=self._counter)
        self._counter += 1
        return new_id

    # ------------------------------------------------------------------ reading
    def _check_context(self, context: list[int], *, opening: bool) -> None:
        limit = self.opening_limit if opening else self.continuation_limit
        if limit is None or len(context) <= limit:
            return
        what = ("the call opening a conversation, which is the system prompt and the "
                "first observation (and under compaction the summary too)"
                if opening else "an observation")
        raise ContextTooLarge(
            f"{what} came to {len(context)} tokens, over the {limit} this mode has room "
            f"for. Image placeholders are counted expanded, as the model sees them. "
            f"Set env_response_length to what the environment actually returns, shrink "
            f"the observation (fewer or smaller frames, shorter text), or raise the "
            f"budget it has to fit inside -- see vagen/harness/budget.py for which one."
        )

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
