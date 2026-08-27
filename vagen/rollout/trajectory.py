"""Token trajectory for one conversation — one training row.

Everything token-level lives behind the client (§7), and this is the part of it with no
dependencies: no torch, no verl, no transformers. A conversation is a sequence of
alternating spans — context we supplied, tokens the model produced — and the record
exists to keep the mask describing exactly that, through the one operation that can
disturb it.

That operation is adopting the engine's prompt. The engine expands multimodal
placeholders its own way, so the prompt it runs is not always the one it was handed;
training on the locally tokenized version means computing log-probs over a sequence the
model never saw. Both are well-formed, neither side sees both, and the loss stays finite,
so nothing reports it. Adopting the engine's version removes the possibility, at the cost
of having to move the mask with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class MaskMisaligned(RuntimeError):
    """The token record and its mask stopped describing the same sequence."""


@dataclass
class Row:
    """What one conversation contributes to a training batch."""

    conversation_id: str | None
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    logprobs: list[float]
    scores: list[float]
    #: True only when every non-empty model response supplied backend logprobs.  The
    #: numeric vector cannot encode this because 0.0 is both a valid logprob and the
    #: alignment fill used when values are unavailable.
    logprobs_complete: bool = True
    #: (start, end) of each model output, as offsets into ``response_ids``. One entry per
    #: turn: a conversation holds several under concat, exactly one under no_concat.
    #: Without these a conversation is one undifferentiated blob and its turns cannot be
    #: told apart, which is why turn numbering had silently become conversation numbering.
    response_spans: list[tuple[int, int]] = field(default_factory=list)
    #: Which conversation this is, counting from 0 in the order they were opened.
    #:
    #: Not the position in ``rows()``. A conversation the model never spoke in is dropped
    #: there, and numbering the survivors moves everything after the gap down by one --
    #: with no hole to notice, since the ids stay contiguous. Under no_concat the id *is*
    #: the environment step, so that would record turn n+1's behaviour against turn n.
    ordinal: int = 0

    def __post_init__(self):
        if not (len(self.response_ids) == len(self.response_mask) == len(self.logprobs) == len(self.scores)):
            raise MaskMisaligned(
                f"row is inconsistent: {len(self.response_ids)} response tokens, "
                f"{len(self.response_mask)} mask entries, {len(self.logprobs)} logprobs"
            )


@dataclass
class Conversation:
    """One conversation's tokens, and which of them the model produced.

    ``prompt_len`` marks the end of the opening context. Everything after it is the
    trainable region, whether the model produced it (mask 1) or we appended it as an
    observation (mask 0).
    """

    conversation_id: str | None = None
    #: Which conversation this is, from 0, in the order they were opened.
    ordinal: int = 0
    token_ids: list[int] = field(default_factory=list)
    mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    _logprobs_complete: bool = True
    scores: list[float] = field(default_factory=list)
    prompt_len: int | None = None
    # Span of the most recent model output, so a turn's reward lands on that turn.
    _last_response: tuple[int, int] | None = None
    #: Every model output so far, in order, as (start, end) into the trainable region.
    response_spans: list[tuple[int, int]] = field(default_factory=list)
    # Length of the newest context span; the only region an adoption can resize.
    _tail_context_len: int | None = None

    # ------------------------------------------------------------------ writing
    def add_context(self, ids: list[int]) -> None:
        """Tokens we supplied: a rendered prompt, or an observation between turns."""
        self.token_ids += list(ids)
        if self.prompt_len is None:
            # Still the opening context; the trainable region has not started.
            self._tail_context_len = None
            return
        self.mask += [0] * len(ids)
        self.logprobs += [0.0] * len(ids)
        self.scores += [0.0] * len(ids)
        self._tail_context_len = len(ids)

    def add_response(self, ids: list[int], logprobs: list[float] | None = None) -> None:
        """Tokens the model produced."""
        if logprobs is not None and len(logprobs) != len(ids):
            raise MaskMisaligned(
                f"response has {len(ids)} tokens but {len(logprobs)} logprobs"
            )
        if ids and logprobs is None:
            self._logprobs_complete = False
        if self.prompt_len is None:
            self.prompt_len = len(self.token_ids)
        start = len(self.mask)
        self.token_ids += list(ids)
        self.mask += [1] * len(ids)
        self.logprobs += list(logprobs) if logprobs else [0.0] * len(ids)
        self.scores += [0.0] * len(ids)
        self._last_response = (start, len(self.mask))
        self.response_spans.append(self._last_response)
        self._tail_context_len = 0

    # ---------------------------------------------------------------- adopting
    def adopt_prompt(self, engine_ids: list[int]) -> None:
        """Replace what we tokenized with what the engine ran.

        Only the newest context span can have changed length. Everything before it was
        adopted on an earlier call and is therefore already in the engine's own form, and
        re-expanding an already-expanded prompt is idempotent — the deduplication that
        precedes the round trip is exactly the inverse of the expansion. So the whole
        delta falls on a trailing run of zeros, and the mask can be corrected without
        locating anything inside the token stream.

        The assumption is checked against the tokens, not against a length. Comparing
        ``len(token_ids) - len(mask)`` to ``prompt_len`` afterwards proves nothing in the
        branch that needs proving: the correction grows the mask by exactly the delta
        that grew the tokens, so the difference is unchanged whatever moved. A delta
        split between the opening region and the tail passes it, and then ``prompt_len``
        points short of the real boundary -- placeholder tokens end up at the head of
        ``response_ids`` with mask 1, trained on as if the model had written them.
        """
        delta = len(engine_ids) - len(self.token_ids)

        # Everything before the newest context must survive byte for byte -- but only
        # once there IS something before it. While the opening prompt is still being
        # assembled every token is context, and the engine's re-expansion of an image
        # placeholder lands in the middle of it, so requiring a byte-identical prefix
        # rejected precisely the case adoption exists for. Fatal, too: nothing catches
        # MaskMisaligned, so it took the whole batch down.
        head_len = 0 if self.prompt_len is None else len(self.token_ids) - (self._tail_context_len or 0)
        if engine_ids[:head_len] != self.token_ids[:head_len]:
            differs = next(
                (i for i in range(min(head_len, len(engine_ids)))
                 if engine_ids[i] != self.token_ids[i]),
                min(head_len, len(engine_ids)),
            )
            raise MaskMisaligned(
                f"the engine re-expanded something before the newest context: token "
                f"{differs} of the first {head_len} changed. Only the trailing context "
                f"may differ, or the opening/response boundary moves without prompt_len "
                f"following it."
            )

        if delta and self._tail_context_len:
            adjusted = self._tail_context_len + delta
            if adjusted < 0:
                raise MaskMisaligned(
                    f"the engine's prompt is {-delta} tokens shorter than the {self._tail_context_len}-token "
                    "context it re-expanded, so the change was not confined to it"
                )
            keep = len(self.mask) - self._tail_context_len
            self.mask = self.mask[:keep] + [0] * adjusted
            self.logprobs = self.logprobs[:keep] + [0.0] * adjusted
            self.scores = self.scores[:keep] + [0.0] * adjusted
            self._tail_context_len = adjusted

        self.token_ids = list(engine_ids)

        if self.prompt_len is None:
            # Nothing trainable yet: the whole thing is still the opening context.
            return
        if len(self.token_ids) - len(self.mask) != self.prompt_len:
            raise MaskMisaligned(
                f"after adopting the engine's prompt, {len(self.token_ids)} tokens minus "
                f"{len(self.mask)} masked is not the {self.prompt_len}-token opening context. "
                "Some region other than the newest context was re-expanded."
            )

    # ------------------------------------------------------------------ reading
    @property
    def response_len(self) -> int:
        """Tokens in the trainable region: everything after the opening call.

        Not ``len(token_ids)``. A budget written against ``max_response_length`` has to
        be measured against the region that budget names -- counting the prompt region
        too over-reserves by the system prompt, which on Sokoban is most of it.
        """
        return 0 if self.prompt_len is None else len(self.token_ids) - self.prompt_len

    def is_trainable(self) -> bool:
        """False for a conversation the model never spoke in — a new conversation
        immediately followed by a terminal step. Such rows carry no gradient and are
        dropped rather than padded."""
        return self.prompt_len is not None and any(self.mask)

    def row(self) -> Row:
        if self.prompt_len is None:
            raise MaskMisaligned("conversation has no model output; check is_trainable() first")
        return Row(
            ordinal=self.ordinal,
            conversation_id=self.conversation_id,
            prompt_ids=self.token_ids[: self.prompt_len],
            response_ids=self.token_ids[self.prompt_len :],
            response_mask=list(self.mask),
            logprobs=list(self.logprobs),
            scores=list(self.scores),
            logprobs_complete=self._logprobs_complete,
            response_spans=list(self.response_spans),
        )

    def add_reward(self, reward: float | list[float]) -> None:
        """Credit the most recent turn.

        ★ Placed on *that turn's* last model token, not the conversation's. A concat
        episode is many turns in one conversation, and summing them onto the final token
        would erase which turn earned what -- the credit assignment the turn structure
        exists to provide.

        A vector must already be aligned to that turn's response; its length is checked
        rather than assumed, because an env that re-encoded the text to build one would
        otherwise misalign it silently.
        """
        if self._last_response is None:
            raise MaskMisaligned("no model output to credit; the environment acted on nothing")
        start, end = self._last_response
        if end == start:
            # An aborted generation returns no tokens. There is nowhere to put the
            # credit: `end - 1` would land on the observation before it and pay the
            # environment's own text, or on nothing at all when the turn is the first.
            return

        if isinstance(reward, (int, float)):
            self.scores[end - 1] += float(reward)
            return

        if len(reward) != end - start:
            raise MaskMisaligned(
                f"reward vector has {len(reward)} entries for a {end - start}-token response; "
                "an env returning per-token rewards must align them to response_token_ids"
            )
        for i, value in enumerate(reward):
            self.scores[start + i] += float(value)
