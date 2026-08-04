"""Context policy, in text.

A harness holds no tokenizer, no client and no env, and never sees a token, a mask or a
reward. Each turn it answers one question: does the next call continue the current
conversation, or start a new one, and seeded with what?

That single axis is the difference between the modes:

    concat      keep the id                     one row per episode
    no-concat   drop it every turn              one row per turn
    compact     drop it when a budget is hit    one row per compaction span

Because the harness only produces calls and consumes text, the same object drives a
closed API for evaluation: a conversation id is ``previous_response_id`` on OpenAI's
Responses API, a session on sglang, a cached prefix on vLLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

Msg = dict  # {"role": ..., "content": ...}


@dataclass
class Call:
    """What to send now. ``conversation_id=None`` starts a new conversation."""

    messages: list[Msg]
    conversation_id: Optional[str] = None


class BaseHarness(ABC):
    def __init__(self, **cfg):
        self.cfg = cfg
        self._system: Msg | None = None
        self._msgs: list[Msg] = []
        self._conversation_id: str | None = None

    # ------------------------------------------------------------------ input
    def begin(self, system: Msg, init_obs: Msg) -> None:
        self._system, self._msgs = system, [init_obs]

    def add_observation(self, obs: Msg) -> None:
        self._msgs.append(obs)

    # ----------------------------------------------------------------- output
    @abstractmethod
    def next_call(self) -> Call: ...

    def accept(self, response) -> Optional[str]:
        """Record the response; return the action text for the environment.

        ``None`` means the harness kept this one for itself. That is the whole of
        compaction: a summary is a model response to a user message like any other, and
        the harness simply does not forward it.
        """
        self._conversation_id = response.conversation_id
        self._msgs.append({"role": "assistant", "content": response.text})
        return response.text


class ConcatHarness(BaseHarness):
    """One conversation for the episode."""

    def next_call(self) -> Call:
        if self._conversation_id is None:
            return Call([self._system, *self._msgs], None)
        # Only what the environment said since the last call; the rest is already there.
        return Call([self._msgs[-1]], self._conversation_id)


class NoConcatHarness(BaseHarness):
    """A conversation per turn: the model sees the system prompt and the latest
    observation, never the history."""

    def next_call(self) -> Call:
        return Call([self._system, self._msgs[-1]], None)


class CompactHarness(BaseHarness):
    """Concat until a budget is hit, then summarise and start again from the summary.

    ``budget`` is counted in whatever unit ``note_usage`` is fed — tokens from the client
    when training, the provider's reported prompt size when evaluating. The two will not
    agree exactly, so log where it actually triggers.
    """

    SUMMARY_REQUEST = "Summarise the conversation so far. Keep every fact needed to continue."

    def __init__(self, budget: int, **cfg):
        super().__init__(**cfg)
        self.budget = budget
        self._used = 0
        self._awaiting_summary = False
        self._summary: Msg | None = None

    def note_usage(self, used: int) -> None:
        self._used = used

    def next_call(self) -> Call:
        if self._awaiting_summary:
            # Ask inside the current conversation, so the model can see what it is
            # summarising.
            return Call([{"role": "user", "content": self.SUMMARY_REQUEST}], self._conversation_id)

        if self._summary is not None:
            seed, self._summary = [self._system, self._summary, self._msgs[-1]], None
            return Call(seed, None)

        if self._conversation_id is not None and self._used >= self.budget:
            self._awaiting_summary = True
            return self.next_call()

        if self._conversation_id is None:
            return Call([self._system, *self._msgs], None)
        return Call([self._msgs[-1]], self._conversation_id)

    def accept(self, response) -> Optional[str]:
        if self._awaiting_summary:
            self._awaiting_summary = False
            self._summary = {"role": "user", "content": f"Summary so far: {response.text}"}
            self._conversation_id = None      # the next call opens a fresh conversation
            self._used = 0
            return None                        # consumed: the environment does not act
        return super().accept(response)
