"""Concat until a budget is hit, then summarise and start again."""

from __future__ import annotations

from typing import Optional

from vagen.core.harness import BaseHarness, Call


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
