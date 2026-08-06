"""Concat until a budget is hit, then summarise and start again."""

from __future__ import annotations

from typing import Optional

from vagen.core.harness import BaseHarness, Call, Msg


def _with_summary(summary: str, observation: Msg) -> Msg:
    """The summary and the observation as one user message.

    Not two messages in a row. What opens a new conversation is a single user turn --
    here is the story so far, and here is where you are now -- so the exchange stays
    system / user / assistant. Chat templates are not obliged to handle two consecutive
    user messages the same way, and an episode should not depend on which way they do.

    The observation, not an initial observation: the environment resets once per episode,
    so every conversation after the first opens on whatever the last ``step`` returned.
    """
    # The blank line is part of the summary text, not something the caller adds. A parts
    # list is concatenated by the chat template with nothing between the parts, so
    # separating them only in the string branch produced
    # "...align it with the target.After your answer, the extracted valid action is..."
    # -- the summary running straight into the observation with no boundary at all.
    body = f"{summary}\n\n"
    content = observation.get("content")
    if isinstance(content, str):
        merged = f"{body}{content}"
    elif isinstance(content, list):
        merged = [{"type": "text", "text": body}, *content]
    else:
        merged = summary
    return {**observation, "role": "user", "content": merged}


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
        self._summary: str | None = None

    def note_usage(self, used: int) -> None:
        self._used = used

    def next_call(self) -> Call:
        if self._awaiting_summary:
            # Ask inside the current conversation, so the model can see what it is
            # summarising.
            return Call([{"role": "user", "content": self.SUMMARY_REQUEST}], self._conversation_id)

        if self._summary is not None:
            summary, self._summary = self._summary, None
            return Call([self._system, _with_summary(summary, self._msgs[-1])], None)

        if self._conversation_id is not None and self._used >= self.budget:
            self._awaiting_summary = True
            return self.next_call()

        if self._conversation_id is None:
            return Call([self._system, *self._msgs], None)
        return Call([self._msgs[-1]], self._conversation_id)

    def accept(self, response) -> Optional[str]:  # noqa: D102 - see base

        if self._awaiting_summary:
            self._awaiting_summary = False
            self._summary = f"Summary so far: {response.text}"
            self._conversation_id = None      # the next call opens a fresh conversation
            self._used = 0
            return None                        # consumed: the environment does not act
        return super().accept(response)
