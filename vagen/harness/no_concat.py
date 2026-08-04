"""A conversation per turn: one training row per turn."""

from __future__ import annotations

from vagen.core.harness import BaseHarness, Call


class NoConcatHarness(BaseHarness):
    """A conversation per turn: the model sees the system prompt and the latest
    observation, never the history."""

    def next_call(self) -> Call:
        return Call([self._system, self._msgs[-1]], None)
