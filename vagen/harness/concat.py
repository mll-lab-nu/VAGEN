"""One conversation for the whole episode: one training row."""

from __future__ import annotations

from vagen.core.harness import BaseHarness, Call


class ConcatHarness(BaseHarness):
    """One conversation for the episode."""

    def next_call(self) -> Call:
        if self._conversation_id is None:
            return Call([self._system, *self._msgs], None)
        # Only what the environment said since the last call; the rest is already there.
        return Call([self._msgs[-1]], self._conversation_id)
