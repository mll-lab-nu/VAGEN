"""One conversation for the whole episode: one training row."""

from __future__ import annotations

from vagen.harness._common import BaseHarness, Call


class ConcatHarness(BaseHarness):
    """One conversation for the episode."""

    #: The only one that does not. One conversation is one row, so a row-local estimator
    #: sees the whole trajectory and verl's own `gae`/`grpo` are safe here.
    splits_episode_across_rows = False

    def next_call(self) -> Call:
        limit = self.max_new_tokens()
        params = {"max_new_tokens": limit} if limit is not None else None
        if self._conversation_id is None:
            return Call([self._system, *self._msgs], None, sampling_params=params)
        # Only what the environment said since the last call; the rest is already there.
        return Call([self._msgs[-1]], self._conversation_id, sampling_params=params)
