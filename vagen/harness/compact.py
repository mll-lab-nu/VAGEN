"""Concat until a budget is hit, then summarise and start again."""

from __future__ import annotations

from typing import Optional

from vagen.core.harness import BaseHarness, Call, Msg


class CompactionMakesNoProgress(RuntimeError):
    """Conversation after conversation hit the budget after a single turn.

    Not a slow run: the episode still finishes, every row is well-formed, and the only
    trace is a rollout that cost twice what it should and a summary per environment step.
    Nothing downstream distinguishes it from compaction working.

    Raised on a repeat, not on the first one. A single conversation cut short by an
    unusually large observation is data, and killing a run over it would be wrong; two in
    a row cannot be, because the second opened on a summary written under this budget --
    if that still leaves no room, nothing later will.
    """


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

    The trigger fires on "another turn would not fit" rather than "we are already over",
    against a turn cost it measures as the episode goes. ``summary_budget`` bounds the
    summary itself. Without one it generated against the
    turn budget, which at ``budget=400`` and a turn budget of 8000 let the summary be
    twenty times the thing it was compressing into -- fine for as long as the model
    happened to write short ones. See ``vagen/harness/budget.py`` for the arithmetic
    these two have to satisfy together.
    """

    SUMMARY_REQUEST = "Summarise the conversation so far. Keep every fact needed to continue."
    #: What the summary is wrapped in when it seeds the next conversation. Named because
    #: the accounting has to charge it: a relation written as ``S + k + E`` is short by
    #: whatever this costs, every time a conversation opens.
    SUMMARY_PREFIX = "Summary so far: "

    def __init__(self, budget: int, summary_budget: int | None = None, **cfg):
        super().__init__(**cfg)
        self.budget = budget
        self.summary_budget = summary_budget
        self._reset_episode_state()

        # What one more turn will cost, measured. The trigger fires before the turn it is
        # deciding about, so without this a conversation waved through at budget-1 still
        # grows by a whole turn before anyone looks again and the budget bounds nothing.
        #
        # Measured and not the configured ceiling: on Sokoban the ceiling is 512 and a
        # real turn is about 80, so charging the ceiling fires after the first turn of
        # every conversation and compaction becomes no_concat with a summary attached.
        #
        # The last continuation, not the largest. A maximum stops predicting and starts
        # remembering: one response at the ceiling -- legal, it is exactly what
        # response_length_per_turn permits -- would set the estimate for the rest of the
        # episode, cut every later conversation off after a single turn, and kill the run
        # claiming a budget of 1300 could not hold a turn costing 825. Being wrong low
        # costs one turn of overshoot, which the ceilings bound anyway; being wrong high
        # has no bound at all. Zero until there is a continuation to measure -- the first
        # turn of the first conversation cannot be predicted from nothing.

    def _reset_episode_state(self) -> None:
        self._used = 0
        self._awaiting_summary = False
        self._summary: str | None = None
        self._turns_here = 0
        self._short_streak = 0
        self.turn_cost = 0        # see the note above __init__'s reset

    def begin(self, system, init_obs) -> None:
        """Start an episode, including the parts of the state only this class has.

        The base class resets the conversation; everything counted here is per-episode
        too. Left behind, a stale ``turn_cost`` compacts the next episode early and a
        stale ``_turns_here`` makes its first conversation look one turn longer than it
        is -- which reads as progress and switches off the no-progress guard exactly
        where it is needed. Not reachable while the loop builds a harness per episode,
        but the base class says not to depend on that.
        """
        self._reset_episode_state()
        super().begin(system, init_obs)

    def note_usage(self, used: int) -> None:
        # A continuation's growth is exactly one turn: the observation that was sent with
        # it plus the response. An opening call also carries the system prompt, and a
        # summary call carries the request and the summary, so neither is a turn and
        # neither may inform the estimate.
        if self._conversation_id is not None and not self._awaiting_summary:
            self.turn_cost = used - self._used
        self._used = used

    def next_call(self) -> Call:
        if self._awaiting_summary:
            # Ask inside the current conversation, so the model can see what it is
            # summarising.
            return Call([{"role": "user", "content": self.SUMMARY_REQUEST}], self._conversation_id,
                        sampling_params={"max_new_tokens": self.summary_budget} if self.summary_budget else None)

        if self._summary is not None:
            summary, self._summary = self._summary, None
            return Call([self._system, _with_summary(summary, self._msgs[-1])], None)

        if self._conversation_id is not None and self._used + self.turn_cost >= self.budget:
            self._short_streak = self._short_streak + 1 if self._turns_here <= 1 else 0
            if self._short_streak >= 2:
                # One turn per conversation is not compaction, it is no_concat at twice
                # the price: two generations per environment step, and each summary
                # summarising a single turn. Arithmetic catches the configurations that
                # guarantee this (see harness/budget.py); reaching it anyway means the
                # data does -- an observation, or a turn, close to the whole budget.
                raise CompactionMakesNoProgress(
                    f"{self._short_streak} conversations in a row reached the budget after a "
                    f"single turn -- this one at {self._used} tokens against "
                    f"compact_budget={self.budget} -- so compacting buys no turns and every "
                    f"environment step is costing two generations. The second of them opened "
                    f"on a summary written under this budget, so it is the budget and not the "
                    f"data: raise trainer.compact_budget above what one turn costs "
                    f"({self._used} tokens here), lower trainer.compact_summary_budget, or "
                    f"shrink the observation."
                )
            self._awaiting_summary = True
            return self.next_call()

        if self._conversation_id is None:
            return Call([self._system, *self._msgs], None)
        return Call([self._msgs[-1]], self._conversation_id)

    def accept(self, response) -> Optional[str]:  # noqa: D102 - see base

        if self._awaiting_summary:
            self._awaiting_summary = False
            self._summary = f"{self.SUMMARY_PREFIX}{response.text}"
            self._conversation_id = None      # the next call opens a fresh conversation
            self._used = 0
            self._turns_here = 0
            # The estimate is about this conversation, and there is no longer one. Carried
            # across, a single expensive turn keeps predicting for conversations it was
            # never part of: it cannot be corrected, because the first turn of a new
            # conversation is an opening call and openings may not inform the estimate.
            # One 512-token response was enough to cut the next two conversations short
            # and kill the run.
            self.turn_cost = 0
            return None                        # consumed: the environment does not act
        self._turns_here += 1
        return super().accept(response)
