"""Concat until a budget is hit, then summarise and start again.

Closely related to CompactionRL (arXiv:2607.05378, Li et al. 2026), which trains task
execution and summary generation jointly under context compaction. Here too the summary is
written by the policy and carries gradient like any other turn -- it is not a free
preprocessing step -- so the compaction seam is a place the credit assignment has to be
right about. See ``TrajectoryView.seam``.
"""

from __future__ import annotations

from typing import Optional

from vagen.rollout import EpisodeUnusable
from vagen.harness._common import BaseHarness, Call, Msg


class CompactionMakesNoProgress(EpisodeUnusable, RuntimeError):
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

    #: One row per conversation, and an episode has as many as it compacts.
    splits_episode_across_rows = True

    SUMMARY_REQUEST = "Summarise the conversation so far. Keep every fact needed to continue."
    #: What the summary is wrapped in when it seeds the next conversation. Named because
    #: the accounting has to charge it: a relation written as ``S + k + E`` is short by
    #: whatever this costs, every time a conversation opens.
    SUMMARY_PREFIX = "Summary so far: "

    def __init__(self, budget: int | None = None, summary_budget: int | None = None,
                 summary_request_len: int | None = None, **cfg):
        super().__init__(**cfg)
        #: What closing a conversation costs. Both halves, because the summary *request*
        #: is a user message into the same conversation, so it lands in the same response
        #: region -- reserving only the summary overflows by the request every single
        #: time, deterministically, and `on_overflow` turns that into a dead batch.
        self.summary_budget = summary_budget
        if summary_request_len is None and summary_budget is not None:
            # Defaulting this to zero under-reserves by exactly the request, on every
            # compaction, deterministically -- and silently, because the caller that
            # forgot to measure it is the one that cannot tell. Only gym_loop knows the
            # real number; anyone constructing this directly gets a bound rather than a
            # blind spot.
            summary_request_len = len(self.SUMMARY_REQUEST.split()) * 3
        self.summary_request_len = summary_request_len or 0
        #: Optional second trigger, kept because the first one alone is not a lever.
        #: "Compact when the next turn does not fit" fills the whole response region, so
        #: on a model whose region is wide the mode never compacts at all and becomes
        #: concat -- and the only way to change that would be to narrow the region, which
        #: is also the training row width. This keeps the two separable.
        self.budget = budget
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
        self._room_resp = None
        self._room_obs = 0

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

    def _reserve(self) -> int:
        """Room for what closes this conversation: the summary *and* its request.

        Both halves. The request is a user message into the same conversation, so it
        lands in the same response region -- reserving only the summary overflows by the
        request every single time, deterministically.
        """
        return (self.summary_budget or 0) + self.summary_request_len

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
            # Bounded like every other call. It was the one exception, safe only because
            # VerlClient falls back to response_length_per_turn -- which the closed-API
            # client this module advertises does not have.
            limit = self.max_new_tokens()
            return Call([self._system, _with_summary(summary, self._msgs[-1])], None,
                        sampling_params={"max_new_tokens": limit} if limit is not None else None)

        if self._conversation_id is not None and self._should_compact():
            self._short_streak = self._short_streak + 1 if self._turns_here <= 1 else 0
            if self._short_streak >= 2:
                # One turn per conversation is not compaction, it is no_concat at twice
                # the price: two generations per environment step, and each summary
                # summarising a single turn. Arithmetic catches the configurations that
                # guarantee this (see harness/budget.py); reaching it anyway means the
                # data does -- an observation, or a turn, close to the whole budget.
                raise CompactionMakesNoProgress(
                    f"{self._short_streak} conversations in a row closed after a single "
                    f"turn. This one had {self._room_resp} tokens of response region and a "
                    f"{self._room_obs}-token observation pending, against "
                    f"max_response_length={self.response_len} less a reserve of "
                    f"{(self.summary_budget or 0) + self.summary_request_len} for the "
                    f"summary and its request"
                    + (f", and compact_budget={self.budget}" if self.budget else "")
                    + f". Compacting buys no turns that way and every environment step "
                    f"costs two generations. Raise max_response_length, lower "
                    f"compact_summary_budget or response_length_per_turn, or shrink the "
                    f"observation."
                )
            self._awaiting_summary = True
            return self.next_call()

        limit = self.max_new_tokens()
        params = {"max_new_tokens": limit} if limit is not None else None
        if self._conversation_id is None:
            return Call([self._system, *self._msgs], None, sampling_params=params)
        return Call([self._msgs[-1]], self._conversation_id, sampling_params=params)

    def exhausted(self) -> bool:
        """Never: making room is what this harness does."""
        return False

    def continues_conversation(self) -> bool:
        """False while a summary is pending: the next call opens on it."""
        return self._summary is None and super().continues_conversation()

    def _should_compact(self) -> bool:
        """Either the next turn does not fit, or the optional budget says close early.

        Two triggers, or-ed. The first is the correct one -- it is measured against the
        region the conversation actually has to fit in, using the real size of the
        observation about to be sent rather than an estimate of the last turn. The second
        exists because the first is not a lever: it only fires when the region is nearly
        full, so a wide region means no compaction at all.
        """
        left = self._left()
        if left is not None and left < self.floor:
            return True
        return bool(self.budget) and self._used + self.turn_cost >= self.budget

    def accept(self, response) -> Optional[str]:  # noqa: D102 - see base

        if self._awaiting_summary:
            self._awaiting_summary = False
            self._summary = f"{self.SUMMARY_PREFIX}{response.text}"
            # Recorded before the id is dropped: this conversation ended because the
            # context filled up, not because the environment stepped. See
            # `BaseHarness.summarised_conversations` for who reads it and why.
            if self._conversation_id is not None:
                self.summarised_conversations.add(self._conversation_id)
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
