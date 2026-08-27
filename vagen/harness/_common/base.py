"""The shared context-policy contract.

A harness holds no tokenizer, no client and no env, and never sees a token, a mask or a
reward. Each turn it answers one question: does the next call continue the current
conversation, or start a new one, and seeded with what?

That single axis is what separates the modes, so they are variations on this class
rather than different mechanisms -- see ``vagen/harness/`` for the implementations.
An **episode** is one agent/environment interaction; a **conversation** is one continuous
exchange with the model; a **turn** is one model call. The policy is which shape you get:

    concat      1 conversation per episode,  many turns in it
    no_concat   many conversations,          1 turn each
    compact     several conversations,       many turns each

Because a harness only produces calls and consumes text, the same object drives a closed API
for evaluation. A conversation id is bookkeeping here rather than a server-side handle --
every shipped backend re-sends the messages -- but nothing in this class assumes that, so a
backend that does support continuation (``previous_response_id`` on OpenAI's Responses API,
a session on sglang, a cached prefix on vLLM) can be wired up without touching the policy.
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
    #: Overrides for this call only, merged over the episode's. A harness needs this
    #: when one of its calls is not like the others: compaction's summary has to be
    #: bounded by its own budget rather than by the turn budget, or it can be longer
    #: than the conversation it is replacing.
    sampling_params: Optional[dict] = None


class BaseHarness(ABC):
    """A context policy, plus the room its conversations have to work in.

    The room is shared by all three because the question is: how much of
    ``max_response_length`` has this conversation spent, and does the next turn still
    fit? What differs is only the answer when it does not -- compaction summarises,
    the other two stop.
    """

    #: Whether one episode can end up in more than one row. Declared here so the trainer
    #: can ask any harness instead of keeping a list of the ones it knows.
    #:
    #: True by default, because the wrong answer is not symmetric: under a harness that
    #: splits, a row-local estimator scores a truncated trajectory and nothing fails --
    #: it trains, the curves look ordinary, and the credit assignment is just absent.
    splits_episode_across_rows: bool = True

    def __init__(self, response_len: int | None = None, floor: int = 1, **cfg):
        self.cfg = cfg
        self._system: Msg | None = None
        self._msgs: list[Msg] = []
        self._conversation_id: str | None = None
        #: The response region a conversation has to fit in. ``None`` disables the
        #: accounting, which is what an evaluation against a closed API wants: there is
        #: no row to fit.
        self.response_len = response_len
        #: The smallest generation worth making. Below it a turn produces half an
        #: ``<answer>`` and the environment parses it as an action anyway.
        self.floor = max(1, floor)
        self._room_resp: int | None = None
        self._room_obs = 0
        #: Conversations this harness closed by asking the model to summarise, rather
        #: than because the environment stepped. Empty for every harness but
        #: ``CompactHarness``; declared here so the estimators can read it off any
        #: harness without asking which one they have.
        #:
        #: ★ Why an advantage estimator cares. A summary is a model emission and an
        #: action, so it ends a "turn" in the token stream -- but no environment step
        #: happened, and the next conversation's first action sees the same world state
        #: the summary did. An estimator that discounts turn-to-turn therefore charges a
        #: compaction seam the same as a real transition, and since compaction frequency
        #: is set by how verbosely the policy writes, the effective horizon becomes a
        #: function of the policy's own output length -- which it changes as it trains.
        self.summarised_conversations: set[str] = set()

    # ------------------------------------------------------------------ input
    def begin(self, system: Msg, init_obs: Msg) -> None:
        """Start an episode. Resets the conversation too.

        Leaving ``_conversation_id`` set would make the first call of a new episode a
        continuation of the last one's conversation -- appending a fresh initial
        observation to a finished episode's context. Not reachable today, since the loop
        builds a harness per episode, but the class should not depend on that.
        """
        self._system, self._msgs = system, [init_obs]
        self._conversation_id = None
        # The room too. A reused harness otherwise reports the *previous* episode's
        # spend, which can read as exhausted before this episode has generated anything.
        # CompactHarness already did this for its own state; the base class was the one
        # depending on the loop building a harness per episode.
        self._room_resp, self._room_obs = None, 0
        # Per-episode, like everything else reset here. Carried over, a reused harness
        # would mark the *previous* episode's conversation ids as seams, and conversation
        # ids are per-episode ordinals -- so the marks land on unrelated conversations.
        self.summarised_conversations = set()

    def add_observation(self, obs: Msg) -> None:
        self._msgs.append(obs)

    # ------------------------------------------------------------------- room
    def note_room(self, response_len: int, obs_len: int) -> None:
        """What the client measured: the region spent, and the observation pending.

        Measured rather than estimated, and measured *before* anything is decided about
        the observation -- which is why rendering has to be separable from recording.
        On a conversation's opening call the observation counts zero: it and the system
        prompt become the *prompt* region, and this budget is the response region's.
        """
        self._room_resp, self._room_obs = response_len, obs_len

    def continues_conversation(self) -> bool:
        """Whether the next call appends to the conversation already open.

        The room a call has is the room of the conversation it will land in, and that is
        not always the one just used: no_concat opens a new one every turn, so measuring
        the old one reports a region that is about to be discarded. Compaction overrides
        this when it is about to reseed.
        """
        return self._conversation_id is not None

    def _reserve(self) -> int:
        """What must stay free for whatever closes the conversation. Nothing, by default."""
        return 0

    def _left(self) -> int | None:
        """Room for the next generation. ``None`` when the accounting is off."""
        if self.response_len is None or self._room_resp is None:
            return None
        return self.response_len - self._room_resp - self._room_obs - self._reserve()

    def max_new_tokens(self) -> int | None:
        """What the next generation may produce, so it cannot overrun the region."""
        left = self._left()
        return None if left is None else max(self.floor, left)

    def exhausted(self) -> bool:
        """No room for another turn, and no way to make room.

        Compaction overrides this: making room is what it does. For the other two the
        conversation -- and under concat the episode -- ends here.
        """
        left = self._left()
        return left is not None and left < self.floor

    def pending_observation(self) -> Optional[Msg]:
        """The message about to be sent, so a caller can measure it first.

        The harness holds the observation from the moment the environment returns it
        until it decides what to do with it -- that gap is where a budget decision has to
        be made, and making it needs the size.
        """
        return self._msgs[-1] if self._msgs else None

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
