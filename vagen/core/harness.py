"""The context-policy contract.

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

Because a harness only produces calls and consumes text, the same object drives a closed
API for evaluation: a conversation id is ``previous_response_id`` on OpenAI's Responses
API, a session on sglang, a cached prefix on vLLM.
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
    def __init__(self, **cfg):
        self.cfg = cfg
        self._system: Msg | None = None
        self._msgs: list[Msg] = []
        self._conversation_id: str | None = None

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
