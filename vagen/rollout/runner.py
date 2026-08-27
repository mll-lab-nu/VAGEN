"""The invariant episode rollout loop.

One loop for training and for evaluation. Whether tokens are recorded is the client's
business, not this loop's, which is why the same code drives a verl rollout and a closed
chat API.

Compaction has no branch here: it is a call whose response the harness keeps instead of
forwarding, which is what ``accept`` returning ``None`` means.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: How many model calls one environment step may take. Compaction spends a second one on
#: the summary; nothing legitimately needs more than a handful.
MAX_CALLS_PER_TURN = 8


@dataclass
class EpisodeResult:
    rewards: list[float] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    turns: int = 0
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)


async def run_episode(env, harness, client, *, seed=None, max_turns: int = 10, **send_kwargs) -> EpisodeResult:
    # Inside the try: a reset that raises left the environment open, and a remote env
    # client holds a server-side session and an HTTP client that only close() releases --
    # one leaked per row of the batch.
    try:
        obs, info = await env.reset(seed)
        harness.begin(await env.system_prompt(), obs_to_message(obs))

        result = EpisodeResult(info=dict(info or {}))
        for _ in range(max_turns):
            action = None
            # A turn may legitimately take more than one call -- compaction spends one on
            # the summary -- but only a bounded number. `accept` returning None is the
            # only way out of this loop, and a backend that returns something the harness
            # keeps every time spins here: measured at 100,001 generations, and 100,001
            # conversations under no_concat, for a client whose `text` came back None.
            # Closed APIs return that for a refusal, and nothing in the type enforces str.
            for _attempt in range(MAX_CALLS_PER_TURN):
                # What the harness is about to decide about, measured rather than
                # estimated: how much of the response region this conversation has spent,
                # and how big the observation waiting to go into it is. Both come from the
                # client because it is the only layer that knows a token. Measuring is
                # side-effect free -- that is what `render` is separable from `encode`
                # for; asking used to ship every picture twice.
                if hasattr(harness, "note_room") and harness.pending_observation() is not None:
                    # The room belongs to the conversation the *next* call will land in,
                    # which is not always the one just used -- no_concat opens a new one
                    # every turn, and compaction opens one on a pending summary.
                    cont = harness.continues_conversation()
                    cid = getattr(harness, "_conversation_id", None)
                    spent = client.response_len(cid) if (cont and cid) else 0
                    obs = client.measure([harness.pending_observation()]) if cont else 0
                    harness.note_room(spent, obs)
                    if harness.exhausted():
                        # No room for another turn, and this policy cannot make room --
                        # compaction can, and says so by never being exhausted. The
                        # episode stops here rather than generating into a space too
                        # small to hold an action. Truncated, not terminated: the
                        # environment had more to give.
                        logger.info("no room left for another turn at turn %d; "
                                    "ending the episode", result.turns)
                        result.truncated = True
                        return result

                call = harness.next_call()
                kw = dict(send_kwargs)
                if call.sampling_params:
                    kw["sampling_params"] = {**(kw.get("sampling_params") or {}), **call.sampling_params}
                response = await client.send(call.messages, call.conversation_id, **kw)
                # A budget-driven harness needs to know how large the conversation has
                # grown. It receives a number, not tokens -- the count is the client's.
                if hasattr(harness, "note_usage"):
                    harness.note_usage(client.usage(response.conversation_id))

                if not response.token_ids and response.token_ids is not None:
                    # Still empty after the client's retries. That is no longer an
                    # interruption, it is an engine that has stopped answering, and the
                    # episode cannot continue: `accept` would return "" -- which is not
                    # None -- so the loop would take it for an action and step the
                    # environment on nothing. Measured before this: three env steps on
                    # '' and zero trainable rows, the whole episode gone from the batch
                    # while the environment had moved three times.
                    #
                    # Ending here rather than raising. The turns already collected are
                    # real and worth training on, and one engine hiccup should not take
                    # the batch down with it.
                    logger.warning("generation still empty after retries; ending the "
                                   "episode at turn %d rather than stepping the "
                                   "environment on an empty action", result.turns)
                    result.truncated = True
                    return result

                # None: the harness kept this one for itself, e.g. a summary. The
                # environment must not act on it -- that would advance the episode by a
                # turn that never happened.
                action = harness.accept(response)
                if action is not None:
                    break
            else:
                raise RuntimeError(
                    f"turn {result.turns} took {MAX_CALLS_PER_TURN} model calls without "
                    f"producing an action. The harness is keeping every response for "
                    f"itself, which one of them was supposed to stop doing."
                )

            obs, reward, terminated, truncated, step_info = await env.step(
                action, response_token_ids=response.token_ids, tokenizer=client.tokenizer
            )
            client.reward(response.conversation_id, reward)

            result.rewards.append(reward if isinstance(reward, (int, float)) else sum(reward))
            result.turns += 1
            result.info.update(step_info or {})

            if terminated or truncated:
                result.terminated, result.truncated = bool(terminated), bool(truncated)
                break

            harness.add_observation(obs_to_message(obs))
        else:
            # Ran out of turns rather than reaching a terminal state. The distinction
            # matters to the value function: a truncated episode should bootstrap, a
            # terminated one should not.
            result.truncated = True
    finally:
        # Every guard on this path -- the context ceilings, CompactionMakesNoProgress,
        # cap_token_ids -- raises mid-episode, and an environment left open is held for
        # the rest of the batch. A rollout that dies on the first episode used to take
        # its simulator down with it.
        await env.close()
    return result


def obs_to_message(obs) -> dict:
    """Environments speak in observations; harnesses speak in messages."""
    if isinstance(obs, dict) and "role" in obs:
        return obs
    text = obs.get("obs_str", "") if isinstance(obs, dict) else str(obs)
    return {"role": "user", "content": text}
