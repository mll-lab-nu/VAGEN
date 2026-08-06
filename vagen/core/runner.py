"""The episode loop.

One loop for training and for evaluation. Whether tokens are recorded is the client's
business, not this loop's, which is why the same code drives a verl rollout and a closed
chat API.

Compaction has no branch here: it is a call whose response the harness keeps instead of
forwarding, which is what ``accept`` returning ``None`` means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    obs, info = await env.reset(seed)
    harness.begin(await env.system_prompt(), obs_to_message(obs))

    result = EpisodeResult(info=dict(info or {}))
    for _ in range(max_turns):
        action = None
        while action is None:
            call = harness.next_call()
            kw = dict(send_kwargs)
            if call.sampling_params:
                kw["sampling_params"] = {**(kw.get("sampling_params") or {}), **call.sampling_params}
            response = await client.send(call.messages, call.conversation_id, **kw)
            # A budget-driven harness needs to know how large the conversation has
            # grown. It receives a number, not tokens -- the count is the client's.
            if hasattr(harness, "note_usage"):
                harness.note_usage(client.usage(response.conversation_id))

            # None: the harness kept this one for itself, e.g. a summary. The
            # environment must not act on it -- that would advance the episode by a
            # turn that never happened.
            action = harness.accept(response)

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

    await env.close()
    return result


def obs_to_message(obs) -> dict:
    """Environments speak in observations; harnesses speak in messages."""
    if isinstance(obs, dict) and "role" in obs:
        return obs
    text = obs.get("obs_str", "") if isinstance(obs, dict) else str(obs)
    return {"role": "user", "content": text}
