"""Concat until the current conversation is full, then summarise and reseed."""

from __future__ import annotations

from vagen.rollout import EpisodeUnusable
from vagen.harness._common import BaseHarness, Msg, assistant, obs_to_message, user


class CompactionMakesNoProgress(EpisodeUnusable, RuntimeError):
    """Consecutive conversations closed after one turn, so compaction buys no room."""


SUMMARY_REQUEST = "Summarise the conversation so far. Keep every fact needed to continue."
SUMMARY_PREFIX = "Summary so far: "


def _with_summary(summary: str, observation: Msg) -> Msg:
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
    """Keep ordinary message history locally and make compaction explicit in the loop."""

    splits_episode_across_rows = True
    SUMMARY_REQUEST = SUMMARY_REQUEST
    SUMMARY_PREFIX = SUMMARY_PREFIX

    def __init__(
        self,
        budget: int | None = None,
        summary_budget: int | None = None,
        summary_request_len: int | None = None,
        response_len: int | None = None,
        floor: int = 1,
        **_cfg,
    ):
        self.budget = budget
        self.summary_budget = summary_budget
        if summary_request_len is None and summary_budget is not None:
            summary_request_len = len(SUMMARY_REQUEST.split()) * 3
        self.summary_request_len = summary_request_len or 0
        self.response_len = response_len
        self.floor = max(1, floor)
        self.summarised_conversations: set[str] = set()

    @property
    def reserve(self) -> int:
        return int(self.summary_budget or 0) + int(self.summary_request_len)

    async def run_episode(self, client, env) -> None:
        self.summarised_conversations.clear()
        observation, _info = await env.reset()
        observation = obs_to_message(observation)
        system = await env.system_prompt()
        messages = [system, observation]
        pending: Msg | None = None

        used = turn_cost = response_spent = 0
        opening = True
        turns_here = short_streak = 0

        while True:
            if pending is not None:
                pending_size = client.size([pending])
                region_limit = self.generation_limit(
                    self.response_len,
                    self.floor,
                    response_spent,
                    pending_size,
                    self.reserve,
                )
                should_compact = (
                    not opening
                    and (region_limit == 0 or bool(self.budget and used + turn_cost >= self.budget))
                )
                if should_compact:
                    short_streak = short_streak + 1 if turns_here <= 1 else 0
                    if short_streak >= 2:
                        raise CompactionMakesNoProgress(
                            f"{short_streak} conversations in a row closed after one turn. "
                            f"The current conversation uses {response_spent} response-region "
                            f"tokens with a {pending_size}-token observation pending, against "
                            f"max_response_length={self.response_len} and reserve={self.reserve}"
                            + (f", compact_budget={self.budget}" if self.budget else "")
                            + ". Raise the budget or shrink the per-turn response/observation."
                        )

                    summary_messages = [*messages, user(SUMMARY_REQUEST)]
                    summary = await client.create(
                        summary_messages,
                        **self.sampling(self.summary_budget),
                    )
                    if self.empty(summary):
                        env.truncate("empty_generation")
                        return
                    self.summarised_conversations.add(summary.conversation_id)
                    messages = [
                        system,
                        _with_summary(f"{SUMMARY_PREFIX}{summary.text}", pending),
                    ]
                    used = turn_cost = response_spent = 0
                    opening, turns_here = True, 0
                else:
                    messages.append(pending)
                pending = None

            # The reserve belongs to continuations: an opening has no prior conversation
            # to close, and the per-turn backend cap remains the tighter bound in normal
            # training configs.
            limit = self.generation_limit(
                self.response_len,
                self.floor,
                response_spent,
                0,
                0 if opening else self.reserve,
            )
            if limit == 0:
                env.truncate("no_room")
                return

            response = await client.create(messages, **self.sampling(limit))
            if self.empty(response):
                env.truncate("empty_generation")
                return

            grown = response.usage.total_tokens
            if not opening:
                turn_cost = grown - used
            used = grown
            response_spent = response.usage.response_tokens
            opening = False
            turns_here += 1

            messages.append(assistant(response.text))
            observation, _reward, terminated, truncated, _info = await env.step(response)
            if terminated or truncated:
                return
            pending = obs_to_message(observation)


__all__ = ["CompactHarness", "CompactionMakesNoProgress"]
