"""One conversation for the whole episode: one training row."""

from __future__ import annotations

from vagen.harness._common import BaseHarness, assistant, obs_to_message


class ConcatHarness(BaseHarness):
    """Append every observation and response to one message list."""

    splits_episode_across_rows = False

    def __init__(self, response_len: int | None = None, floor: int = 1, **_cfg):
        self.response_len = response_len
        self.floor = max(1, floor)
        self.summarised_conversations: set[str] = set()

    async def run_episode(self, client, env) -> None:
        observation, _info = await env.reset()
        messages = [await env.system_prompt(), obs_to_message(observation)]
        spent = 0

        while True:
            pending = 0 if not spent else client.size([messages[-1]])
            limit = self.generation_limit(self.response_len, self.floor, spent, pending)
            if limit == 0:
                env.truncate("no_room")
                return
            response = await client.create(messages, **self.sampling(limit))
            if self.empty(response):
                env.truncate("empty_generation")
                return
            spent = response.usage.response_tokens
            messages.append(assistant(response.text))
            observation, _reward, terminated, truncated, _info = await env.step(response)
            if terminated or truncated:
                return
            messages.append(obs_to_message(observation))
