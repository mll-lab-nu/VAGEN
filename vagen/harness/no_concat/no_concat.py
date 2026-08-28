"""A fresh conversation for every environment turn."""

from __future__ import annotations

from vagen.harness._common import BaseHarness, obs_to_message


class NoConcatHarness(BaseHarness):
    """Show the model only the system prompt and latest observation."""

    splits_episode_across_rows = True

    def __init__(self, response_len: int | None = None, floor: int = 1, **_cfg):
        self.response_len = response_len
        self.floor = max(1, floor)
        self.summarised_conversations: set[str] = set()

    async def run_episode(self, client, env) -> None:
        observation, _info = await env.reset()
        observation = obs_to_message(observation)
        system = await env.system_prompt()

        while True:
            limit = self.generation_limit(self.response_len, self.floor, 0)
            if limit == 0:
                env.truncate("no_room")
                return
            response = await client.create([system, observation], **self.sampling(limit))
            if self.empty(response):
                env.truncate("empty_generation")
                return
            observation, _reward, terminated, truncated, _info = await env.step(response)
            if terminated or truncated:
                return
            observation = obs_to_message(observation)
