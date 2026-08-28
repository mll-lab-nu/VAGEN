"""Bind environment rewards to the exact model call that earned them."""

from __future__ import annotations

from numbers import Real


class ScoringSeam:
    """Environment facade used by harnesses.

    Harness code can read rewards, but cannot forget to archive one: ``step`` records it
    against ``Response.call_id`` before returning. It also owns episode status so the
    runner remains lifecycle wiring rather than an interaction loop.
    """

    def __init__(self, env, client, result, *, seed=None, max_turns: int | None = None):
        self.env = env
        self.client = client
        self.result = result
        self.seed = seed
        self.max_turns = int(max_turns) if max_turns else None

    def __getattr__(self, name):
        return getattr(self.env, name)

    async def reset(self):
        observation, info = await self.env.reset(self.seed)
        self.result.info.update(info or {})
        return observation, info

    async def system_prompt(self):
        return await self.env.system_prompt()

    async def step(self, response):
        observation, reward, terminated, truncated, info = await self.env.step(response)
        self.client.reward_call(response.call_id, reward)

        value = float(reward) if isinstance(reward, Real) else float(sum(reward))
        self.result.rewards.append(value)
        self.result.turns += 1
        self.result.info.update(info or {})

        if self.max_turns and self.result.turns >= self.max_turns and not terminated:
            truncated = True
        self.result.terminated = bool(terminated)
        self.result.truncated = bool(truncated)
        return observation, reward, bool(terminated), bool(truncated), info

    def truncate(self, reason: str) -> None:
        self.result.truncated = True
        self.result.info["rollout_stop_reason"] = reason

    async def close(self):
        await self.env.close()


__all__ = ["ScoringSeam"]
