"""Shared wrapper for ending an episode after its configured turn budget.

``max_turns`` used to be enforced by ``run_episode``, as ``for _ in range(max_turns)``.
That put the episode's length in the runner: neither the environment whose budget it is,
nor the harness, but a third place that both had to agree with. An environment could not
be asked how long one of its episodes runs, and evaluation and training each passed the
number in separately.

It is the environment's property, so the environment enforces it. :class:`TurnLimit` wraps
one and ends the episode itself once the turns are spent.

★ Exhausting the turns is **truncation, not termination**, and the two are not
interchangeable here: a truncated episode should bootstrap from ``V`` of the state it
stopped in, a terminated one should not, because there is nothing after it. Reporting a
turn limit as termination teaches the value function that running out of time is worth
zero. The wrapper says which it is through ``info["truncated"]``; ``GymEnvAdapter`` reads
that into the ``truncated`` slot of the five-value contract.

``run_episode`` keeps a ``range(max_turns)`` of its own. It is a backstop now rather than
the rule -- an environment that never returns ``done`` should still not spin forever.
"""

from __future__ import annotations

import inspect
from typing import Any


class TurnLimit:
    """An environment that ends its own episode after ``max_turns`` steps."""

    def __init__(self, env: Any, max_turns: int):
        if int(max_turns) <= 0:
            raise ValueError(f"max_turns must be positive, got {max_turns!r}")
        self.env = env
        self.max_turns = int(max_turns)
        self.turns_taken = 0

    # ------------------------------------------------------------------ env facade
    async def reset(self, seed=None):
        self.turns_taken = 0
        return await self.env.reset(seed=seed)

    async def close(self):
        await self.env.close()

    async def system_prompt(self):
        return await self.env.system_prompt()

    def __getattr__(self, name):
        # Everything this wrapper has no opinion about belongs to the environment
        # underneath -- STATE_REWARD_SPEC, `success`, the renderer, whatever a caller
        # reaches for. Only the four methods above are intercepted.
        return getattr(self.env, name)

    async def step(self, action: str, response_token_ids=None, tokenizer=None):
        # Keep the token-aware contract visible to GymEnvAdapter even though this wrapper
        # sits outside StateRewardWrapper. A bare ``**kwargs`` signature is not enough:
        # the adapter deliberately forwards tokens only when it can prove the receiver
        # accepts them, otherwise every ordinary gym env would get unexpected arguments.
        kwargs = {}
        try:
            params = inspect.signature(self.env.step).parameters.values()
            accepts = any(
                p.name == "response_token_ids" or p.kind is inspect.Parameter.VAR_KEYWORD
                for p in params
            )
        except (TypeError, ValueError):
            accepts = False
        if accepts:
            kwargs = {"response_token_ids": response_token_ids, "tokenizer": tokenizer}

        obs, reward, done, info = await self.env.step(action, **kwargs)
        self.turns_taken += 1

        if not done and self.turns_taken >= self.max_turns:
            done = True
            info = {**(info or {}), "truncated": True}
        return obs, reward, done, info
