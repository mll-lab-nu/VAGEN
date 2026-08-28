"""Lifecycle wiring shared by training and evaluation.

The harness owns the episode loop. The runner only binds seed/status/reward archival and
guarantees environment cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vagen.rollout.scoring import ScoringSeam


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


async def run_episode(env, harness, client, *, seed=None, max_turns: int | None = None, **_kwargs):
    result = EpisodeResult()
    scored = ScoringSeam(env, client, result, seed=seed, max_turns=max_turns)
    try:
        await harness.run_episode(client, scored)
    finally:
        await scored.close()
    return result


__all__ = ["EpisodeResult", "run_episode"]
