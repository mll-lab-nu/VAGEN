"""The harness contract shared by training and evaluation.

A harness owns the agent/environment interaction loop. It builds ordinary message
lists, asks ``client.create(messages)`` for a response, and steps the environment with
that response. Token rendering, conversation routing, reward archival, and VERL row
assembly stay outside the harness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

Msg = dict


class BaseHarness(ABC):
    """One reusable episode policy."""

    splits_episode_across_rows: bool = True

    @abstractmethod
    async def run_episode(self, client, env) -> None:
        """Run until the environment reports termination or truncation."""

    @staticmethod
    def generation_limit(
        response_len: int | None,
        floor: int,
        spent: int,
        pending: int = 0,
        reserve: int = 0,
    ) -> int | None:
        """Room left in VERL's response region for one generation."""
        if response_len is None:
            return None
        left = int(response_len) - int(spent) - int(pending) - int(reserve)
        return left if left >= max(1, int(floor)) else 0

    @staticmethod
    def sampling(limit: int | None) -> dict:
        return {} if limit is None else {"sampling_params": {"max_new_tokens": limit}}

    @staticmethod
    def empty(response) -> bool:
        """Whether a backend exhausted its retries without producing an action."""
        return response.token_ids is not None and not response.token_ids


def assistant(text: str) -> Msg:
    return {"role": "assistant", "content": text}


def user(text: str) -> Msg:
    return {"role": "user", "content": text}


def obs_to_message(obs) -> Msg:
    """Environments speak in observations; harnesses speak in messages."""
    if isinstance(obs, dict) and "role" in obs:
        return obs
    text = obs.get("obs_str", "") if isinstance(obs, dict) else str(obs)
    return {"role": "user", "content": text}


__all__ = ["BaseHarness", "Msg", "assistant", "obs_to_message", "user"]
