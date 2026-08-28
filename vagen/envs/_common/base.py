"""The shared environment contract.

Until now this existed only at the runner's call site and in prose, which meant an
environment could satisfy it by accident and fail in the one case nobody wrote down.

An environment scores actions and decides when an episode is over. The framework-facing
adapter receives a response object; legacy gym implementations may continue to consume
its text plus optional token metadata behind that adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

Obs = dict          # {"obs_str": ..., "multi_modal_input": {...}}
Reward = Union[float, list[float]]


class BaseEnv(ABC):
    """What ``run_episode`` requires of an environment."""

    @abstractmethod
    async def reset(self, seed: Optional[int] = None) -> tuple[Obs, dict]:
        """Start an episode and return the first observation."""

    @abstractmethod
    async def system_prompt(self) -> Obs:
        """The standing instructions, as an observation."""

    @abstractmethod
    async def step(
        self,
        response: Any,
    ) -> tuple[Obs, Reward, bool, bool, dict]:
        """Apply one action.

        Returns ``(obs, reward, terminated, truncated, info)``.

        ★ ``terminated`` and ``truncated`` are distinct. Terminated means the episode
        reached an end state and there is nothing to bootstrap from; truncated means it
        was cut short and the value function should still bootstrap. Reporting a turn
        limit as termination biases every episode that hits the cap.

        ``reward`` may be a scalar, which lands on the last token the model produced, or
        a vector aligned to ``response.token_ids``, which lets an environment say *which
        part* of a response earned what. Its length is checked against the response.

        ★ The response carries token ids and the inference client's tokenizer so an
        environment that scores per token has what it needs. Decoding the ids is safe; re-encoding
        the text is not. BPE is not compositional, so a re-encoded response can split
        differently from how it was generated, and a reward vector built on that is
        misaligned against the sequence being trained -- silently, since both are
        well-formed and the loss stays finite.
        """

    async def close(self) -> None:
        """Release anything the episode held. Optional."""
