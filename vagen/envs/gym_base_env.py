# env_base.py
from __future__ import annotations
from typing import Any, Dict, Tuple
from abc import ABC, abstractmethod


class GymBaseEnv(ABC):
    """
    Abstract async environment API.
    The handler does not assume any obs/data schema beyond what you return.

    Contract:
      - reset(seed) -> (obs, info)
      - step(action_str) -> (obs, reward, done, info)
    """

    def __init__(self, env_config: Dict[str, Any]):
        self.config = env_config

    @abstractmethod
    async def close(self) -> None:
        """Async teardown."""
        raise NotImplementedError

    @abstractmethod
    async def reset(self, seed: int):
        raise NotImplementedError

    @abstractmethod
    async def step(self, action_str: str):
        raise NotImplementedError
    
    @abstractmethod
    async def system_prompt(self):
        raise NotImplementedError



