"""Model-family rendering contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    """Render model-family chat edges independently of the inference backend."""

    def __init__(self, tokenizer, processor=None):
        self.tokenizer = tokenizer
        self.processor = processor

    @abstractmethod
    def render(self, messages: list[dict], *, opening: bool) -> tuple[list[int], list[Any]]:
        """Return token ids and image objects for one new message edge."""

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @abstractmethod
    def placeholder_ids(self) -> tuple[set[int], set[int]]:
        """Image placeholder ids and their bracketing sentinel ids."""


__all__ = ["ModelAdapter"]
