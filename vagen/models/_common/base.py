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

    def processor_template_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Keep only kwargs referenced by the processor's Jinja template.

        Transformers 5.8 treats unknown template kwargs as processor-call kwargs and
        warns once per render. Shared defaults include ``enable_thinking`` for models
        that support it, while Qwen3-VL does not, so filtering here avoids thousands
        of warnings without changing templates that actually expose the option.
        """
        if self.processor is None or not kwargs:
            return dict(kwargs)
        templates = getattr(self.processor, "chat_template", None)
        if isinstance(templates, dict):
            templates = templates.values()
        else:
            templates = (templates,)
        source = "\n".join(template for template in templates if isinstance(template, str))
        return {name: value for name, value in kwargs.items() if name in source}

    @abstractmethod
    def placeholder_ids(self) -> tuple[set[int], set[int]]:
        """Image placeholder ids and their bracketing sentinel ids."""


__all__ = ["ModelAdapter"]
