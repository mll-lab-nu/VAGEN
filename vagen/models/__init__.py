"""Model-family adaptation facade and registry."""

from __future__ import annotations

from collections.abc import Callable

from vagen.models._common import *  # noqa: F401,F403
from vagen.models._common import ModelAdapter
from vagen.models._common import __all__ as _COMMON_EXPORTS
from vagen.models.qwen import QwenModelAdapter

MODEL_ADAPTERS: dict[str, type[ModelAdapter]] = {"qwen": QwenModelAdapter}


def register_model_adapter(name: str) -> Callable[[type[ModelAdapter]], type[ModelAdapter]]:
    def decorator(cls: type[ModelAdapter]) -> type[ModelAdapter]:
        if not issubclass(cls, ModelAdapter):
            raise TypeError(f"{cls!r} is not a ModelAdapter")
        existing = MODEL_ADAPTERS.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"model adapter {name!r} is already registered")
        MODEL_ADAPTERS[name] = cls
        return cls
    return decorator


def build_model_adapter(name: str, tokenizer, processor=None, **kwargs) -> ModelAdapter:
    try:
        cls = MODEL_ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown model adapter {name!r}; choose from {sorted(MODEL_ADAPTERS)}"
        ) from exc
    return cls(tokenizer, processor, **kwargs)


__all__ = [
    *_COMMON_EXPORTS,
    "MODEL_ADAPTERS",
    "ModelAdapter",
    "QwenModelAdapter",
    "build_model_adapter",
    "register_model_adapter",
]
