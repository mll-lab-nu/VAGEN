"""Model-family adaptation facade and registry."""

from __future__ import annotations

from collections.abc import Callable

from vagen.models._common import *  # noqa: F401,F403
from vagen.models._common import ModelAdapter
from vagen.models._common import __all__ as _COMMON_EXPORTS
from vagen.models.glm import GLMModelAdapter
from vagen.models.internvl import InternVLModelAdapter
from vagen.models.qwen import QwenModelAdapter

MODEL_ADAPTERS: dict[str, type[ModelAdapter]] = {
    "glm": GLMModelAdapter,
    "internvl": InternVLModelAdapter,
    "qwen": QwenModelAdapter,
}


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
    if name == "auto":
        name = detect_model_adapter(tokenizer, processor)
    try:
        cls = MODEL_ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown model adapter {name!r}; choose from {sorted(MODEL_ADAPTERS)}"
        ) from exc
    return cls(tokenizer, processor, **kwargs)


def detect_model_adapter(tokenizer, processor=None) -> str:
    """Resolve a supported family from processor/tokenizer metadata."""
    holders = [holder for holder in (processor, tokenizer) if holder is not None]
    names = " ".join(type(holder).__name__.lower() for holder in holders)
    model_types = " ".join(
        str(getattr(getattr(holder, "config", None), "model_type", "")).lower()
        for holder in holders
    )
    identity = f"{names} {model_types}"
    if "internvl" in identity:
        return "internvl"
    if "glm" in identity:
        return "glm"
    if "qwen" in identity:
        return "qwen"
    raise ValueError(
        f"could not detect a model adapter from {names!r}; set trainer.model_adapter "
        f"explicitly to one of {sorted(MODEL_ADAPTERS)}"
    )


__all__ = [
    *_COMMON_EXPORTS,
    "MODEL_ADAPTERS",
    "GLMModelAdapter",
    "InternVLModelAdapter",
    "ModelAdapter",
    "QwenModelAdapter",
    "build_model_adapter",
    "detect_model_adapter",
    "register_model_adapter",
]
