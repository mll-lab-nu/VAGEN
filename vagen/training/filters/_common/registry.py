"""Registry for selectable training-batch filters."""

from enum import Enum
from typing import Any

FILTER_REGISTRY: dict[str, Any] = {}


def register_filter(name_or_enum: str) -> Any:
    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in FILTER_REGISTRY and FILTER_REGISTRY[name] is not fn:
            raise ValueError(f"filter {name!r} is already registered")
        FILTER_REGISTRY[name] = fn
        return fn

    return decorator


__all__ = ["FILTER_REGISTRY", "register_filter"]
