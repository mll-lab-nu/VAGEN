"""Registry for selectable training metrics."""

from enum import Enum
from typing import Any

METRIC_REGISTRY: dict[str, Any] = {}


def register_metric(name_or_enum: str) -> Any:
    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in METRIC_REGISTRY and METRIC_REGISTRY[name] is not fn:
            raise ValueError(f"metric {name!r} is already registered")
        METRIC_REGISTRY[name] = fn
        return fn

    return decorator


__all__ = ["METRIC_REGISTRY", "register_metric"]
