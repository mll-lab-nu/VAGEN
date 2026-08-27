"""Public declarations for selectable training algorithms."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

from vagen.algorithms._common.registry import (
    needs_critic,
    needs_value_mask,
    publishes_turn_id,
    requires_undiscounted,
    spans_rows,
    wants_turn_lumped_reward,
)


@dataclass(frozen=True)
class AlgorithmSpec:
    """One registered algorithm and the capabilities its trainer integration needs."""

    name: str
    implementation: Callable

    @property
    def needs_critic(self) -> bool:
        return needs_critic(self.name)

    @property
    def needs_value_mask(self) -> bool:
        return needs_value_mask(self.name)

    @property
    def spans_rows(self) -> bool:
        return spans_rows(self.name)

    @property
    def requires_undiscounted(self) -> bool:
        return requires_undiscounted(self.name)

    @property
    def publishes_turn_id(self) -> bool:
        return publishes_turn_id(self.name)

    @property
    def wants_turn_lumped_reward(self) -> bool:
        return wants_turn_lumped_reward(self.name)


ALGORITHMS: dict[str, AlgorithmSpec] = {}


def register_algorithm(name: str, implementation: Callable) -> AlgorithmSpec:
    spec = AlgorithmSpec(name=name, implementation=implementation)
    existing = ALGORITHMS.get(name)
    if existing is not None and existing.implementation is not implementation:
        raise ValueError(f"algorithm {name!r} is already registered")
    ALGORITHMS[name] = spec
    return spec


def resolve_algorithm(name: str) -> AlgorithmSpec:
    if name in ALGORITHMS:
        return ALGORITHMS[name]
    if "." not in name and ":" not in name:
        raise ValueError(f"unknown algorithm {name!r}; choose from {sorted(ALGORITHMS)}")
    module_name, _, attr = name.rpartition(":") if ":" in name else name.rpartition(".")
    try:
        spec = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"could not import algorithm {name!r}: {exc}") from exc
    if not isinstance(spec, AlgorithmSpec):
        raise TypeError(f"{name} resolved to {spec!r}, which is not an AlgorithmSpec")
    return spec


def registered_algorithms() -> tuple[str, ...]:
    return tuple(sorted(ALGORITHMS))


__all__ = [
    "ALGORITHMS",
    "AlgorithmSpec",
    "register_algorithm",
    "registered_algorithms",
    "resolve_algorithm",
]
