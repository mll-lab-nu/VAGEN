"""Context policies, by name.

A registry rather than an import in the trainer: which policy a run uses is a config
value, and adding one should not mean editing the agent loop. Mirrors how advantage
estimators are registered.
"""

from vagen.core.harness import BaseHarness, Call
from vagen.harness.compact import CompactHarness
from vagen.harness.concat import ConcatHarness
from vagen.harness.no_concat import NoConcatHarness

HARNESSES: dict[str, type[BaseHarness]] = {
    "concat": ConcatHarness,
    "no_concat": NoConcatHarness,
    "compact": CompactHarness,
}


def build_harness(name: str, **kwargs) -> BaseHarness:
    """Instantiate by name, failing with the available options rather than a KeyError."""
    if name not in HARNESSES:
        raise ValueError(f"unknown harness {name!r}; choose from {sorted(HARNESSES)}")
    return HARNESSES[name](**kwargs)


__all__ = ["BaseHarness", "Call", "ConcatHarness", "NoConcatHarness", "CompactHarness",
           "HARNESSES", "build_harness"]
