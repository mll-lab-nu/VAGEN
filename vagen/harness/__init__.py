"""Context policies, by name.

A registry rather than an import in the trainer: which policy a run uses is a config
value, and adding one should not mean editing the agent loop. Mirrors how advantage
estimators are registered.

Two ways to reach a policy that is not one of the three built in, so that a custom
``BaseHarness`` runs in training and in evaluation without either being edited:

    @register_harness("mine")            # then `harness: mine`
    class MyHarness(BaseHarness): ...

    harness: mypkg.harnesses:MyHarness   # an import path, with nothing to register

The import path is there because evaluation is often where a new policy is tried first,
and an eval config is a yaml rather than a package -- there is nowhere to put a decorator
that would have run by then. For training, ``actor_rollout_ref.model.external_lib`` is the
other way in: verl imports it inside every worker, each of which builds its own registry.
"""

from __future__ import annotations

import importlib
from typing import Callable

from vagen.harness._common import BaseHarness
from vagen.harness.compact import CompactHarness
from vagen.harness.concat import ConcatHarness
from vagen.harness.no_concat import NoConcatHarness

HARNESSES: dict[str, type[BaseHarness]] = {
    "concat": ConcatHarness,
    "no_concat": NoConcatHarness,
    "compact": CompactHarness,
}


def register_harness(name: str) -> Callable[[type], type]:
    """Register a ``BaseHarness`` subclass under ``name``.

    Refuses to overwrite a *different* class already holding the name: a silent rebinding
    means a run reports the policy it was configured with and executes another one.
    Re-registering the same class is fine -- a module is legitimately imported more than
    once, since verl builds a registry per worker process.
    """
    def decorator(cls: type) -> type:
        _require_harness(cls, name)
        existing = HARNESSES.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"harness {name!r} is already registered to {existing.__qualname__}; "
                f"pick another name rather than shadowing it."
            )
        HARNESSES[name] = cls
        return cls
    return decorator


def resolve_harness(name: str) -> type[BaseHarness]:
    """The class for ``name``: a registered name, or an import path.

    An import path is ``module:Class`` or ``module.Class``. It is checked against
    ``BaseHarness`` before it is returned, so a path naming some other class fails here
    rather than at the first ``run_episode`` with an AttributeError.
    """
    if name in HARNESSES:
        return HARNESSES[name]
    if "." not in name and ":" not in name:
        raise ValueError(
            f"unknown harness {name!r}; choose from {sorted(HARNESSES)}, or give an "
            f"import path like 'mypkg.harnesses:MyHarness'."
        )
    module_name, _, attr = name.rpartition(":") if ":" in name else name.rpartition(".")
    try:
        cls = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"could not import harness {name!r}: {exc}") from exc
    _require_harness(cls, name)
    return cls


def _require_harness(cls, name: str) -> None:
    if not (isinstance(cls, type) and issubclass(cls, BaseHarness)):
        raise TypeError(
            f"{name} resolved to {cls!r}, which does not subclass BaseHarness -- so the "
            "harness run_episode(client, env) contract is not guaranteed."
        )


def budget_mode(name: str) -> str:
    """Which built-in's token accounting applies to ``name``.

    ``vagen/harness/budget.py`` reasons about three shapes, and it takes a string. A
    custom policy has a name it has never heard of, so without this a subclass of
    ``CompactHarness`` gets no compaction budget wired and a concat-like one loses its
    observation ceiling -- both silently, both only visible as a run that behaves like a
    different mode than its name.
    """
    cls = resolve_harness(name)
    if issubclass(cls, CompactHarness):
        return "compact"
    if issubclass(cls, NoConcatHarness):
        return "no_concat"
    return "concat"


def build_harness(name: str, **kwargs) -> BaseHarness:
    """Instantiate by name or import path, failing with the available options."""
    return resolve_harness(name)(**kwargs)


__all__ = ["BaseHarness", "ConcatHarness", "NoConcatHarness", "CompactHarness",
           "HARNESSES", "build_harness", "budget_mode", "register_harness",
           "resolve_harness"]
