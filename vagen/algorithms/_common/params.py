"""Shared helpers for reading algorithm hyperparameters from ``config.algorithm``.

verl's ``AlgoConfig`` is a dataclass with a fixed field list. A custom algorithm may need
a knob that is not on that list; Hydra requires such keys to be appended with ``+``.
Reading a required knob with a default can turn a typo or missing override into a silently
different algorithm.

So: :func:`required` for a knob that defines the algorithm, :func:`optional` for one
that genuinely has a right default. Prefer ``required``. A parameter deserves a default
only when leaving it out means something -- not merely when a value has to be picked.
"""

from __future__ import annotations

_MISSING = object()


def _read(config, name):
    if config is None:
        return _MISSING
    if hasattr(config, "get"):
        value = config.get(name, _MISSING)
        if value is not _MISSING:
            return value
    return getattr(config, name, _MISSING)


def required(config, name: str, estimator: str, why: str):
    """An estimator hyperparameter with no sensible default.

    Args:
        name: the key under ``algorithm``.
        estimator: the registered estimator name, for the error message.
        why: what silently happens if it is missing -- the part a reader needs in order
            to understand why this is not just defaulted.
    """
    value = _read(config, name)
    if value is _MISSING or value is None:
        raise ValueError(
            f"{estimator} needs algorithm.{name}. {why} "
            f"verl's AlgoConfig has no such field, so append it with hydra's plus: "
            f"`+algorithm.{name}=...`."
        )
    return value


def optional(config, name: str, default):
    """An estimator hyperparameter that means something specific when left out."""
    value = _read(config, name)
    return default if value is _MISSING or value is None else value
