"""Shared evaluation backend contracts and registry."""

from vagen.evaluation.backends._common.base import ModelAdapter
from vagen.evaluation.backends._common.registry import (
    REGISTRY,
    register_adapter,
    register_client,
)

__all__ = ["ModelAdapter", "REGISTRY", "register_adapter", "register_client"]
