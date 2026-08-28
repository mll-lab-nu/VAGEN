"""Shared evaluation backend contracts and registry."""

from vagen.evaluation.backends._common.base import EvaluationBackend
from vagen.evaluation.backends._common.registry import (
    REGISTRY,
    register_adapter,
    register_client,
)

__all__ = ["EvaluationBackend", "REGISTRY", "register_adapter", "register_client"]
