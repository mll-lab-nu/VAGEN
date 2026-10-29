"""Evaluation backend facade; importing it registers the built-in backend."""

from vagen.evaluation.backends._common import (
    EvaluationBackend,
    REGISTRY,
    register_adapter,
    register_client,
)
from vagen.evaluation.backends.openrouter import OpenRouterAdapter, build_client_openrouter

__all__ = [
    "EvaluationBackend",
    "OpenRouterAdapter",
    "REGISTRY",
    "build_client_openrouter",
    "register_adapter",
    "register_client",
]
