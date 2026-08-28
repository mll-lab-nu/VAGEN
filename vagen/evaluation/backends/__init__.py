"""Evaluation backend facade; importing it registers all built-in backends."""

from vagen.evaluation.backends._common import (
    EvaluationBackend,
    REGISTRY,
    register_adapter,
    register_client,
)
from vagen.evaluation.backends.claude import ClaudeAdapter, build_client_claude
from vagen.evaluation.backends.gemini import GeminiAdapter, build_client_gemini
from vagen.evaluation.backends.openai import OpenAIAdapter, build_client_azure, build_client_openai
from vagen.evaluation.backends.openai_responses import OpenAIResponsesAdapter
from vagen.evaluation.backends.sglang import SGLangAdapter, build_client_sglang
from vagen.evaluation.backends.together import TogetherAdapter, build_client_together
from vagen.evaluation.backends.vllm import VLLMAdapter, build_client_vllm

__all__ = [
    "ClaudeAdapter",
    "EvaluationBackend",
    "GeminiAdapter",
    "OpenAIAdapter",
    "OpenAIResponsesAdapter",
    "REGISTRY",
    "SGLangAdapter",
    "TogetherAdapter",
    "VLLMAdapter",
    "build_client_azure",
    "build_client_claude",
    "build_client_gemini",
    "build_client_openai",
    "build_client_sglang",
    "build_client_together",
    "build_client_vllm",
    "register_adapter",
    "register_client",
]
