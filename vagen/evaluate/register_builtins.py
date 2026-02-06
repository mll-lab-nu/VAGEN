# All comments are in English.
from __future__ import annotations
from vagen.evaluate.registry import REGISTRY
from vagen.evaluate.clients import (
    build_client_openai, build_client_azure, build_client_sglang,
    build_client_vllm, build_client_together,
    build_client_claude, build_client_gemini
)
from vagen.evaluate.adapters.openai_adapter import OpenAIAdapter
from vagen.evaluate.adapters.sglang_adapter import SGLangAdapter
from vagen.evaluate.adapters.vllm_adapter import VLLMAdapter
from vagen.evaluate.adapters.together_adapter import TogetherAdapter
from vagen.evaluate.adapters.claude_adapter import ClaudeAdapter
from vagen.evaluate.adapters.gemini_adapter import GeminiAdapter

# Clients
REGISTRY.register_client("openai",   build_client_openai)
REGISTRY.register_client("azure",    build_client_azure)
REGISTRY.register_client("sglang",   build_client_sglang)
REGISTRY.register_client("vllm",     build_client_vllm)
REGISTRY.register_client("together", build_client_together)
REGISTRY.register_client("claude",   build_client_claude)  
REGISTRY.register_client("gemini",   build_client_gemini)  

# Adapters
REGISTRY.register_adapter("openai",   lambda **k: OpenAIAdapter(**k))
REGISTRY.register_adapter("azure",    lambda **k: OpenAIAdapter(**k))  # Azure uses deployment as model
REGISTRY.register_adapter("sglang",   lambda **k: SGLangAdapter(**k))
REGISTRY.register_adapter("vllm",     lambda **k: VLLMAdapter(**k))
REGISTRY.register_adapter("together", lambda **k: TogetherAdapter(**k))
REGISTRY.register_adapter("claude",   lambda **k: ClaudeAdapter(**k))  # NEW
REGISTRY.register_adapter("gemini",   lambda **k: GeminiAdapter(**k))  # NEW
