from __future__ import annotations
import os
from typing import Any, Dict
from openai import AsyncOpenAI
from vagen.evaluation.backends.openai import OpenAIAdapter
from vagen.evaluation.backends._common.registry import register_adapter, register_client


@register_client("vllm")
def build_client_vllm(cfg: Dict[str, Any]) -> AsyncOpenAI:
    base_url = cfg.get("base_url", "http://127.0.0.1:8000/v1")
    api_key = cfg.get("api_key", os.getenv("VLLM_API_KEY", "EMPTY"))
    return AsyncOpenAI(api_key=api_key, base_url=base_url)

@register_adapter("vllm")
class VLLMAdapter(OpenAIAdapter):
    """
    vLLM OpenAI server adapter (OpenAI-compatible).
    """
    pass
