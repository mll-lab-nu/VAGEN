from __future__ import annotations
import os
from typing import Any, Dict
from openai import AsyncOpenAI
from vagen.evaluation.backends.openai import OpenAIAdapter
from vagen.evaluation.backends._common.registry import register_adapter, register_client


@register_client("together")
def build_client_together(cfg: Dict[str, Any]) -> AsyncOpenAI:
    base_url = cfg.get("base_url", "https://api.together.xyz/v1")
    api_key = cfg.get("api_key") or os.getenv("TOGETHER_API_KEY", "")
    if not api_key:
        raise ValueError("Together API key missing.")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)

@register_adapter("together")
class TogetherAdapter(OpenAIAdapter):
    """
    Together AI adapter (OpenAI-compatible).
    """
    pass
