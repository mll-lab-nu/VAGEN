# All comments are in English.
from __future__ import annotations
from vagen.evaluate.adapters.openai_adapter import OpenAIAdapter
from vagen.evaluate.registry import register_adapter

@register_adapter("together")
class TogetherAdapter(OpenAIAdapter):
    """
    Together AI adapter (OpenAI-compatible).
    """
    pass
