# All comments are in English.
from __future__ import annotations
import os
from typing import Any, Dict
from openai import AsyncAzureOpenAI, AsyncOpenAI

def build_client_openai(cfg: Dict[str, Any]) -> AsyncOpenAI:
    api_key = cfg.get("api_key") or os.getenv("OPENAI_API_KEY", "")
    base_url = cfg.get("base_url")
    return AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key)

def build_client_azure(cfg: Dict[str, Any]) -> AsyncAzureOpenAI:
    endpoint = cfg.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = cfg.get("azure_api_key") or os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = cfg.get("azure_api_version") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    if not endpoint or not api_key:
        raise ValueError("Azure endpoint/api_key missing.")
    return AsyncAzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=api_key)

def build_client_sglang(cfg: Dict[str, Any]) -> AsyncOpenAI:
    base_url = cfg.get("base_url", "http://127.0.0.1:30000/v1")
    api_key = cfg.get("api_key", os.getenv("SGLANG_API_KEY", "EMPTY"))
    return AsyncOpenAI(api_key=api_key, base_url=base_url)

def build_client_vllm(cfg: Dict[str, Any]) -> AsyncOpenAI:
    base_url = cfg.get("base_url", "http://127.0.0.1:8000/v1")
    api_key = cfg.get("api_key", os.getenv("VLLM_API_KEY", "EMPTY"))
    return AsyncOpenAI(api_key=api_key, base_url=base_url)

def build_client_together(cfg: Dict[str, Any]) -> AsyncOpenAI:
    base_url = cfg.get("base_url", "https://api.together.xyz/v1")
    api_key = cfg.get("api_key") or os.getenv("TOGETHER_API_KEY", "")
    if not api_key:
        raise ValueError("Together API key missing.")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)

# ---------- NEW: Claude (Anthropic) ----------
def build_client_claude(cfg: Dict[str, Any]):
    """
    Build anthropic.AsyncAnthropic; requires 'anthropic' package.
    """
    from anthropic import AsyncAnthropic
    api_key = cfg.get("api_key") or os.getenv("ANTHROPIC_API_KEY", "")
    base_url = cfg.get("base_url", None)
    if not api_key:
        raise ValueError("Anthropic API key missing.")
    if base_url:
        return AsyncAnthropic(api_key=api_key, base_url=base_url)
    return AsyncAnthropic(api_key=api_key)

# ---------- NEW: Gemini ----------
class _GeminiClient:
    """Tiny wrapper configuring google-generativeai globally."""
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai

def build_client_gemini(cfg: Dict[str, Any]) -> _GeminiClient:
    api_key = cfg.get("api_key") or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("Gemini API key missing (set GEMINI_API_KEY or GOOGLE_API_KEY).")
    return _GeminiClient(api_key)
