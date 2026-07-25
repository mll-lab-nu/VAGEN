"""This module is adapt from https://github.com/zeno-ml/zeno-build"""
try:
    from .providers.gemini_utils import generate_from_gemini_completion
except Exception:
    print('Google Cloud not set up, skipping import of providers.gemini_utils.generate_from_gemini_completion')

try:
    from .providers.hf_utils import generate_from_huggingface_completion
except Exception as _e:
    print(f'HuggingFace not set up, skipping ({_e})')

from .providers.openai_utils import (
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
)
try:
    from .providers.api_utils import (
        generate_with_api,
    )
except Exception as _e:
    print(f'API providers (dashscope/anthropic/google) not set up, skipping ({_e})')

try:
    from .utils import call_llm
except Exception as _e:
    print(f'llms.utils not available, skipping ({_e})')

__all__ = [
    "generate_from_openai_completion",
    "generate_from_openai_chat_completion",
    "generate_from_huggingface_completion",
    "generate_from_gemini_completion",
    "call_llm",
]
