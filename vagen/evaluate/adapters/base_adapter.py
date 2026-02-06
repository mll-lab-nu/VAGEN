# All comments are in English.
from __future__ import annotations
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from PIL import Image

class ModelAdapter(ABC):
    """Abstract adapter for chat.completions-like multimodal models."""

    @abstractmethod
    def format_system(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def format_user_turn(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def acompletion(self, messages: List[Dict[str, Any]], **chat_config: Any) -> str:
        raise NotImplementedError

    def is_retryable_error(self, exc: BaseException) -> Optional[bool]:
        """
        Optional: Override this to customize retry behavior for specific exceptions.

        Returns:
            True: This error should be retried
            False: This error should NOT be retried
            None: Use default retry logic (recommended for most cases)

        Example:
            def is_retryable_error(self, exc):
                # Custom logic for this API
                if isinstance(exc, MyAPIRateLimitError):
                    return True
                if isinstance(exc, MyAPIAuthError):
                    return False
                return None  # Use default for other errors
        """
        return None  # By default, use ThrottledAdapter's default logic
