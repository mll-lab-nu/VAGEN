# All comments are in English.
from __future__ import annotations
import asyncio
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from vagen.evaluate.adapters.base_adapter import ModelAdapter

@dataclass
class ThrottleRetryPolicy:
    """Config for concurrency gate and retry backoff."""
    max_concurrency: int = 2
    max_retries: int = 6
    min_backoff: float = 0.5
    max_backoff: float = 8.0
    backoff_multiplier: float = 2.0
    jitter_frac: float = 0.25
    shared_gate: Optional[asyncio.Semaphore] = None
    retryable_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504)

def _get_status_code(exc: BaseException) -> Optional[int]:
    """Extract HTTP status code from various exception types."""
    # Try status_code first (OpenAI, Anthropic, most REST APIs)
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code

    # Try code attribute (Google API exceptions, gRPC)
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code

    # Try from response object
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if isinstance(code, int):
            return code

    return None

def _get_retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Extract retry delay from exception headers or message."""
    # Try from response headers first
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers:
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if ra is not None:
            try:
                return float(ra)
            except Exception:
                pass

    # Try parsing from error message (e.g., Gemini: "Please retry in 7.787239912s.")
    import re
    msg = str(exc)
    match = re.search(r'retry in ([\d.]+)s', msg, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            pass

    return None

def _is_transient_network_error(exc: BaseException) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    hints = [
        "timeout", "timed out", "rate limit", "ratelimit", "too many requests",
        "connection reset", "connection aborted", "connection error",
        "temporarily unavailable", "service unavailable", "bad gateway",
        "gateway timeout", "retry later",
    ]
    return any(h in msg for h in hints) or any(h in name for h in ["timeout", "ratelimit", "rate"])

def _is_non_retryable(exc: BaseException) -> bool:
    """Check if exception is clearly non-retryable (auth, invalid params, etc.)."""
    # Check for non-retryable status codes (client errors except 429)
    code = _get_status_code(exc)
    if code is not None:
        # 4xx client errors (except 429 rate limit) are usually not retryable
        if 400 <= code < 500 and code != 429:
            return True

    # Check exception class name and message for non-retryable errors
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    non_retryable_hints = [
        "authentication", "auth", "api key", "invalid",
        "permission", "not found", "bad request",
    ]
    if any(h in name or h in msg for h in non_retryable_hints):
        return True

    return False

def _is_retryable(exc: BaseException, retryable_codes: Tuple[int, ...]) -> bool:
    """By default retry all errors except clearly non-retryable ones."""
    # If it's clearly non-retryable, don't retry
    if _is_non_retryable(exc):
        return False

    # Otherwise retry: could be rate limit, timeout, server error, etc.
    return True

class ThrottledAdapter(ModelAdapter):
    """Thin wrapper adding concurrency gate and retry backoff to any ModelAdapter."""

    def __init__(self, inner: ModelAdapter, policy: Optional[ThrottleRetryPolicy] = None):
        self.inner = inner
        self.policy = policy or ThrottleRetryPolicy()
        self._gate = self.policy.shared_gate or asyncio.BoundedSemaphore(self.policy.max_concurrency)

    def format_system(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        return self.inner.format_system(text, images)

    def format_user_turn(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        return self.inner.format_user_turn(text, images)

    async def acompletion(self, messages: List[Dict[str, Any]], **chat_config: Any) -> str:
        attempt = 0
        while True:
            try:
                async with self._gate:
                    return await self.inner.acompletion(messages, **chat_config)
            except Exception as e:
                # Check if adapter has custom retry logic
                adapter_decision = None
                if hasattr(self.inner, 'is_retryable_error'):
                    adapter_decision = self.inner.is_retryable_error(e)

                # If adapter returned a decision (True/False), use it
                # If adapter returned None, use default logic
                should_retry = adapter_decision if adapter_decision is not None else _is_retryable(e, self.policy.retryable_status_codes)

                if not should_retry:
                    raise
                attempt += 1
                if attempt > self.policy.max_retries:
                    raise
                retry_after = _get_retry_after_seconds(e)
                backoff = self._compute_backoff(attempt, retry_after)
                await asyncio.sleep(backoff)

    def _compute_backoff(self, attempt: int, retry_after: Optional[float]) -> float:
        if retry_after is not None:
            return max(self.policy.min_backoff, min(self.policy.max_backoff, retry_after))
        base = min(self.policy.max_backoff, self.policy.min_backoff * (self.policy.backoff_multiplier ** (attempt - 1)))
        jitter = base * self.policy.jitter_frac * random.random()
        return base + jitter
