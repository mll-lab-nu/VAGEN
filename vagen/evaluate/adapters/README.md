# Model Adapters

This directory contains adapters for various AI model APIs.

## Available Adapters

- **OpenAIAdapter**: OpenAI API (GPT-4, GPT-3.5, etc.)
- **ClaudeAdapter**: Anthropic Claude API
- **GeminiAdapter**: Google Gemini API
- **AzureAdapter**: Azure OpenAI Service
- **TogetherAdapter**: Together.ai API
- **SGLangAdapter**: SGLang inference server
- **VLLMAdapter**: vLLM inference server

## Retry Mechanism

All adapters are wrapped with `ThrottledAdapter` which provides:
- **Automatic retry** on transient errors (rate limits, timeouts, server errors)
- **Exponential backoff** with jitter
- **Concurrency control** to prevent overwhelming APIs

### Customizing Retry Behavior

Each adapter can customize which errors should be retried by implementing the `is_retryable_error()` method.

#### Method Signature

```python
def is_retryable_error(self, exc: BaseException) -> Optional[bool]:
    """
    Determine if an error should be retried.

    Returns:
        True: Always retry this error
        False: Never retry this error
        None: Use default retry logic (recommended)
    """
```

#### Example: GeminiAdapter

```python
class GeminiAdapter(ModelAdapter):
    def is_retryable_error(self, exc: Any):
        """Custom retry logic for Gemini API errors."""
        # Always retry ResourceExhausted (rate limit/quota exceeded)
        if exc.__class__.__name__ == "ResourceExhausted":
            return True

        # Always retry server errors (5xx)
        code = getattr(exc, "code", None)
        if isinstance(code, int) and 500 <= code < 600:
            return True

        # For other errors, use default logic
        return None
```

#### Example: Custom OpenAI Handling

```python
class OpenAIAdapter(ModelAdapter):
    def is_retryable_error(self, exc: Any):
        """Custom retry logic for OpenAI API."""
        from openai import RateLimitError, APITimeoutError, AuthenticationError

        # Always retry rate limits and timeouts
        if isinstance(exc, (RateLimitError, APITimeoutError)):
            return True

        # Never retry authentication errors
        if isinstance(exc, AuthenticationError):
            return False

        # Use default logic for other errors
        return None
```

### Default Retry Logic

If `is_retryable_error()` returns `None` or is not implemented, the default logic applies:

**Retry all errors EXCEPT:**
- HTTP 4xx client errors (except 429 rate limit)
- Errors containing: `authentication`, `auth`, `api key`, `invalid`, `permission`, `not found`, `bad request`

**Always retry:**
- HTTP 429 (rate limit)
- HTTP 5xx (server errors)
- Connection errors, timeouts
- Errors with "rate limit", "quota", "retry" in message

### Retry Configuration

Configure retry behavior via `ThrottleRetryPolicy`:

```python
from view_suite.evaluate.adapters.throttled_adapter import ThrottleRetryPolicy

policy = ThrottleRetryPolicy(
    max_retries=6,              # Maximum retry attempts
    min_backoff=0.5,            # Minimum backoff time (seconds)
    max_backoff=8.0,            # Maximum backoff time (seconds)
    backoff_multiplier=2.0,     # Exponential backoff multiplier
    jitter_frac=0.25,           # Jitter fraction to avoid thundering herd
    max_concurrency=2,          # Max concurrent requests
)
```

### Retry Delay Detection

The system automatically extracts retry delay from:

1. **HTTP Headers**: `Retry-After` header
2. **Error Messages**: Patterns like "Please retry in 7.5s"
3. **Exponential Backoff**: If no delay specified, uses exponential backoff with jitter

Example (Gemini):
```
Error: "ResourceExhausted: Please retry in 7.787239912s"
→ System will wait ~7.79 seconds before retrying
```

## Creating a New Adapter

1. Inherit from `ModelAdapter`
2. Implement required methods: `format_system()`, `format_user_turn()`, `acompletion()`
3. Optionally implement `is_retryable_error()` for custom retry logic
4. Register in `registry.py`

### Minimal Example

```python
from view_suite.evaluate.adapters.base_adapter import ModelAdapter

class MyCustomAdapter(ModelAdapter):
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def format_system(self, text: str, images: List[Image.Image]):
        # Convert to your API format
        return {"role": "system", "content": text}

    def format_user_turn(self, text: str, images: List[Image.Image]):
        # Convert to your API format
        return {"role": "user", "content": text}

    async def acompletion(self, messages: List[Dict[str, Any]], **chat_config: Any) -> str:
        # Call your API
        response = await self.client.generate(messages, **chat_config)
        return response.text

    def is_retryable_error(self, exc: Any):
        # Optional: customize retry logic
        if isinstance(exc, MyAPIRateLimitError):
            return True
        return None  # Use default logic for other errors
```

## Best Practices

1. **Return `None` for most cases**: Let the default logic handle common errors
2. **Be explicit for API-specific errors**: If you know an error type is always/never retryable, specify it
3. **Test your retry logic**: Ensure rate limits and quota errors are retried
4. **Log retry attempts**: Use logging to track retry behavior
5. **Respect retry delays**: Always extract and use `Retry-After` hints from the API

## Troubleshooting

### Errors Not Being Retried

Check if your error is being caught by the default non-retryable logic:
- Does it have a 4xx status code (except 429)?
- Does the error message contain "authentication", "invalid", etc.?

Solution: Implement `is_retryable_error()` to explicitly return `True` for this error type.

### Too Many Retries

- Increase `max_retries` in `ThrottleRetryPolicy`
- Check if API is returning non-transient errors that should not be retried

### API Still Rate Limiting

- Decrease `max_concurrency` in `ThrottleRetryPolicy`
- Increase `min_backoff` and `max_backoff`
- Verify the adapter correctly extracts retry delay from error messages
