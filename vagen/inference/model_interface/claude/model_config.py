from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from vagen.inference.model_interface.base_model_config import BaseModelConfig

@dataclass
class ClaudeModelConfig(BaseModelConfig):
    """Configuration for Claude API model interface."""
    
    # Claude specific parameters
    api_key: Optional[str] = None  # If None, will use environment variable
    base_url: Optional[str] = None  # For custom endpoints
    
    # Model parameters
    model_name: str = "claude-3-haiku-20240307"
    max_retries: int = 3
    timeout: int = 60
    
    # Generation parameters (inherited from base)
    # max_tokens, temperature already defined in base
    top_p: float = 0.9
    top_k: int = -1  # Claude uses -1 for no limit
    stop_sequences: Optional[List[str]] = field(default_factory=lambda: ["<|im_end|>"])
    
    # Claude specific parameters
    metadata: Optional[Dict[str, str]] = None
    stream: bool = False
    
    # Batch processing options
    use_batch_api: bool = False  # Whether to use batch API by default
    batch_poll_interval: int = 5  # Seconds between polling batch status
    batch_max_wait_time: int = 3600  # Maximum time to wait for batch completion
    
    # Provider identifier
    provider: str = "claude"
    
    def config_id(self) -> str:
        """Generate unique identifier for this configuration."""
        batch_suffix = "_batch" if self.use_batch_api else ""
        return f"ClaudeModelConfig({self.model_name},max_tokens={self.max_tokens},temp={self.temperature}{batch_suffix})"
    
    @staticmethod
    def get_provider_info() -> Dict[str, Any]:
        """Get information about the Claude provider."""
        return {
            "description": "Anthropic Claude API for conversation models",
            "supports_multimodal": True,
            "supports_batch": True,
            "supported_models": [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-sonnet-latest",
                "claude-3-7-sonnet-20250219"
            ],
            "default_model": "claude-3-haiku-20240307",
            "batch_api_info": {
                "max_requests_per_batch": 10000,
                "result_availability": "24 hours",
                "pricing_discount": "50%"
            }
        }