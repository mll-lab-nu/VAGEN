from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from vagen.mllm_agent.model_interface.base_model_config import BaseModelConfig

@dataclass
class RouterAPIModelConfig(BaseModelConfig):
    """Configuration for OpenRouter API model interface."""
    
    # OpenRouter specific parameters
    api_key: Optional[str] = None  # If None, will use environment variable
    base_url: str = "https://openrouter.ai/api/v1"
    
    # HTTP headers for OpenRouter
    site_url: Optional[str] = None  # For HTTP-Referer header
    site_name: Optional[str] = None  # For X-Title header
    
    # Model parameters
    model_name: str = "qwen/qwen2.5-vl-7b-instruct:free"
    max_retries: int = 3
    timeout: int = 60
    
    # Generation parameters (inherited from base)
    # max_tokens, temperature already defined in base
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    
    # Provider identifier
    provider: str = "routerapi"
    
    def config_id(self) -> str:
        """Generate unique identifier for this configuration."""
        return f"RouterAPIModelConfig({self.model_name},max_tokens={self.max_tokens},temp={self.temperature})"
    
    @staticmethod
    def get_provider_info() -> Dict[str, Any]:
        """Get information about the OpenRouter provider."""
        return {
            "description": "OpenRouter API for accessing various LLMs and VLMs",
            "supports_multimodal": True,
            "supported_models": [
                "qwen/qwen2.5-vl-3b-instruct:free",
                "qwen/qwen2.5-vl-7b-instruct:free",
                "qwen/qwen2.5-vl-72b-instruct:free",
                # Additional models can be added here
            ],
            "default_model": "qwen/qwen2.5-vl-7b-instruct:free"
        }