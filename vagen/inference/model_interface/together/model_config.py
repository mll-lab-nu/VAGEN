from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from vagen.inference.model_interface.base_model_config import BaseModelConfig

@dataclass
class TogetherModelConfig(BaseModelConfig):
    """Configuration for Together AI model interface."""
    
    # Together AI specific parameters
    api_key: Optional[str] = None  # If None, will use environment variable
    base_url: str = "https://api.together.xyz"
    
    # Model parameters
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    max_retries: int = 3
    timeout: int = 60
    
    # Generation parameters (inherited from base)
    # max_tokens, temperature already defined in base
    top_p: float = 0.9
    top_k: int = 40
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    
    # Provider identifier
    provider: str = "together"
    
    def config_id(self) -> str:
        """Generate unique identifier for this configuration."""
        return f"TogetherModelConfig({self.model_name},max_tokens={self.max_tokens},temp={self.temperature})"
    
    @staticmethod
    def get_provider_info() -> Dict[str, Any]:
        """Get information about the Together AI provider."""
        return {
            "description": "Together AI for various open source models",
            "supports_multimodal": True,
            "supported_models": [
                "Qwen/Qwen2.5-VL-72B-Instruct",
            ],
            "default_model": "Qwen/Qwen2.5-VL-72B-Instruct"
        }