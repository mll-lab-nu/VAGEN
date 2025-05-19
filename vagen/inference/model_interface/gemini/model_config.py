from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from vagen.inference.model_interface.base_model_config import BaseModelConfig

@dataclass
class GeminiModelConfig(BaseModelConfig):
    """Configuration for Google Gemini API model interface."""
    
    # Gemini specific parameters
    api_key: Optional[str] = None  # If None, will use environment variable
    project_id: Optional[str] = None  # Google Cloud project ID
    location: str = "us-central1"  # Default region
    
    # Model parameters
    model_name: str = "gemini-2.0-flash-001"
    max_retries: int = 3
    timeout: int = 60
    
    # Generation parameters (inherited from base)
    # max_tokens, temperature already defined in base
    top_p: float = 0.9
    top_k: int = 40
    stop_sequences: Optional[List[str]] = field(default_factory=lambda: ["<|im_end|>"])

    
    # Provider identifier
    provider: str = "gemini"
    
    def config_id(self) -> str:
        """Generate unique identifier for this configuration."""
        return f"GeminiModelConfig({self.model_name},max_tokens={self.max_tokens},temp={self.temperature})"
    
    @staticmethod
    def get_provider_info() -> Dict[str, Any]:
        """Get information about the Gemini provider."""
        return {
            "description": "Google Gemini API for generative AI models",
            "supports_multimodal": True,
            "supported_models": [
                "gemini-1.5-flash-001",
                "gemini-1.5-pro-001",
                "gemini-2.0-flash",
                "gemini-2.0-pro-001",
                "gemini-2.5-flash-preview-04-17",
                "gemini-2.0-flash"
            ],
            "default_model": "gemini-2.0-flash"
        }