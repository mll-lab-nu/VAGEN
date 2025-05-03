import logging
from typing import Dict, Any, Optional

from .base_model import BaseModelInterface
from .providers.vllm_model import VLLMModelInterface

logger = logging.getLogger(__name__)

def create_model_interface(config: Dict[str, Any]) -> Optional[BaseModelInterface]:
    """
    Factory function to create an appropriate model interface based on configuration.
    
    Args:
        config: Configuration dictionary containing:
            - provider: Model provider type (e.g., "vllm", "api")
            - model_name: Name of the model to use
            - Other provider-specific parameters
            
    Returns:
        Initialized model interface or None if initialization fails
    """
    provider = config.get("provider", "vllm").lower()
    model_name = config.get("model_name", "")
    
    logger.info(f"Creating model interface for provider '{provider}' with model '{model_name}'")
    
    try:
        if provider == "vllm":
            return VLLMModelInterface(config)
        # Add more providers as needed
        # elif provider == "openai":
        #     return OpenAIModelInterface(config)
        # elif provider == "claude":
        #     return ClaudeModelInterface(config)
        else:
            logger.error(f"Unknown model provider: {provider}")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize model interface: {str(e)}")
        return None

class ModelFactory:
    """
    Class-based factory for creating and managing model interfaces.
    Provides additional functionality beyond the simple factory function.
    """
    
    @staticmethod
    def create(config: Dict[str, Any]) -> Optional[BaseModelInterface]:
        """
        Create a model interface based on configuration.
        
        Args:
            config: Configuration dictionary with model parameters
            
        Returns:
            Initialized model interface or None if initialization fails
        """
        return create_model_interface(config)
    
    @staticmethod
    def get_available_providers() -> Dict[str, Dict[str, Any]]:
        """
        Get information about available model providers.
        
        Returns:
            Dictionary mapping provider names to their capabilities
        """
        return {
            "vllm": {
                "description": "Local model inference using vLLM",
                "supports_multimodal": True,
                "supported_models": [
                    "Qwen/Qwen2.5-0.5B-Instruct",
                    "Qwen/Qwen2.5-VL-3B-Instruct"
                ]
            }
            # Add more providers as they become available
        }
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and complete a model configuration with defaults if needed.
        
        Args:
            config: Model configuration to validate
            
        Returns:
            Validated configuration with defaults applied
        """
        # Copy to avoid modifying the original
        validated = config.copy()
        
        # Set defaults
        if "provider" not in validated:
            validated["provider"] = "vllm"
        
        # Provider-specific validation
        if validated["provider"] == "vllm":
            if "model_name" not in validated:
                validated["model_name"] = "Qwen/Qwen2.5-0.5B-Instruct"
                
            if "tensor_parallel_size" not in validated:
                validated["tensor_parallel_size"] = 1
                
            # Ensure Qwen models have trust_remote_code=True
            if "Qwen" in validated.get("model_name", ""):
                validated["trust_remote_code"] = True
        
        return validated