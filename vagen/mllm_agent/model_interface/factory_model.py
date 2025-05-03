# vagen/mllm_agent/model_interface/factory_model.py

import logging
from typing import Dict, Any, Optional

from .base_model import BaseModelInterface
from .base_model_config import BaseModelConfig
from . import REGISTERED_MODEL

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory for creating and managing model interfaces.
    Uses REGISTERED_MODEL to find and create model instances.
    """
    
    @staticmethod
    def create(config: Dict[str, Any]) -> BaseModelInterface:
        """
        Create a model interface based on configuration.
        
        Args:
            config: Configuration dictionary with model parameters
            
        Returns:
            Initialized model interface
            
        Raises:
            ValueError: If provider is unknown or initialization fails
        """
        provider = config.get("provider", "vllm").lower()
        model_name = config.get("model_name", "")
        
        logger.info(f"Creating model interface for provider '{provider}' with model '{model_name}'")
        
        if provider not in REGISTERED_MODEL:
            available_providers = list(REGISTERED_MODEL.keys())
            raise ValueError(f"Unknown provider '{provider}'. Available providers: {available_providers}")
        
        try:
            # Get model and config classes from registry
            model_cls = REGISTERED_MODEL[provider]["model_cls"]
            config_cls = REGISTERED_MODEL[provider]["config_cls"]
            
            # Create config instance
            model_config = config_cls(**config)
            
            # Create and return model instance
            return model_cls(model_config)
            
        except Exception as e:
            logger.error(f"Failed to initialize model interface: {str(e)}")
            raise
    
    @staticmethod
    def create_from_config_instance(config: BaseModelConfig) -> BaseModelInterface:
        """
        Create a model interface from a config instance.
        
        Args:
            config: Model configuration instance
            
        Returns:
            Initialized model interface
        """
        # Find provider by checking config class type
        for provider, info in REGISTERED_MODEL.items():
            if isinstance(config, info["config_cls"]):
                model_cls = info["model_cls"]
                return model_cls(config)
        
        raise ValueError(f"No registered provider found for config type {type(config).__name__}")
    
    @staticmethod
    def get_available_providers() -> Dict[str, Dict[str, Any]]:
        """
        Get information about available model providers.
        
        Returns:
            Dictionary mapping provider names to their capabilities
        """
        providers = {}
        
        for provider, info in REGISTERED_MODEL.items():
            provider_info = {
                "model_class": info["model_cls"].__name__,
                "config_class": info["config_cls"].__name__,
            }
            
            # Add provider-specific info if available
            config_cls = info["config_cls"]
            if hasattr(config_cls, 'get_provider_info'):
                provider_info.update(config_cls.get_provider_info())
            
            providers[provider] = provider_info
        
        return providers
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and complete a model configuration with defaults.
        
        Args:
            config: Model configuration to validate
            
        Returns:
            Validated configuration with defaults applied
        """
        provider = config.get("provider", "vllm").lower()
        
        if provider not in REGISTERED_MODEL:
            logger.warning(f"Unknown provider '{provider}' during validation")
            return config
        
        try:
            # Get config class from registry
            config_cls = REGISTERED_MODEL[provider]["config_cls"]
            
            # Create config instance (which applies defaults)
            model_config = config_cls(**config)
            
            # Convert back to dictionary
            return model_config.to_dict()
            
        except Exception as e:
            logger.warning(f"Could not validate config for provider {provider}: {e}")
            # Return original config if validation fails
            return config
    
    @staticmethod
    def create_from_yaml_config(yaml_config: Dict[str, Any], 
                                model_name: str) -> BaseModelInterface:
        """
        Create a model interface from YAML config format.
        
        Args:
            yaml_config: Full YAML config with potentially multiple models
            model_name: Name of the specific model to create
            
        Returns:
            Initialized model interface
            
        Raises:
            KeyError: If model_name not found in config
        """
        if "models" in yaml_config:
            models_config = yaml_config["models"]
        else:
            models_config = yaml_config
            
        if model_name not in models_config:
            raise KeyError(f"Model '{model_name}' not found in configuration")
            
        model_config = models_config[model_name]
        return ModelFactory.create(model_config)
    
    @staticmethod
    def is_provider_supported(provider: str) -> bool:
        """
        Check if a provider is supported.
        
        Args:
            provider: Provider name to check
            
        Returns:
            True if provider is supported, False otherwise
        """
        return provider.lower() in REGISTERED_MODEL