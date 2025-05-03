"""
Abstract base class for model interfaces.
Defines the minimal standard interface that all model implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseModelInterface(ABC):
    """
    Base class for all model interfaces that provides a standardized
    interface for interacting with different underlying models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model interface with a configuration dictionary.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        
    @abstractmethod
    def generate(self, prompts: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses for the given prompts.
        
        Args:
            prompts: List of prompts (strings or dicts with messages/images)
            **kwargs: Additional generation parameters
            
        Returns:
            List of response dictionaries with at least 'text' field
        """
        pass
    
    @abstractmethod
    def format_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format conversation messages into a prompt string.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get basic information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "name": self.config.get("name", "unknown"),
            "type": self.config.get("type", "unknown"),
        }