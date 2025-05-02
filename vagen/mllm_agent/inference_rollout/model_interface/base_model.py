from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseModelInterface(ABC):
    """
    Simplified base class for model interfaces that focuses only on
    the essential methods needed for inference.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config

    @abstractmethod
    def generate(self, prompts: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses for the given prompts.
        
        Args:
            prompts: List of prompts (strings or dicts with messages/images)
            **kwargs: Additional generation parameters
            
        Returns:
            List of response dictionaries with text, finish_reason, etc.
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

    @property
    @abstractmethod
    def is_multimodal(self) -> bool:
        """Whether the model supports images."""
        pass

    @abstractmethod
    def process_images(self, images: List[Any]) -> List[Any]:
        """
        Process images for multi-modal models.
        
        Args:
            images: List of image data
            
        Returns:
            Processed image data
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
            "is_multimodal": self.is_multimodal
        }