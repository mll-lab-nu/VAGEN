from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Optional, Dict, Any

@dataclass
class BaseModelConfig(ABC):
    """Abstract base configuration for all model interfaces."""
    
    # Common parameters across all models
    model_name: str
    max_tokens: int = 1024
    temperature: float = 0.7
    seed: Optional[int] = None
    
    @abstractmethod
    def config_id(self) -> str:
        """Generate a unique identifier for this configuration."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for model initialization."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
    
    def get(self, key, default=None):
        """
        Get the value of a config key.
        Args:
            key: Key to get
            default: Default value if key is not found
        """
        return getattr(self, key, default)