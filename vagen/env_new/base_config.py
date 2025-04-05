from dataclasses import dataclass, field
from abc import ABC, abstractmethod
@dataclass
class BaseConfig(ABC):
    
    @abstractmethod
    def config_id(self) -> str: # config identifier, wandb and mllm rollout manager use this to identify the config
        pass
    
    def __init__(self, **kwargs):
        pass
    
    def get(self, key, default=None):
        """
        Get the value of a config key.
        Args:
            key: Key to get
            default: Default value if key is not found
        """
        return getattr(self, key, default)