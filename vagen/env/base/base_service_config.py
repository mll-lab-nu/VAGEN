from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, List, Union
@dataclass
class BaseServiceConfig(ABC):
    max_workers: int = 10
    
    def get(self, key, default=None):
        """
        Get the value of a config key.
        Args:
            key: Key to get
            default: Default value if key is not found
        """
        return getattr(self, key, default)