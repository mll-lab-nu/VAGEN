from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Dict

class BaseEnv(ABC):
    def __init__(self, config):
        self.config = config    
    
    
    @abstractmethod
    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: Action to take, must be in action space, or default invalid action
            
        Returns:
            obs, reward, done, info
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close the environment."""
        pass
    
    @abstractmethod
    def reset(self, seed: Optional[Any] = None) -> Tuple[Any, Dict]:
        """
        Reset the environment.
        NOTE: the environment should be same for the same seed
        Args:
            seed: Seed for the environment
            
        Returns:
            obs,info
        """
        pass