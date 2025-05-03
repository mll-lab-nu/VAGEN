# vagen/mllm_agent/base_rollout.py
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any

class BaseRollout(ABC):
    """
    Abstract base class for all rollout managers (inference and training).
    Defines common interface for environment interaction.
    """
    
    @abstractmethod
    def reset(self, env_configs: List[Dict]) -> Dict[str, Tuple[Dict, Dict]]:
        """Reset environments with provided configurations."""
        pass
    
    @abstractmethod
    def run(self, max_steps: int = None) -> None:
        """Run the rollout process."""
        pass
    
    @abstractmethod
    def recording_to_log(self) -> List[Dict]:
        """Format and return results for logging."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass