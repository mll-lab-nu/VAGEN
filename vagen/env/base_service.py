from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any, Union
import uuid
from concurrent.futures import ThreadPoolExecutor

class BaseService(ABC):
    """
    Abstract base class for environment services.
    Focuses on batch operations for efficient parallel processing.
    Single environment operations are implemented as convenience methods
    that call the corresponding batch methods.
    """
    
    def __init__(self, max_workers: int = 10):
        """
        Initialize the BaseService.
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.environments = {}  # Dictionary to store environment instances
        self.env_configs = {}   # Dictionary to store environment configs
        self.max_workers = max_workers
    
    @abstractmethod
    def create_environments_batch(self, configs: List[Dict[str, Any]]) -> List[str]:
        """
        Create multiple environments based on the provided configurations in parallel.
        
        Args:
            configs: List of environment configurations
            
        Returns:
            List of environment IDs
            
        Note:
            Implementation should use parallel processing for efficiency.
            Should handle errors gracefully and clean up any partially created environments.
        """
        pass
    
    @abstractmethod
    def reset_batch(self, env_ids: List[str], seeds: Optional[List[Optional[int]]] = None) -> List[Tuple[Dict, Dict]]:
        """
        Reset multiple environments in parallel.
        
        Args:
            env_ids: List of environment IDs to reset
            seeds: Optional list of seeds, one per environment (None for no seed)
            
        Returns:
            List of (observation, info) tuples
            
        Note:
            Implementation should use parallel processing for efficiency.
            If seeds is None, use default seeding behavior for all environments.
            If seeds is provided but shorter than env_ids, remaining environments use None.
        """
        pass
    
    @abstractmethod
    def step_batch(self, env_ids: List[str], actions: List[str]) -> List[Tuple[Dict, float, bool, Dict]]:
        """
        Take a step in multiple environments in parallel.
        
        Args:
            env_ids: List of environment IDs
            actions: List of actions to take in each environment
            
        Returns:
            List of (observation, reward, done, info) tuples
            
        Note:
            Implementation should use parallel processing for efficiency.
            Length of env_ids and actions must match.
        """
        pass
    
    @abstractmethod
    def compute_reward_batch(self, env_ids: List[str]) -> List[float]:
        """
        Compute the total reward for multiple environments in parallel.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            List of reward values
            
        Note:
            Implementation should use parallel processing for efficiency.
        """
        pass
    
    @abstractmethod
    def get_system_prompts_batch(self, env_ids: List[str]) -> List[str]:
        """
        Get system prompts for multiple environments in parallel.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            List of system prompt strings
            
        Note:
            Implementation should use parallel processing for efficiency.
        """
        pass
    
    @abstractmethod
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """
        Close multiple environments and clean up resources in parallel.
        
        Args:
            env_ids: Optional list of environment IDs to close. If None, close all environments.
            
        Note:
            Implementation should use parallel processing for efficiency.
            Should handle errors gracefully during cleanup.
        """
        pass
    