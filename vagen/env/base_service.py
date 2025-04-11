from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any, Union
import uuid
import logging

class BaseService(ABC):
    """
    Abstract base class for environment services.
    Does not include network communication - just the core environment management.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the BaseService.
        
        Args:
            logger: Optional logger for service logs
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.environments = {}  # Dictionary to store environment instances
        self.env_configs = {}   # Dictionary to store environment configs
    
    @abstractmethod
    def create_environment(self, config: Dict[str, Any]) -> str:
        """
        Create an environment with the given configuration.
        
        Args:
            config: Environment configuration
            
        Returns:
            Environment ID
        """
        pass
    
    def create_environments(self, configs: List[Dict[str, Any]]) -> List[str]:
        """
        Create multiple environments based on the provided configurations.
        
        Args:
            configs: List of environment configurations
            
        Returns:
            List of environment IDs
        """
        env_ids = []
        for config in configs:
            try:
                env_id = self.create_environment(config)
                env_ids.append(env_id)
            except Exception as e:
                self.logger.error(f"Error creating environment: {str(e)}")
                # Clean up environments created so far on error
                for created_id in env_ids:
                    self.close(created_id)
                raise
        return env_ids
    
    @abstractmethod
    def reset(self, env_id: str, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """
        Reset an environment.
        
        Args:
            env_id: Environment ID
            seed: Optional seed for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        pass
    
    def reset_batch(self, env_ids: List[str], seeds: Optional[List[int]] = None) -> List[Tuple[Dict, Dict]]:
        """
        Reset multiple environments.
        
        Args:
            env_ids: List of environment IDs
            seeds: Optional list of seeds, one per environment
            
        Returns:
            List of (observation, info) tuples
        """
        results = []
        for i, env_id in enumerate(env_ids):
            seed = seeds[i] if seeds and i < len(seeds) else None
            try:
                result = self.reset(env_id, seed)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error resetting environment {env_id}: {str(e)}")
                results.append(({}, {"error": str(e)}))
        return results
    
    @abstractmethod
    def step(self, env_id: str, action: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in an environment.
        
        Args:
            env_id: Environment ID
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        pass
    
    def step_batch(self, env_ids: List[str], actions: List[str]) -> List[Tuple[Dict, float, bool, Dict]]:
        """
        Take a step in multiple environments.
        
        Args:
            env_ids: List of environment IDs
            actions: List of actions, one per environment
            
        Returns:
            List of (observation, reward, done, info) tuples
        """
        if len(env_ids) != len(actions):
            raise ValueError("Number of environment IDs must match number of actions")
            
        results = []
        for env_id, action in zip(env_ids, actions):
            try:
                result = self.step(env_id, action)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error stepping environment {env_id}: {str(e)}")
                results.append(({}, 0.0, True, {"error": str(e)}))
        return results
    
    @abstractmethod
    def compute_reward(self, env_id: str) -> float:
        """
        Compute the total reward for an environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            Reward value
        """
        pass
    
    def compute_reward_batch(self, env_ids: List[str]) -> List[float]:
        """
        Compute the total reward for multiple environments.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            List of reward values
        """
        results = []
        for env_id in env_ids:
            try:
                reward = self.compute_reward(env_id)
                results.append(reward)
            except Exception as e:
                self.logger.error(f"Error computing reward for environment {env_id}: {str(e)}")
                results.append(0.0)
        return results
    
    @abstractmethod
    def get_system_prompt(self, env_id: str) -> str:
        """
        Get the system prompt for an environment.
        
        Args:
            env_id: Environment ID
            
        Returns:
            System prompt string
        """
        pass
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> List[str]:
        """
        Get system prompts for multiple environments.
        
        Args:
            env_ids: List of environment IDs
            
        Returns:
            List of system prompt strings
        """
        results = []
        for env_id in env_ids:
            try:
                prompt = self.get_system_prompt(env_id)
                results.append(prompt)
            except Exception as e:
                self.logger.error(f"Error getting system prompt for environment {env_id}: {str(e)}")
                results.append("")
        return results
    
    @abstractmethod
    def close(self, env_id: str) -> None:
        """
        Close an environment and clean up resources.
        
        Args:
            env_id: Environment ID
        """
        pass
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """
        Close multiple environments and clean up resources.
        
        Args:
            env_ids: Optional list of environment IDs to close. If None, close all environments.
        """
        # If no env_ids provided, close all environments
        if env_ids is None:
            env_ids = list(self.environments.keys())
            
        for env_id in env_ids:
            try:
                self.close(env_id)
            except Exception as e:
                self.logger.error(f"Error closing environment {env_id}: {str(e)}")
    
    def _generate_env_id(self) -> str:
        """
        Generate a unique environment ID.
        
        Returns:
            A unique environment ID string
        """
        return str(uuid.uuid4())