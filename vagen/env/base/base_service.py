from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any, Union
import uuid
from concurrent.futures import ThreadPoolExecutor

class BaseService(ABC):
    """
    Abstract base class for environment services.
    Implements batch operations for efficient parallel processing.
    Single environment operations are provided as convenience methods
    that invoke the corresponding batch methods.
    """
    
    @abstractmethod
    def create_environments_batch(self, ids2configs: Dict[str, Any]) -> None:
        """
        Create multiple environments in parallel.

        Args:
            ids2configs (Dict[Any, Any]): 
                A dictionary where each key is an environment ID and the corresponding
                value is the configuration for that environment.
            id: a string
            config: {"env_name": env_name, "env_config": env_config}
                env_name: The name of the environment to create.
                env_config: A dictionary containing the configuration parameters for the environment.
        Returns:
            None

        Note:
            The implementation should create all environments concurrently.
            It should gracefully handle errors and perform cleanup of any partially created environments.
        """
        pass

    @abstractmethod
    def reset_batch(self, ids2seeds: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """
        Reset multiple environments in parallel.

        Args:
            ids2seeds (Dict[Any, Any]):
                A dictionary where each key is an environment ID and the corresponding
                value is a seed value (or None for using default seeding behavior).

        Returns:
            Dict[Any, Tuple[Any, Any]]:
                A dictionary mapping environment IDs to tuples of the form (observation, info),
                where 'observation' is the initial state after reset, and 'info' contains additional details.

        Note:
            For environments with a None seed, the default seeding behavior should be applied.
        """
        pass

    @abstractmethod
    def step_batch(self, ids2actions: Dict[str, Any]) -> Dict[str, Tuple[Dict, float, bool, Dict]]:
        """
        Step through multiple environments in parallel.

        Args:
            ids2actions (Dict[Any, Any]):
                A dictionary where each key is an environment ID and the corresponding
                value is the action to execute in that environment.

        Returns:
            Dict[Any, Tuple[Dict, float, bool, Dict]]:
                A dictionary mapping environment IDs to tuples of the form 
                (observation, reward, done, info), where:
                    - 'observation' is the new state of the environment after the action,
                    - 'reward' is a float representing the reward received,
                    - 'done' is a boolean indicating whether the environment is finished,
                    - 'info' contains additional information or context.

        Note:
            The implementation should process all steps in parallel while ensuring that 
            each action is correctly applied to its corresponding environment.
        """
        pass

    @abstractmethod
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[str, float]:
        """
        Compute the total reward for multiple environments in parallel.

        Args:
            env_ids (List[str]): A list of environment IDs.

        Returns:
            Dict[Any, float]:
                A dictionary mapping each environment ID to its computed total reward.

        Note:
            The implementation should compute rewards concurrently.
        """
        pass

    @abstractmethod
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[str, str]:
        """
        Retrieve system prompts for multiple environments in parallel.

        Args:
            env_ids (List[str]): A list of environment IDs.

        Returns:
            Dict[Any, str]:
                A dictionary mapping each environment ID to its corresponding system prompt string.

        Note:
            The implementation should retrieve all system prompts concurrently.
        """
        pass

    @abstractmethod
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """
        Close multiple environments and clean up resources in parallel.

        Args:
            env_ids (Optional[List[str]]):
                A list of environment IDs to close. If None, all environments should be closed.

        Returns:
            None

        Note:
            The implementation should perform cleanup concurrently and handle any errors gracefully.
        """
        pass
