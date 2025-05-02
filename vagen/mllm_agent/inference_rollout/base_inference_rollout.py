from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Tuple, Any

class BaseInferenceRollout(ABC):
    """
    Abstract base class for inference rollout managers.
    Focuses on core environment interaction functionality.
    Logging and metrics calculation are handled by the caller.
    """

    @abstractmethod
    def reset(self, env_configs: List[Dict]) -> Dict[str, Tuple[Dict, Dict]]:
        """
        Reset environments based on provided configurations.

        Args:
            env_configs: List of environment configurations

        Returns:
            Dictionary mapping environment IDs to (observation, info) tuples
        """
        pass

    @abstractmethod
    def run(self, max_steps: int = 10) -> None:
        """
        Run inference on all environments until completion or max steps.
        This is the main entry point for inference.

        Args:
            max_steps: Maximum number of steps to run for each environment
        """
        pass

    @abstractmethod
    def recording_to_log(self) -> List[Dict]:
        """
        Format and return results in a format compatible with PPO trainer logging.

        Returns:
            List of dictionaries with:
            - env_id: Environment ID
            - config_id: Configuration ID
            - output_str: Formatted output string
            - image_data: List of images (if applicable)
            - metrics: Dictionary of metrics for this environment
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close all environments and clean up resources.
        """
        pass