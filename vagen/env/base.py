from abc import ABC, abstractmethod
import re
from typing import Optional, List, Tuple, Any, Dict
from copy import deepcopy
from transformers import AutoTokenizer
import torch
from PIL import Image
import numpy as np
from dataclasses import dataclass, field


# ===============================
# Prompt Template
# ===============================

@dataclass
class PromptTemplate:
    """Dataclass for managing environment-specific prompts."""
    system_prompt: str = ""
    instruction_prompt: str = ""
    init_observation_template: str = ""
    valid_action_template: str = ""
    invalid_action_template: str = ""

    def get_task_instruction(self) -> str:
        """Get the system and instruction prompts"""
        return self.system_prompt + '\n' +self.instruction_prompt



@dataclass
class EnvObservation:
    """
    Dataclass for managing environment observation.
    
    Attributes:
        observation_template: String template for text-based observations.
        multi_modal_observation: Dictionary for storing observations of various modalities
                                (e.g., visual, audio, text).
        placeholder_format: Format string for generating image placeholders. Default: "<image{index}>"

    NOTE currently only support text and image observation
    """
    observation_template: str = ""
    multi_modal_observation: Dict[str, Any] = field(default_factory=dict)
    placeholder_format: str = "<image{index}>"
    
    def __post_init__(self):
        """Validate observation data after initialization."""
        if not isinstance(self.multi_modal_observation, dict):
            raise TypeError("multi_modal_observation must be a dictionary")
            
        # Validate placeholder_format has {index} in it
        if "{index}" not in self.placeholder_format:
            raise ValueError("placeholder_format must contain '{index}' for proper formatting")
    
    def create_observation(
            self,
            template: str,
            contents: List[Any],
            replace_keys: List[str] = ['observation']
        ) -> None:
        """
        Create an observation template with text and images.
        
        Args:
            template: The text part of the observation
            contents: A list of content objects to include in the observation (str, int, Image.Image, ...)
            replace_keys: The key to replace in the observation template

        This function will:
        1. Create placeholders in the text for each image
        2. Add the images to the multi_modal_observation dictionary
        3. Set the observation_template with image placeholders
        """
        # Start with the input text
        result_template = template
        assert len(replace_keys) == len(contents), "replace_keys and contents must have the same length"
        
        # Add each image and insert placeholder
        for i, content in enumerate(contents):

            assert replace_keys[i] in result_template, f"replace_keys[{i}] must be in text"

            if not isinstance(content, Image.Image):
                # transform content to string
                result_template = result_template.replace(replace_keys[i], str(content))
                continue

            # Create the placeholder for this image
            placeholder = self.placeholder_format.format(index=i)
                
            # Add the image to multi_modal_observation dict
            self.multi_modal_observation[placeholder] = content
            
            # Replace keys in text with placeholders or append to the end if no keys
            result_template = result_template.replace(replace_keys[i], placeholder)
                
        # Set the observation template
        self.observation_template = result_template
    
    @classmethod
    def merge_observation(cls, observation_list: List["EnvObservation"], 
                          placeholder_format: Optional[str] = None) -> "EnvObservation":
        """
        Merge a list of observations into a single observation.
        
        Args:
            observation_list: List of EnvObservation objects to merge
            placeholder_format: Optional format for placeholders in the merged observation
                               (defaults to first observation's format)
                               
        Returns:
            A new EnvObservation with combined template and observations
        """
        if not observation_list:
            return cls()
            
        # Use provided placeholder format or the first observation's format
        if placeholder_format is None and observation_list:
            placeholder_format = observation_list[0].placeholder_format
            
        merged_observation = cls(placeholder_format=placeholder_format)
        merged_template = ""
        
        # Placeholder mapping from old placeholders to new ones
        placeholder_mapping = {}
        image_counter = 0
        
        # First pass: collect all templates and build mapping
        for obs_idx, obs in enumerate(observation_list):
            # Extract all placeholders from this observation's template that are also in multi_modal_observation
            # This ensures we only process actual image placeholders that have corresponding images
            placeholders = [p for p in obs.multi_modal_observation.keys() 
                           if p in obs.observation_template]
            
            # Create mapping for each placeholder in this specific observation
            for old_placeholder in placeholders:
                mapping_key = (obs_idx, old_placeholder)
                new_placeholder = merged_observation.placeholder_format.format(index=image_counter)
                placeholder_mapping[mapping_key] = new_placeholder
                image_counter += 1
        
        # Second pass: build the new template and add observations
        for obs_idx, obs in enumerate(observation_list):
            temp_template = obs.observation_template
            
            # Get all placeholders for this observation
            placeholders = [p for p in obs.multi_modal_observation.keys() if p in obs.observation_template]
            
            # Replace all placeholders with their new versions
            for old_placeholder in placeholders:
                mapping_key = (obs_idx, old_placeholder)
                if mapping_key in placeholder_mapping:
                    new_placeholder = placeholder_mapping[mapping_key]
                    
                    # Add the image to the merged observation with new placeholder
                    merged_observation.multi_modal_observation[new_placeholder] = obs.multi_modal_observation[old_placeholder]
                    
                    # Replace placeholder in template
                    temp_template = temp_template.replace(old_placeholder, new_placeholder)
            
            # Append this template to the merged one
            if merged_template and temp_template:
                merged_template += "\n" + temp_template
            else:
                merged_template += temp_template
        
        merged_observation.observation_template = merged_template
        return merged_observation


@dataclass
class EnvFeedbackSingleStep:
    """
    Dataclass for managing environment feedback for a single step.
    
    This is used when multiple actions can be taken in a single environment step.
    
    Attributes:
        env_observation: Observation from the environment after taking the action.
        step_action_str: String representation of the action taken.
        step_reward: Reward received for taking the action.
        step_done: Flag indicating if the episode is done after this step.
        step_info: Additional information about the step.
    """
    step_observation: EnvObservation = field(default_factory=EnvObservation)
    step_reward: float = 0.0
    step_done: bool = False
    step_info: Dict[str, Any] = field(default_factory=dict)
    
    def is_terminal(self) -> bool:
        """Check if this step resulted in a terminal state."""
        return self.step_done


@dataclass
class EnvFeedback:
    """
    Dataclass for managing environment feedback across multiple steps.
    
    Attributes:
        env_feedbacks: List of individual step feedbacks.
        info: Additional information about the overall feedback.
    """
    env_feedbacks: List[EnvFeedbackSingleStep] = field(default_factory=list)
    llm_raw_response: str = ""

    @property
    def observation(self) -> EnvObservation:
        """Get the merged observation from all steps."""
        return EnvObservation.merge_observation([feedback.step_observation for feedback in self.env_feedbacks])
    
    @property
    def reward(self) -> float:
        """Get the total reward from all steps."""
        return sum([feedback.step_reward for feedback in self.env_feedbacks])

    @property
    def reward_per_step(self) -> List[float]:
        """Get the reward from all steps."""
        return [feedback.step_reward for feedback in self.env_feedbacks]

    @property
    def done(self) -> List[bool]:
        """Check if any step resulted in a terminal state."""
        return any(feedback.is_terminal() for feedback in self.env_feedbacks)
    
    @property
    def info(self) -> Dict[str, Any]:
        """Get the info from all steps."""
        llm_raw_response = self.llm_raw_response
        merged_info = []
        for feedback in self.env_feedbacks:
            merged_info.append(feedback.step_info)
        return {
            'llm_raw_response': llm_raw_response,
            'info_each_step': merged_info,
        }
    
    def add_step(self, step: EnvFeedbackSingleStep) -> None:
        """Add a new step feedback to the collection."""
        self.env_feedbacks.append(step)
    
    def get_last_observation(self) -> Optional[EnvObservation]:
        """Get the observation from the last step, if available."""
        if not self.env_feedbacks:
            return None
        return self.env_feedbacks[-1].env_observation


@dataclass
class EnvConfig:
    """
    Dataclass for managing environment configuration.
    """
    env_name: str
    env_config: Dict[str, Any]
    seed: int


# ===============================
# Base Env
# ===============================
class BaseEnv(ABC):
    """
    Abstract base class for all environments.
    The class needs to be consistent with gymnasium interface.
    Should inplement: step, reset, 

    """

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Any:
        """
        Reset the environment.
        NOTE: the environment should be same for the same seed
        Args:
            seed: Seed for the environment
            
        Returns:
            rendered environment
        """

    @abstractmethod
    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: Action to take, must be in action space, or default invalid action
            
        Returns:
            observation (rendered environment), reward, done, info
        """

    @abstractmethod
    def success(self) -> bool:
        """Check if the current environment is successful."""

    @abstractmethod
    def finished(self) -> bool:
        """Check if the current environment is finished."""

    @abstractmethod
    def render(self, mode: str = 'text') -> Any:
        """Render the environment."""

    @abstractmethod
    def copy(self) -> 'BaseEnv':
        """Create a deep copy of the environment."""

    @abstractmethod
    def close(self):
        """Close the environment."""




# ===============================
# Base Game
# ===============================
class BaseGame(ABC):
    """
    Abstract base class for all games. Main function: step() and reset()
    - step() Workflow:
        1. Raw input --> Preprocess --> [Action];
        2. [Action] --> Env.step --> Feedback;
        3. Feedback --> Postprocess --> Observation;
    - reset() function: reset environment; return initial observation (instruction)
    """

    PROMPT_TEMPLATE = PromptTemplate()

    def __init__(self, **env_config):
        self.env_config = env_config
        self.traj_reward = 0

    @classmethod
    def get_task_instruction(cls) -> str:
        """Get the initial instruction for the environment."""
        return cls.PROMPT_TEMPLATE.get_task_instruction()

    @staticmethod
    def convert_numpy_to_PIL(numpy_array: np.ndarray) -> Image.Image:
        """Convert a numpy array to a PIL RGB image."""
        if numpy_array.shape[-1] == 3:
            # Convert numpy array to RGB PIL Image
            return Image.fromarray(numpy_array, mode='RGB')
        else:
            raise ValueError(f"Unsupported number of channels: {numpy_array.shape[-1]}. Expected 3 (RGB).")

    @abstractmethod
    def _preprocess(self, text: str) -> Dict:
        """Preprocess the raw text from LLM to action space."""

    @abstractmethod
    def _postprocess(self, step_result: Dict) -> EnvFeedback:
        """Postprocess the observation from environment to feedback for LLM."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> EnvFeedback:
        """Reset the environment."""

    @abstractmethod
    def step(self, action: List) -> List[EnvFeedback]:
        """Execute one step in the environment."""

    @abstractmethod
    def finished(self) -> bool:
        """Check if the current environment is finished."""
    
    @abstractmethod
    def success(self) -> bool:
        """Check if the current environment is successful."""

    @abstractmethod
    def close(self):
        """Close the environment."""

    @classmethod
    def name_repr(cls) -> str:
        """Get the name of the environment."""
        return cls.__name__
    
    @classmethod
    @abstractmethod
    def config_repr(cls, config: Dict) -> str:
        """Get the config of the environment."""
    
    def get_traj_reward(self) -> float:
        """Get the reward of the environment."""
        return self.traj_reward