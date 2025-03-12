from abc import ABC, abstractmethod
import re
from typing import Optional, List, Tuple, Any, Dict
from copy import deepcopy
from transformers import AutoTokenizer
import torch
from PIL import Image
import numpy as np
from dataclasses import dataclass, field

@dataclass
class EnvConfig:
    """
    Dataclass for managing environment configuration.
    """
    env_name: str
    env_config: Dict[str, Any]
    seed: int

class BaseEnv(ABC):
    @abstractmethod
    def _reset(self, seed: Optional[int] = None) -> Any:
        """
        Reset the environment.
        NOTE: the environment should be same for the same seed
        Args:
            seed: Seed for the environment
            
        Returns:
            rendered environment
        """
        pass
    
    @abstractmethod
    def _step(self, action) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: Action to take, must be in action space, or default invalid action
            
        Returns:
            observation (rendered environment), reward, done, info
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close the environment."""
        pass
    
    
    def step(self, action:Any) -> Tuple[Any, Any, Any, Any]:
        """
        Execute one step in the environment.
        Args:
            action: Action to take, must be in action space, or default invalid action
            
        Returns:
            observation (rendered environment), reward, done, info
        """
        obs,reward,done,info = self._step(action)
        return obs, reward, done, info
    
    def reset(self, seed: Optional[int] = None) -> Any:
        """
        Reset the environment.
        NOTE: the environment should be same for the same seed
        Args:
            seed: Seed for the environment  
        Returns:
            obs,info
        """
        obs,info = self._reset(seed)
        return obs,info
    
        
class BaseInterface(ABC):
    def __init__(self, **env_config):
        self.env_config = env_config
        
    @classmethod
    def name_repr(cls) -> str:
        """Get the name of the environment."""
        return cls.__name__
        
    @abstractmethod
    def _reset(self, seed: Optional[int] = None) -> Tuple[Any, float, bool, Dict]:
        """Reset the environment."""
        pass
    
    @abstractmethod
    def _step(self, action:str) -> Tuple[Any, float, bool, Dict]:
        """Execute action string in the environment."""
        # return observation, reward, done, info
        # info must contain "llm_raw_response" key, which is a string
        pass
    
    @classmethod
    @abstractmethod
    def config_repr(cls, config: Dict) -> str:
        """Get the config of the environment."""
        pass
    
    
    @abstractmethod
    def close(self):
        """Close the environment."""
        pass
    
    @abstractmethod
    def get_task_instruction(self) -> str:
        """Get the task instruction."""
        pass
    
    
    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        """Execute action string in the environment."""
        """Please use the following assertions to validate the output, 
        then you can rewrite the step in your own class to improve the performance"""
        
        
        assert isinstance(action, str), f"action must be str, got {type(action)}"
        obs,reward,done,info = self._step(action)
        assert isinstance(reward, (int, float)), f"reward must be int or float, got {type(reward)}"
        assert isinstance(done, bool), f"done must be bool, got {type(done)}"
        assert isinstance(info, dict), f"info must be dict, got {type(info)}"
        assert isinstance(obs, dict), f"obs must be dict, got {type(obs)}"
        assert "llm_raw_response" in info, f"info must contain 'llm_raw_response' key"
        assert isinstance(info["llm_raw_response"], str), f"info['llm_raw_response'] must be str, got {type(info['llm_raw_response'])}"
        assert "text_template" in obs, f"obs must contain 'text_template' key"
        assert isinstance(obs["text_template"], str), f"obs['text_template'] must be str, got {type(obs['text_template'])}"
        
        if "multi_modal_data" in obs:
            for key, image in obs["multi_modal_data"].items():
                assert isinstance(image, Image.Image), f"image must be PIL.Image.Image, got {type(image)}"
        return obs, reward, done, info
    
            
    def reset(self, seed: int):
        """Reset the environment."""
        assert isinstance(seed, int), f"seed must be int, got {type(seed)}"
        obs, info = self._reset(seed)
        assert isinstance(info, dict), f"info must be dict, got {type(info)}"
        assert isinstance(obs, dict), f"obs must be dict, got {type(obs)}"
        assert "llm_raw_response" in info, f"info must contain 'llm_raw_response' key"
        assert isinstance(info["llm_raw_response"], str), f"info['llm_raw_response'] must be str, got {type(info['llm_raw_response'])}"
        assert "text_template" in obs, f"obs must contain 'text_template' key"
        assert isinstance(obs["text_template"], str), f"obs['text_template'] must be str, got {type(obs['text_template'])}"
        
        if "multi_modal_data" in obs:
            for key, image in obs["multi_modal_data"].items():
                assert isinstance(image, Image.Image), f"image must be PIL.Image.Image, got {type(image)}"
        return obs, info
    
    def get_traj_reward(self) -> float:
        """Get the reward of the environment."""
        return self.traj_reward
    





def preprocess_text(text: str) -> dict:
    """Preprocess the raw text from llm to a list of strings

    1. Extract think from the first <think> ... </think>
    2. Extract answer from the first <answer> ... </answer>
    3. Split the answer by comma into a list of strings
    
    Args:
        text: raw text from llm

    Returns:
        dict with keys: llm_raw_response, think, answer_list
    """
    # Extract content from <think> tags if they exist
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    
    # Extract content from <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)

    answer_list, thinking, answer_content = [], "", ""
    
    if think_match:
        thinking = think_match.group(1).strip()
    
    if answer_match:
        # Get the answer content and split by comma
        answer_content = answer_match.group(1).strip()
        # Split by comma and strip whitespace from each item
        answer_list = [item.strip() for item in answer_content.split(',') if item.strip()]
    
    return {
        'llm_raw_response': text,
        'answer_list': answer_list,
        'think': thinking,
        'answer': answer_content
    }

def convert_numpy_to_PIL(numpy_array: np.ndarray) -> Image.Image:
        """Convert a numpy array to a PIL RGB image."""
        if numpy_array.shape[-1] == 3:
            # Convert numpy array to RGB PIL Image
            return Image.fromarray(numpy_array, mode='RGB')
        else:
            raise ValueError(f"Unsupported number of channels: {numpy_array.shape[-1]}. Expected 3 (RGB).")


if __name__ == "__main__":
    text = """
    <think>
    I am thinking about the problem.
    </think>
    <answer>
    answer1, answer2, answer3
    </answer>
    """
    print(preprocess_text(text))