from abc import ABC, abstractmethod
import re
from typing import Optional, List, Tuple, Any, Dict
from copy import deepcopy
from transformers import AutoTokenizer
import torch
from PIL import Image
import numpy as np
from dataclasses import dataclass, field
from .utils.io_utils import validate_reset_io,validate_step_io
           
class BaseInterface(ABC):
    image_placeholder="<image>"
    
    def __init__(self, config: Dict):
        self.config = config
    
    @classmethod
    @abstractmethod
    def config_repr(cls, config) -> str:
        """convert config to str"""
        pass
    
    @abstractmethod
    def close(self):
        """Close the environment."""
        pass
    
    @abstractmethod
    def get_task_instruction(self) -> str:
        """Get the task instruction."""
        pass
    
    @abstractmethod
    @validate_step_io
    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        pass
    
    @abstractmethod
    @validate_reset_io    
    def reset(self, seed: int) -> Tuple[Dict, Dict]:
        """Reset the environment."""
        pass
    
    @abstractmethod
    def get_traj_reward(self) -> float:
        """Get the reward of the environment."""
    

