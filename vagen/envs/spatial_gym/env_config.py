from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from omegaconf import ListConfig, OmegaConf
import os
import json
import numpy as np

from .evaluation.task_types import EvalTaskType

@dataclass
class SpatialGymConfig:
    """
    Configuration for the SpatialGym environment.
    
    Parameters:
        exp_type: Exploration type ('active')
        perspective: Perspective of exploration ('ego' or 'allo')
        eval_tasks: List of evaluation tasks with their configurations
        max_exp_steps: Maximum exploration steps for active exploration
        render_mode: Rendering mode ('vision')
    """

    # common configuration (Moved from BaseEnvConfig)
    env_name: str = field(default="SpatialGym", init=False)
    max_actions_per_step: int = field(default=1, init=False)    
    prompt_format: str = field(default="free_think", init=False)
    action_sep: str = field(default="|", init=False)
    image_size: Tuple[int, int] = field(default=(384, 384), init=False)
    format_reward: bool = True  # Added default
    special_token_list: List[str] = field(default_factory=list) # Added default
    image_placeholder: str = "<image>" # Added default
    
    # Environment specific configuration
    name: str = 'unnamed_env'
    render_mode: str = field(default="vision")
    replay: bool = False
    # Room configuration (minimal additions from RAGEN)
    room_config: Dict[str, Any] = field(default_factory=lambda: {"room_size": [10, 10], "n_objects": 3, "room_num": 1, "topology": 0})

    # Field of view and base directory
    field_of_view: int = field(default=90, init=False)
    data_dir: str = os.path.join(os.path.dirname(__file__), "room_data/")
    # Exploration configuration
    exp_type: str = 'passive'
    perspective: str = 'ego'
    max_exp_steps: int = 20
    kwargs: Dict = None
    proxy_agent: str = 'scout'
    # Evaluation configuration
    # Each eval task entry supports {task_type, task_kwargs}
    eval_tasks: List[Dict[str, Any]] = field(default_factory=lambda: [{"task_type": "rot", "task_kwargs": {}}])

    prompt_config: Dict[str, Any] = field(default_factory=lambda: {})

    calculate_information_gain: bool = True
    
    def config_id(self) -> str:
        eval_task_str = ", ".join([f"{task['task_type']}" for task in self.eval_tasks])
        return f"SpatialGymConfig(mode={self.render_mode},format={self.prompt_format},eval_tasks={eval_task_str})"

    def generate_seeds(self, size):
        ks = self.kwargs or {}
        start = int(ks.get('seed_start', 0))
        end = ks.get('seed_end')
        if end is None:
            return [start + i for i in range(size)]
        end = int(end)
        count = max(0, min(size, end - start + 1))
        return [start + i for i in range(count)]

    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_exp_type()
        self._validate_field_of_view()
        self._validate_eval_tasks()

    def _validate_exp_type(self):
        """Validate exp_type parameter."""
        valid_exp_types = ["passive", "active"]
        if self.exp_type not in valid_exp_types:
            raise ValueError(f"exp_type must be one of {valid_exp_types}")

    def _validate_field_of_view(self):
        """Validate field_of_view parameter."""
        assert self.field_of_view == 90, "Field of view must be 90 for spatial gym"

    def _validate_eval_tasks(self):
        """Validate eval_tasks parameter."""
        valid_eval_tasks = EvalTaskType.get_short_names()

        # assert len(self.eval_tasks) == 1, "Only one evaluation task is supported"

        if isinstance(self.eval_tasks, ListConfig):
            self.eval_tasks = OmegaConf.to_container(self.eval_tasks, resolve=True)

        if isinstance(self.eval_tasks, np.ndarray):
            self.eval_tasks = self.eval_tasks.tolist()

        
        if not self.eval_tasks:
            raise ValueError("eval_tasks must be non-empty")
        
        for i, task in enumerate(self.eval_tasks):
            if not isinstance(task, dict) or 'task_type' not in task:
                raise ValueError("Each eval_task must be a dict with 'task_type' key")
            
            task_type = task['task_type']
            if task_type not in valid_eval_tasks:
                raise ValueError(f"task_type '{task_type}' must be one of {valid_eval_tasks}")
            # validate task_kwargs if present
            if 'task_kwargs' in task and task['task_kwargs'] is not None:
                assert isinstance(task['task_kwargs'], dict), "task_kwargs must be a dict"

    def get_room_config(self) -> Dict[str, Any]:
        """Get configuration for room generation."""
        return self.room_config
    
    def get_observation_config(self) -> Dict[str, Any]:
        return {
            'field_of_view': self.field_of_view,
            'prompt_config': self.prompt_config,
            'render_mode': self.render_mode,
            'exp_type': self.exp_type,
            'proxy_agent': self.proxy_agent,
        }

    def get_model_config(self) -> Dict[str, Any]:
        ks = self.kwargs or {}
        return ks.get('model_config') or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'name': self.name,
            'room_config': self.room_config,
            'exp_type': self.exp_type,
            'perspective': self.perspective,
            'eval_tasks': self.eval_tasks,
            'max_exp_steps': self.max_exp_steps,
            'calculate_information_gain': self.calculate_information_gain,
            'image_size': self.image_size,
            'prompt_config': self.prompt_config,
            'field_of_view': self.field_of_view,
            'format_reward': self.format_reward,
            'special_token_list': self.special_token_list,
            'image_placeholder': self.image_placeholder,
        }
    



if __name__ == "__main__":
    config = SpatialGymConfig(eval_tasks=[{"task_type": "rot"}])
    print(config)
    print(config.to_dict())
    print(config.config_id())
    print(config.eval_tasks)