"""
Dataclass for inference configuration.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import os
from pathlib import Path

@dataclass
class InferenceConfig:
    """Configuration for inference runs."""
    
    # Server parameters
    server_url: str = "http://localhost:5000"
    server_timeout: int = 600
    server_max_workers: int = 48
    
    # Inference parameters
    batch_size: int = 32
    max_steps: int = 10
    split: str = "test"
    debug: bool = False
    
    # Output configuration
    output_dir: str = "inference_outputs"
    
    # WandB configuration
    use_wandb: bool = True
    wandb_project: str = "vagen-inference"
    wandb_entity: Optional[str] = None
    
    # Display settings
    show_progress: bool = True
    val_generations_to_log_to_wandb: int = 10
    
    # Alternative inference modes
    skip_generation: bool = False
    use_cached_results: bool = False
    cached_results_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        # Create output directory if it doesn't exist
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'InferenceConfig':
        """Create a configuration instance from a dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'InferenceConfig':
        """Load configuration from a YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to a YAML file."""
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)