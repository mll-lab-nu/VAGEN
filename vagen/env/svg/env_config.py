from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, fields, field
from typing import Optional, List, Union, Dict, Any

@dataclass
class SvgEnvConfig(BaseEnvConfig):
    """Configuration for the SVG environment"""
    dataset_name: str = "starvector/svg-icons-simple"
    data_dir: str = "data"
    seed: int = 42
    split: str = "train"
    action_sep: str = "~~"
    # Score configuration
    model_size: str = "small"  # 'small', 'base', or 'large'
    # Weights for different scoring components
    dino_weight: Optional[float] = None
    structural_weight: Optional[float] = None
    dreamsim_weight: Optional[float] = None
    # Device configuration
    device: Dict[str, Any] = field(default_factory=lambda: {"dino": 0, "dreamsim": 0})
    # Reward configuration
    format_reward: float = 0.5
    format_penalty: float = 0.0
    # Prompt configuration
    prompt_format: str = "free_think" 
    # "free_think", "no_think", "grounding", "worldmodeling", "grounding_worldmodeling"
    use_accuracy_reward: bool = False
    
    def __post_init__(self):
        """Process device configuration after initialization"""
        # Convert device numbers to CUDA device format
        processed_device = {}
        for key, value in self.device.items():
            if isinstance(value, (int, float)):
                processed_device[key] = f"cuda:{int(value)}"
            else:
                # If it's already a string (like "cuda:1"), keep it as is
                processed_device[key] = value
        self.device = processed_device
    
    def config_id(self) -> str:
        """Generate a unique identifier for this configuration"""
        id_fields = [
            "dataset_name", 
            "model_size", 
            "format_reward", 
            "format_penalty"
        ]
        
        id_str = ",".join([f"{field.name}={getattr(self, field.name)}" 
                          for field in fields(self) 
                          if field.name in id_fields])
        
        # Add optional fields if they're set
        optional_fields = ["dino_weight", "structural_weight", "dreamsim_weight"]
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                id_str += f",{field_name}={value}"
                
        return f"SvgEnvConfig({id_str})"
    
    def get_score_config(self) -> Dict:
        """Get the score configuration dictionary"""
        score_config = {
            "model_size": self.model_size,
            "device": self.device  # Include processed device configuration in score config
        }
        
        # Add optional weights if set
        if self.dino_weight is not None:
            score_config["dino_weight"] = self.dino_weight
        if self.structural_weight is not None:
            score_config["structural_weight"] = self.structural_weight
        if self.dreamsim_weight is not None:
            score_config["dreamsim_weight"] = self.dreamsim_weight
            
        return score_config