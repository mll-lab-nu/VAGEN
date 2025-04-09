from vagen.env.base_config import BaseConfig
from dataclasses import dataclass, fields, field
from typing import Optional, List, Union, Dict

@dataclass
class SVGConfig(BaseConfig):
    """Configuration for the SVG environment"""
    dataset_name: str = "starvector/svg-emoji-simple"
    data_dir: str = "vagen/env/svg/data"
    seed: int = 16
    split: str = "train"
    # Score configuration
    model_size: str = "small"  # 'small', 'base', or 'large'
    dino_only: bool = False
    dino_weight: Optional[float] = None
    structural_weight: Optional[float] = None
    color_weight: Optional[float] = None
    code_weight: Optional[float] = None
    # Reward configuration
    format_reward: float = 0.5
    format_penalty: float = 0.0
    # Analysis mode for logging
    analysis_mode: bool = False
    
    def config_id(self) -> str:
        """Generate a unique identifier for this configuration"""
        id_fields = [
            "dataset_name", 
            "model_size", 
            "dino_only", 
            "format_reward", 
            "format_penalty"
        ]
        
        id_str = ",".join([f"{field.name}={getattr(self, field.name)}" 
                          for field in fields(self) 
                          if field.name in id_fields])
        
        # Add optional fields if they're set
        optional_fields = ["dino_weight", "structural_weight", "color_weight", "code_weight"]
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                id_str += f",{field_name}={value}"
                
        return f"SVGConfig({id_str})"
    
    def get_score_config(self) -> Dict:
        """Get the score configuration dictionary"""
        score_config = {
            "model_size": self.model_size,
            "dino_only": self.dino_only,
        }
        
        # Add optional weights if set
        if self.dino_weight is not None:
            score_config["dino_weight"] = self.dino_weight
        if self.structural_weight is not None:
            score_config["structural_weight"] = self.structural_weight
        if self.color_weight is not None:
            score_config["color_weight"] = self.color_weight
        if self.code_weight is not None:
            score_config["code_weight"] = self.code_weight
            
        return score_config


if __name__ == "__main__":
    # Example usage
    config = SVGConfig(
        dataset_name="starvector/svg-emoji-simple",
        data_dir="data/svg",
        model_size="small",
        dino_only=False,
        dino_weight=5.0
    )
    
    print(config.config_id())
    print(config.get_score_config())