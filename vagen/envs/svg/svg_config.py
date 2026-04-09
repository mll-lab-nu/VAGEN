from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class SvgEnvConfig:
    """Configuration for the SVG environment."""

    # Dataset
    dataset_name: str = "starvector/svg-icons-simple"
    data_dir: str = "data"
    split: str = "train"
    seed: int = 42

    # Prompt / action
    prompt_format: str = "free_think"  # "free_think", "wm", "free_wm"
    action_sep: str = "~~"
    max_actions_per_step: int = 1
    image_placeholder: str = "<image>"

    # Scoring model
    model_size: str = "small"  # 'small', 'base', 'large'
    dino_weight: Optional[float] = None
    structural_weight: Optional[float] = None
    dreamsim_weight: Optional[float] = None
    device: Dict[str, Any] = field(
        default_factory=lambda: {"dino": 0, "dreamsim": 0}
    )

    # Reward
    format_reward: float = 0.5
    format_penalty: float = 0.0

    def __post_init__(self):
        processed = {}
        for key, value in self.device.items():
            if isinstance(value, (int, float)):
                processed[key] = f"cuda:{int(value)}"
            else:
                processed[key] = value
        self.device = processed

    def get_score_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "model_size": self.model_size,
            "device": self.device,
        }
        if self.dino_weight is not None:
            cfg["dino_weight"] = self.dino_weight
        if self.structural_weight is not None:
            cfg["structural_weight"] = self.structural_weight
        if self.dreamsim_weight is not None:
            cfg["dreamsim_weight"] = self.dreamsim_weight
        return cfg
