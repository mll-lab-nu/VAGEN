from vagen.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass, fields, field
from typing import Dict, Any

@dataclass
class SVGServiceConfig(BaseServiceConfig):
    model_size: str = "small"
    max_workers: int = 5
    # Whether to initialize models on service startup
    preload_models: bool = False
    # Configuration for different model devices
    device: Dict[str, Any] = field(default_factory=lambda: {"dino": 0, "dreamsim": 0})
    use_state_reward: bool = False