from vagen.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass, fields, field

@dataclass
class SokobanServiceConfig(BaseServiceConfig):
    use_state_reward: bool = False
    top_strings_m: int = 1000
    top_strings_k: int = 5