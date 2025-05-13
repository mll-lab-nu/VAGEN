from vagen.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass, fields, field

@dataclass
class SokobanServiceConfig(BaseServiceConfig):
    use_state_reward: bool = False