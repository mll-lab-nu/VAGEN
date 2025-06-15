from vagen.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass, fields,field
from typing import Dict, List, Tuple, Optional, Any, Union

@dataclass
class FrozenLakeServiceConfig(BaseServiceConfig):
    device: Dict[str, Any] = field(default_factory=lambda: {"clip": 0})
    use_state_reward: bool = False
    top_strings_m: int = 1000
    top_strings_k: int = 5