from vagen.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass, fields,field
from typing import Dict, List, Tuple, Optional, Any, Union

@dataclass
class FrozenLakeServiceConfig(BaseServiceConfig):
    preload_reward_model: Dict[str, Any] = field(default_factory=lambda: {"clip": False})
    device: Dict[str, Any] = field(default_factory=lambda: {"clip": 0})