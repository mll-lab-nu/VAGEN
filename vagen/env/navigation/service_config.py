from vagen.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass, fields,field

@dataclass
class NavigationServiceConfig(BaseServiceConfig):
    devices: list = field(default_factory=lambda: [0])