from vagen.env.base_service_config import BaseServiceConfig
from dataclasses import dataclass, fields,field

@dataclass
class FrozenLakeServiceConfig(BaseServiceConfig):
    pass