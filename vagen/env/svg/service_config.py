from vagen.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass, fields,field

@dataclass
class SVGServiceConfig(BaseServiceConfig):
    model_size="small"