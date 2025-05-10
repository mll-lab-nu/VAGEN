from vagen.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass, fields,field

@dataclass
class PrimitiveSkillServiceConfig(BaseServiceConfig):
    max_process_workers: int = field(default=8)
    max_thread_workers: int = field(default=4)
    devices: list = field(default_factory=lambda: [0,1])
    spawn_method: str = field(default="fork")
    timeout: int = field(default=120)
    use_state_reward: bool = False