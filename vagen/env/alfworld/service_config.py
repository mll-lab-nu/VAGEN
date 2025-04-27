from vagen.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass, fields,field

@dataclass
class ALFWorldServiceConfig(BaseServiceConfig):
    """
    Configuration for ALFWorldService, extends BaseServiceConfig.
    Attributes:
        max_workers: Total number of process workers (inherited)
        max_thread_workers: Number of threads per process
        timeout: Timeout in seconds for inter-process commands
    """
    max_workers: int = 10
    max_thread_workers: int = 4
    timeout: int = 1200
