from dataclasses import dataclass


@dataclass
class SpatialGymServiceConfig:
    """Configuration for SpatialGym service."""
    use_state_reward: bool = False