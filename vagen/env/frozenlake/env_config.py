from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, fields,field
from typing import Optional, List, Union

@dataclass
class FrozenLakeEnvConfig(BaseEnvConfig):
    env_name: str = "frozenlake"
    desc: Optional[List[str]] = None  # environment map
    is_slippery: bool = False
    size: int = 4
    p: float = 0.8  # probability of frozen tile
    render_mode: str = "vision"  # "text" or "vision"
    max_actions_per_step: int = 3
    min_actions_to_succeed: int = 5
    prompt_format: str = "grounding_worldmodeling"
    
    # configs for process reward for grounding and world modeling
    use_state_reward: bool = False
    grounding_reward_weight: float = 0.5
    worldmodeling_reward_weight: float = 0.5
    
    def config_id(self) -> str:
        id_fields=["is_slippery", "size", "p", "render_mode", "max_actions_per_step", "min_actions_to_succeed","format_reward"]
        id_str = ",".join([f"{field.name}={getattr(self, field.name)}" for field in fields(self) if field.name in id_fields])
        return f"FrozenLakeEnvConfig({id_str})"

if __name__ == "__main__":
    config = FrozenLakeEnvConfig()
    print(config.config_id())