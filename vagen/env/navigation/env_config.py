from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, field, fields

@dataclass
class NavigationEnvConfig(BaseEnvConfig):
    """Configuration class for the Navigation environment."""
    resolution: int = 300
    eval_set: str = 'base'
    exp_name: str = 'test_base'
    down_sample_ratio: float = 1.0
    fov: int = 100
    multiview: bool = False
    render_mode: str= 'vision'
    max_actions_per_step: int = 10
    max_action_penalty: float = -0.1
    format_reward: float = 0.5
    gpu_device: int = 0

    def config_id(self) -> str:
        """Generate a unique identifier for this configuration."""
        id_fields = ["resolution", "eval_set", "exp_name", "down_sample_ratio", 
                    "fov", "multiview", "render_mode", "max_actions_per_step"]
        id_str = ",".join([f"{field.name}={getattr(self, field.name)}" for field in fields(self) if field.name in id_fields])
        return f"NavigationEnvConfig({id_str})"

if __name__ == "__main__":
    config = NavigationEnvConfig()
    print(config.config_id())