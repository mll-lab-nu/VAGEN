from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, field, fields

@dataclass
class NavigationEnvConfig(BaseEnvConfig):
    """Configuration class for the Navigation environment."""
    env_name: str = "navigation"
    resolution: int = 255
    eval_set: str = 'base'
    down_sample_ratio: float = 1.0
    fov: int = 100
    multiview: bool = False
    render_mode: str= 'vision'
    max_actions_per_step: int = 5
    max_action_penalty: float = -0.1
    format_reward: float = 0.5
    gpu_device: int = 0
    prompt_format: str = "grounding_worldmodeling" 
    success_threshold: float = 1.5
    step_length: float = 0.5
    # "free_think", "no_think", "grounding", "worldmodeling", "grounding_worldmodeling"
    
    
    # configs for process reward for grounding and world modeling
    max_objects_in_state: int = 5
    use_state_reward: bool = False
    grounding_reward_weight: float = 0.5
    worldmodeling_reward_weight: float = 0.5

    def config_id(self) -> str:
        """Generate a unique identifier for this configuration."""
        id_fields = ["eval_set","render_mode", "max_actions_per_step"]
        id_str = ",".join([f"{field.name}={getattr(self, field.name)}" for field in fields(self) if field.name in id_fields])
        return f"NavigationEnvConfig({id_str})"

if __name__ == "__main__":
    config = NavigationEnvConfig()
    print(config.config_id())