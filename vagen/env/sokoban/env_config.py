from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, field, fields
from .utils import generate_seeds
@dataclass
class SokobanEnvConfig(BaseEnvConfig):
    env_name: str = "sokoban"
    dim_room: tuple = (6, 6)
    max_steps: int = 100
    num_boxes: int = 1
    render_mode: str = "vision" # "vision" or "text"
    min_actions_to_succeed: int = 5
    max_actions_per_step: int = 3
    prompt_format: str = "grounding_worldmodeling" 
    # "free_think", "no_think", "grounding", "worldmodeling", "grounding_worldmodeling"
    # "grounding_symbolic", "worldmodeling_symbolic", "grounding_worldmodeling_symbolic"
    # "grounding_structured", "worldmodeling_structured", "grounding_worldmodeling_structured"
    
    
    # configs for process reward for grounding and world modeling
    use_state_reward: bool = False
    grounding_reward_weight: float = 0.5
    worldmodeling_reward_weight: float = 0.5
    
    def config_id(self) -> str:
        id_fields = ["dim_room", "max_steps", "num_boxes", "render_mode", "min_actions_to_succeed", "max_actions_per_step"]
        id_str = ",".join([f"{field.name}={getattr(self, field.name)}" for field in fields(self) if field.name in id_fields])
        return f"SokobanEnvConfig({id_str})"

    def generate_seeds(self,size,seed=0,n_candidate: int = 20000,) -> list:
        return generate_seeds(size=size, 
                              config=self,
                              min_actions_to_succeed=self.min_actions_to_succeed,
                              seed=seed,
                              n_candidate=n_candidate,)
        

        
    
    
if __name__ == "__main__":
    config = SokobanEnvConfig()
    print(config.config_id())
    print(config.generate_seeds(10))
   