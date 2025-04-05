from vagen.env_new.base_config import BaseConfig
from dataclasses import dataclass, field

@dataclass
class SokobanConfig(BaseConfig):
    dim_room: tuple = (6, 6)
    max_steps: int = 100
    num_boxes: int = 1
    render_mode: str = "vision"
    min_actions_to_succeed: int = 5
    max_actions_per_step: int = 3
    format_reward = 0.5
    
    def config_id(self) -> str:
        return str(self)

    
    
if __name__ == "__main__":
    config = SokobanConfig()
    print(config)
   