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
    
    def config_id(self) -> str:
        id_fields = ["dim_room", "max_steps", "num_boxes", "render_mode", "min_actions_to_succeed", "max_actions_per_step"]
        id_str = ",".join([f"{field.name}={getattr(self, field.name)}" for field in field(self) if field.name in id_fields])
        return f"SokobanConfig({id_str})"

    
    
if __name__ == "__main__":
    config = SokobanConfig()
    print(config.config_id())
   