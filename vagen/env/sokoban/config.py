from vagen.env.base_config import BaseConfig
from dataclasses import dataclass, field, fields
from .utils import generate_seeds
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
        id_str = ",".join([f"{field.name}={getattr(self, field.name)}" for field in fields(self) if field.name in id_fields])
        return f"SokobanConfig({id_str})"

    def generate_seeds(self,size,seed=0,n_candidate: int = 20000,) -> list:
        return generate_seeds(size=size, 
                              config=self,
                              min_actions_to_succeed=self.min_actions_to_succeed,
                              seed=seed,
                              n_candidate=n_candidate,)

        
    
    
if __name__ == "__main__":
    config = SokobanConfig()
    print(config.config_id())
    print(config.generate_seeds(10))
   