from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, fields,field
from typing import Optional, List, Union

@dataclass
class ManipulationEnvConfig(BaseEnvConfig):
    env_id: str = "AlignTwoCube" # AlignTwoCube,PlaceTwoCube,PutAppleInDrawer,StackThreeCube
    render_mode: str = "vision" # vision, text
    
    def config_id(self) -> str:
        id_fields=["env_id","render_mode"]
        id_str = ",".join([f"{field.name}={getattr(self, field.name)}" for field in fields(self) if field.name in id_fields])
        return f"ManipulationEnvConfig({id_str})"

if __name__ == "__main__":
    config = ManipulationEnvConfig()
    print(config.config_id())