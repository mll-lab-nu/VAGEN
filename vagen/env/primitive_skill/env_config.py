from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, fields,field
from typing import Optional, List, Union

@dataclass
class PrimitiveSkillEnvConfig(BaseEnvConfig):
    env_id: str = "AlignTwoCube" # AlignTwoCube,PlaceTwoCube,PutAppleInDrawer,StackThreeCube
    render_mode: str = "vision" # vision, text
    max_actions_per_step: int = 2
    action_sep:str= field(default='|')
    record_video: bool = field(default=False)
    video_record_dir: str = field(default='./test')
    mask_success: bool = field(default=False)
    
    def config_id(self) -> str:
        id_fields=["env_id","render_mode","max_actions_per_step"]
        id_str = ",".join([f"{field.name}={getattr(self, field.name)}" for field in fields(self) if field.name in id_fields])
        return f"PrimitiveSkillEnvConfig({id_str})"

if __name__ == "__main__":
    config = PrimitiveSkillEnvConfig()
    print(config)
    print(config.config_id())