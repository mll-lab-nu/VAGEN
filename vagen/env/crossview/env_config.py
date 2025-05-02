from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, fields,field
from typing import Optional, List, Union
@dataclass
class CrossViewQAEnvConfig(BaseEnvConfig):
    data_path: str = "crossviewQA_train_qwenformat_singleletter.json"
    image_dir: str = "extracted_images"
    image_size: tuple = (300, 300)
    
    def config_id(self) -> str:
        return f"CrossViewQAEnv"