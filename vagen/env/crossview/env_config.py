from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, fields,field
from typing import Optional, List, Union

@dataclass
class CrossViewEnvConfig(BaseEnvConfig):
    type: str = "ICLRL" #baseRL, ICLRL, reasoning, cogmap,cogmap_and_reasoning,base_sft_no_think,base_sft_think
    reward_type: str = 'base' # format-only, answer-only, format-and-answer
    split: str = "train" # train, test
    image_path: str = "MindCube/extracted_images"
    image_size: tuple = (300, 300)
    def config_id(self) -> str:
        return f"CrossViewQAEnv"
    def generate_seeds(self, size, seed=0, n_candidate = 20000):
        return [i for i in range(size)]
            
            
            
if __name__ == "__main__":
    config = CrossViewEnvConfig()
    print(config.config_id())