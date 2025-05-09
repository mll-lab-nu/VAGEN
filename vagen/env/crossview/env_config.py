from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, fields,field
from typing import Optional, List, Union

@dataclass
class CrossViewEnvConfig(BaseEnvConfig):
    split:str ="train"
    image_dir: str = "extracted_images"
    image_size: tuple = (300, 300)
    render_mode: str = "vision"
    train_data_path: str = "crossviewQA_train_qwenformat_singleletter.json"
    test_data_path: str = "crossviewQA_tinybench.jsonl"
    def config_id(self) -> str:
        return f"CrossViewQAEnv"
    
    def generate_seeds(self, size, seed=0, n_candidate = 20000):
        return [i for i in range(size)]
            
            
            
if __name__ == "__main__":
    config = CrossViewEnvConfig()
    print(config.config_id())