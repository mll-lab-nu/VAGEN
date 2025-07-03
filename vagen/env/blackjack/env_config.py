from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, fields
from typing import Optional, List

@dataclass
class BlackjackEnvConfig(BaseEnvConfig):
    env_name: str = "blackjack"
    natural: bool = False  # Whether to give extra reward for natural blackjack
    sab: bool = False  # Whether to follow Sutton & Barto rules exactly
    render_mode: str = "vision"  # "text" or "vision"
    max_actions_per_step: int = 1  # Typically one action at a time in Blackjack
    prompt_format: str = "free_think"
    
    # configs for process reward (if needed in future)
    use_state_reward: bool = False
    
    def config_id(self) -> str:
        id_fields = [
            "natural", 
            "sab", 
            "render_mode", 
            "max_actions_per_step",
            "format_reward"
        ]
        id_str = ",".join([
            f"{field.name}={getattr(self, field.name)}" 
            for field in fields(self) 
            if field.name in id_fields
        ])
        return f"BlackjackEnvConfig({id_str})"

if __name__ == "__main__":
    config = BlackjackEnvConfig()
    print(config.config_id())