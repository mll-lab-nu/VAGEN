from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, field, fields
from typing import Optional, List
import os

@dataclass
class ALFWorldEnvConfig(BaseEnvConfig):
    """Configuration class for the ALFWorld environment."""
    alf_config_path: str = "path/to/alf-config.yaml"
    max_actions_per_step: int = 1
    action_only_prompt: bool = False
    render_mode: str = "text"  # @TODO Only "text" mode is supported for now

    def __post_init__(self):
        # Expand any ${env:VAR} or $VAR references in the path
        raw = self.alf_config_path
        # Convert Hydra-style ${env:VAR} to shell-style $VAR
        raw = raw.replace('${env:', '$').replace('}', '')
        # Expand environment variables
        self.alf_config_path = os.path.expandvars(raw)

    def config_id(self) -> str:
        """Generate a unique identifier for this configuration."""
        id_fields = ["alf_config_path", "render_mode", "action_only_prompt", "max_actions_per_step"]
        id_str = ",".join([
            f"{field.name}={getattr(self, field.name)}"
            for field in fields(self)
            if field.name in id_fields
        ])
        return f"ALFWorldEnvConfig({id_str})"
