from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ManiSkillEnvConfig:
    """Configuration for the ManiSkill primitive-skill environment."""

    # Task: AlignTwoCube, PlaceTwoCube, PutAppleInDrawer, StackThreeCube
    env_id: str = "AlignTwoCube"
    render_mode: str = "vision"  # "vision" or "text"
    max_actions_per_step: int = 2
    action_sep: str = "|"
    image_placeholder: str = "<image>"

    # Prompt
    prompt_format: str = "free_think"  # "free_think", "wm", "free_wm"

    # Reward
    format_reward: float = 0.1
    success_reward: float = 10.0
    step_penalty: float = 0.1

    # Env internals
    mask_success: bool = True
    record_video: bool = False
    video_record_dir: str = "./test"
