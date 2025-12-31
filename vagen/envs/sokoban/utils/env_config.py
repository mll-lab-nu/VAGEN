
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SokobanEnvConfig:
    """Configuration for Sokoban environment"""
    dim_room: Tuple[int, int] = (6, 6)  # Room dimensions (height, width)
    max_steps: int = 100      # Maximum steps per episode
    num_boxes: int = 1        # Number of boxes in the room
    render_mode: str = "text" # "text" or "vision"
    max_actions_per_step: int = 3  # Max actions per step
    action_sep: str = ","     # Separator between actions
    image_placeholder: str = "<image>"  # Placeholder for vision mode

    # Map generation constraints
    min_solution_steps: Optional[Tuple[int, int]] = None  # (min, max) range for solution steps
    reset_seed_max_tries: int = 10000  # Max tries to find a valid seed
    min_solution_bfs_max_depth: int = 200  # Max BFS depth for solution
    prompt_format: str = "wm"  # "free_think" or "wm"
    format_reward: float = 0.1  # Reward for following the format correctly