import numpy as np
from typing import Dict, List, Optional
from PIL import Image

from vagen.envs._common.response_format import (
    canonical_free_think,
    canonical_wm,
    parse_free_think_sections,
    parse_wm_sections,
    split_actions,
)

PROMPT_FORMATS = frozenset({"wm", "free_think"})


def parse_free_think(response: str, action_sep: str = ",", max_actions: int = 3) -> Dict:
    sections = parse_free_think_sections(response)
    actions = split_actions(sections.answer, action_sep, max_actions)
    action_content = action_sep.join(actions)

    return {
        "llm_raw_response": response,
        "llm_response": canonical_free_think(sections),
        "perception_content": "",
        "reasoning_content": sections.reasoning,
        "prediction_content": "",
        "action_content": action_content,
        "actions": actions,
        "format_correct": sections.format_correct,
    }


def parse_wm(response: str, action_sep: str = ",", max_actions: int = 3) -> Dict:
    sections = parse_wm_sections(response)
    actions = split_actions(sections.answer, action_sep, max_actions)
    action_content = action_sep.join(actions)

    return {
        "llm_raw_response": response,
        "llm_response": canonical_wm(sections),
        "perception_content": sections.perception,
        "reasoning_content": sections.reasoning,
        "prediction_content": sections.prediction,
        "action_content": action_content,
        "actions": actions,
        "format_correct": sections.format_correct,
    }


def parse_response(response: str, prompt_format: str = "free_think", action_sep: str = ",", max_actions: int = 3) -> Dict:
    """Parse LLM response based on the specified prompt format"""
    if prompt_format == "free_think":
        return parse_free_think(response, action_sep, max_actions)
    elif prompt_format == "wm":
        return parse_wm(response, action_sep, max_actions)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")


def numpy_to_pil(numpy_array: np.ndarray) -> Image.Image:
    """Convert numpy (H, W, 3) to PIL.Image in RGB."""
    if numpy_array.shape[-1] == 3:
        return Image.fromarray(numpy_array.astype(np.uint8), mode="RGB")
    raise ValueError(f"Unsupported channels: {numpy_array.shape[-1]}. Expected 3 (RGB).")


def generate_random_map(size: int = 8, p: float = 0.8, seed: Optional[int] = None) -> List[str]:
    """
    Generate a random valid FrozenLake map with random start and goal.

    Args:
        size: Size of the map (size x size)
        p: Probability that a tile is frozen (not a hole)
        seed: Random seed for reproducibility

    Returns:
        List of strings representing the map
    """
    rng = np.random.default_rng(seed)

    valid = False
    while not valid:
        # Generate random map
        random_map = rng.choice(["F", "H"], size=(size, size), p=[p, 1 - p])
        # Randomly choose start and goal (must be different)
        start_r, start_c = rng.integers(size, size=2)
        goal_r, goal_c = rng.integers(size, size=2)
        if (start_r, start_c) == (goal_r, goal_c):
            continue
        random_map[start_r, start_c] = "S"
        random_map[goal_r, goal_c] = "G"
        # Check if map is valid (there is a path from start to goal)
        valid = is_valid(random_map)

    return ["".join(row) for row in random_map]


def is_valid(board: np.ndarray) -> bool:
    """
    Check if there is a valid path from start (S) to goal (G).
    Uses BFS to find a path.

    Args:
        board: 2D numpy array representing the map

    Returns:
        True if there is a valid path, False otherwise
    """
    from collections import deque

    nrow, ncol = board.shape
    start = None
    goal = None

    # Find start and goal positions
    for i in range(nrow):
        for j in range(ncol):
            if board[i, j] == "S":
                start = (i, j)
            elif board[i, j] == "G":
                goal = (i, j)

    if start is None or goal is None:
        return False

    # BFS to find path
    visited = set()
    queue = deque([start])
    visited.add(start)

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        row, col = queue.popleft()

        if (row, col) == goal:
            return True

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if (0 <= new_row < nrow and 0 <= new_col < ncol and
                (new_row, new_col) not in visited and
                board[new_row, new_col] != "H"):

                visited.add((new_row, new_col))
                queue.append((new_row, new_col))

    return False
