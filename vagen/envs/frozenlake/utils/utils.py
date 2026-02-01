import re
import numpy as np
from typing import Dict, List, Optional
from PIL import Image


def parse_free_think(response: str, action_sep: str = ",", max_actions: int = 3) -> Dict:
    """
    Parse free_think format response: <think>...</think><answer>...</answer>

    Args:
        response: Raw LLM response string
        action_sep: Separator between actions
        max_actions: Maximum number of actions to extract

    Returns:
        Dict containing parsed components and validation info
    """
    # Pattern to match <think>...</think><answer>...</answer>
    pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
    match = re.search(pattern, response, re.DOTALL)

    format_correct = match is not None

    if not match:
        think_content = ""
        action_content = ""
        actions = []
    else:
        think_content = match.group(1).strip()
        action_content = match.group(2).strip()

        # Split actions by separator and clean them
        actions = [action.strip().lower() for action in action_content.split(action_sep) if action.strip()]

        # Limit to max_actions
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = action_sep.join(actions)

    # Reconstruct formatted response
    llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>"

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }


def parse_wm(response: str, action_sep: str = ",", max_actions: int = 3) -> Dict:
    """
    Parse wm format response:
    <observation>...</observation>
    <think>...</think>
    <answer>...</answer>
    <prediction>...</prediction>
    """
    pattern = (
        r'<observation>(.*?)</observation>\s*'
        r'<think>(.*?)</think>\s*'
        r'<answer>(.*?)</answer>\s*'
        r'<prediction>(.*?)</prediction>'
    )

    match = re.search(pattern, response, re.DOTALL)
    format_correct = match is not None

    if not match:
        observation_content = ""
        think_content = ""
        prediction_content = ""
        action_content = ""
        actions: List[str] = []
    else:
        observation_content = match.group(1).strip()
        think_content = match.group(2).strip()
        action_content = match.group(3).strip()
        prediction_content = match.group(4).strip()

        # Parse actions
        actions = [
            action.strip().lower()
            for action in action_content.split(action_sep)
            if action.strip()
        ]

        # Limit number of actions
        if len(actions) > max_actions:
            actions = actions[:max_actions]
            action_content = action_sep.join(actions)

    # Reconstruct formatted response (canonical)
    llm_response = (
        f"<observation>{observation_content}</observation>"
        f"<think>{think_content}</think>"
        f"<answer>{action_content}</answer>"
        f"<prediction>{prediction_content}</prediction>"
    )

    # For backward-compat with old keys: treat think as reasoning
    reasoning_content = think_content

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "observation_content": observation_content,
        "think_content": think_content,
        "reasoning_content": reasoning_content,
        "prediction_content": prediction_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
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
