
from typing import Dict, List, Optional, Tuple, Any
from gymnasium.utils import seeding
import numpy as np
def generate_random_map(size: int = 8, p: float = 0.8, seed: Optional[int] = None) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
        seed: optional seed to ensure the generation of reproducible maps

    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    np_random, _ = seeding.np_random(seed)

    # generate random start and end points
    while not valid:
        p = min(1, p)
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])

        while True:
            start_r = np_random.integers(0, size)
            start_c = np_random.integers(0, size)
            goal_r = np_random.integers(0, size)
            goal_c = np_random.integers(0, size)
            
            # Ensure start and goal are different positions
            if (start_r, start_c) != (goal_r, goal_c):
                break
            
        board[start_r][start_c] = "S"
        board[goal_r][goal_c] = "G"
        
        valid = is_valid(board, size)
    return ["".join(x) for x in board]


def is_valid(board: List[List[str]], max_size: int) -> bool:
    """Check if the board is valid (has a path from start to goal)"""
    frontier, discovered = [], set()
    # find the start point
    start_r, start_c = np.where(np.array(board) == "S")
    frontier.append((start_r[0], start_c[0]))
    # dfs to check if there is a path from start to goal
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


def state_to_sentences(state_dict):
    """
    Convert game state dictionary to descriptive sentences about spatial relationships.
    
    Args:
        state_dict (dict): Dictionary containing:
            - player_position: tuple (row, col)
            - target_position: tuple (row, col) 
            - hole_positions: list of tuples [(row, col), ...]
            - grid_size: tuple (rows, cols)
    
    Returns:
        list: List of descriptive sentences
    """
    sentences = []
    
    player_pos = state_dict['player_position']
    target_pos = state_dict['target_position']
    hole_positions = state_dict['hole_positions']
    
    def get_relative_position(pos1, pos2):
        """
        Get relative position description between two positions.
        pos1 is the reference point, pos2 is described relative to pos1.
        """
        row1, col1 = pos1
        row2, col2 = pos2
        
        if pos1 == pos2:
            return "at the same place as"
        
        # Determine row relationship
        if row1 == row2:
            if col1 > col2:
                return "at the same row and to the left of"
            else:  # col1 < col2
                return "at the same row and to the right of"
        elif col1 == col2:
            if row1 > row2:
                return "above and at the same column as"
            else:  # row1 < row2
                return "below and at the same column as"
        else:
            # Different row and column
            row_desc = "above" if row1 > row2 else "below"
            col_desc = "on the left side" if col1 > col2 else "on the right side"
            return f"{row_desc} and {col_desc} of"
    
    # Describe target relative to player
    target_relation = get_relative_position(player_pos, target_pos)
    sentences.append(f"target is {target_relation} the player")
    
    # Describe each hole relative to player
    for i, hole_pos in enumerate(hole_positions):
        hole_relation = get_relative_position(player_pos, hole_pos)
        sentences.append(f"hole{i} is {hole_relation} the player")
    
    return sentences

import numpy as np
import json # Included for potential future use or just to show output format

# Assuming this helper function is available or defined here
def get_relative_relation(player_pos: tuple[int, int], object_pos: tuple[int, int]) -> tuple[str, str]:
    """
    Calculates the vertical and horizontal relation of an object relative to the player.

    Args:
        player_pos: A tuple (row, col) for the player's position (y, x).
        object_pos: A tuple (row, col) for the object's position (y, x).
        Rows increase downwards, columns increase to the right.

    Returns:
        A tuple (vertical_relation, horizontal_relation) using "above", "below",
        "same" for vertical and "left", "right", "same" for horizontal.
    """
    p_row, p_col = player_pos
    o_row, o_col = object_pos

    # Vertical relation (row comparison)
    # Smaller row index means "above" (closer to the top of the grid)
    if o_row < p_row:
        vertical_relation = "above"
    elif o_row > p_row:
        vertical_relation = "below"
    else:
        vertical_relation = "same"

    # Horizontal relation (column comparison)
    # Smaller column index means "left" (closer to the left of the grid)
    if o_col < p_col:
        horizontal_relation = "left"
    elif o_col > p_col:
        horizontal_relation = "right"
    else:
        horizontal_relation = "same"

    return (vertical_relation, horizontal_relation)


def convert_frozenlake_state_to_relative_list(state_dict: dict) -> list[dict]:
    """
    Converts a FrozenLake state dictionary into a list of dictionaries
    representing relative object positions (Target, Holes) to the player,
    matching the LLM parser's expected JSON output format for Groundtruth.

    Uses generic "hole" ID for all holes. Ignores grid_size.

    Args:
        state_dict: The dictionary containing player_position, target_position,
                    hole_positions, and grid_size (ignored).

    Returns:
        A list of dictionaries, each describing a target or hole's relative
        position to the player. Returns empty list if player_position is missing.
    """
    player_pos = state_dict.get("player_position")
    if player_pos is None:
        print("Warning: 'player_position' not found in state_dict. Cannot calculate relative positions.")
        return [] # Cannot calculate relative positions without player

    relative_positions_list = []

    # Process Target position (FrozenLake usually has only one target)
    target_pos = state_dict.get("target_position")
    if target_pos: # Check if target_position exists and is not empty
        v_rel, h_rel = get_relative_relation(player_pos, target_pos)
        relative_positions_list.append({
            "object_id": "target",
            "vertical_relation": v_rel,
            "horizontal_relation": h_rel
        })
    # Note: FrozenLake has only one target, so no loop needed for targets

    # Process Hole positions
    hole_positions = state_dict.get("hole_positions", [])
    for hole_pos in hole_positions:
        # Use generic 'hole' ID for all holes
        v_rel, h_rel = get_relative_relation(player_pos, hole_pos)
        relative_positions_list.append({
            "object_id": "hole",
            "vertical_relation": v_rel,
            "horizontal_relation": h_rel
        })

    return relative_positions_list