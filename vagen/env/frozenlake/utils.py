
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