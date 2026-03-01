import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ..core.room import Room, BaseRoom
from ..core.constant import CANDIDATE_OBJECTS, ObjectInfo
from ..core.object import Object, Agent, Gate
from .generate_room_layout import generate_room_layout
from ..core.relationship import PairwiseRelationship
from ..actions.base import BaseAction

# Descriptive gate names to replace generic "door_X"
GATE_NAMES = [
    "red door", "blue door", "green door", "yellow door", "purple door",
    "orange door", "black door", "white door", "brown door", "gray door"
]


class RoomGenerator:
    """Room generator.
    1. Coordinates:
        - World coordinates are (x, y). Mask indexing is [x, y]. That is, rows map to x and columns map to y.
        - In mask indexing, row increasing means x increasing; column increasing means y increasing.
        - Neighbor convention on masks: "up" is y+1 and "down" is y-1; "right" is x+1 and "left" is x-1.
        - For visualization, we render with origin at bottom-left to keep y increasing upwards in world space.
    2. Room id:
        - Room id is an integer in [1, 99].
        - 0 for wall, 1 for main room, 100 for north-south door, 101 for east-west door.
    """
    
    @staticmethod
    def _validate_rotation_tasks(room: 'Room', agent: 'Agent', eval_tasks: list) -> bool:
        """Validate by attempting to create rotation tasks."""
        from ..evaluation.task_types import EvalTaskType
        try:
            for task in eval_tasks:
                if task.get('task_type') in ['rot', 'rot_dual']:
                    EvalTaskType.create_task(task['task_type'], np.random.default_rng(42), room, agent, task.get('task_kwargs', {})).generate_question()
            return True
        except:
            return False

    @staticmethod
    def _generate_objects_and_agent(mask, n_objects, fix_object_n, np_random, candidate_list=CANDIDATE_OBJECTS, gates= None, room_name=""):
        """Generate objects and agent for a room layout."""
        objects = RoomGenerator._gen_objects(
            n=n_objects,
            random_generator=np_random,
            room_size=[mask.shape[0], mask.shape[1]],
            perspective_taking=True,
            candidate_list=candidate_list,
            mask=mask,
            fix_object_n=fix_object_n,
        )
        
        # Find agent position
        valid_positions = RoomGenerator._get_valid_positions(mask, room_id=1)
        agent_pos = np_random.choice(valid_positions)
        while any(np.allclose(agent_pos, obj.pos) for obj in objects):
            agent_pos = np_random.choice(valid_positions)
            
        agent = Agent(name='agent', pos=agent_pos)
        agent.room_id = 1
        agent.init_room_id = 1
        
        room = Room(objects=objects, name=room_name, mask=mask.copy(), gates=gates)
        
        return room, agent
    @staticmethod
    def _default_mask(room_size: tuple[int, int]) -> np.ndarray:
        x_size, y_size = int(room_size[0]), int(room_size[1])
        # mask shape follows mask[x, y] convention: first axis = rows = x, second axis = cols = y
        # Create mask with walls (0) on the border instead of -1
        mask = np.ones((x_size + 2, y_size + 2), dtype=np.int8)
        mask[[0, -1], :] = 0  # Top and bottom walls
        mask[:, [0, -1]] = 0  # Left and right walls
        return mask

    @staticmethod
    def _get_valid_positions(mask: np.ndarray, room_id: int | None = None) -> List[Tuple[int, int]]:
        """Get valid positions in the mask."""
        if room_id is not None:
            valid = np.argwhere(mask == int(room_id))
        else:
            valid = np.argwhere((mask >= 1) & (mask < 100))
        return [(int(pos[0]), int(pos[1])) for pos in valid]
    
    @staticmethod
    def _gen_gates_from_mask(msk: np.ndarray) -> List[Gate]:
        gates: List[Gate] = []
        h, w = msk.shape
        cnt = 0
        # vertical doors (100, go through horizontally): look up and down for room ids
        xs, ys = np.where(msk == 100)
        for x, y in zip(xs.tolist(), ys.tolist()):
            up, down = int(msk[x - 1, y]), int(msk[x + 1, y]) # NOTE up and down are with respect mask indexing
            if 1 <= up < 100 and 1 <= down < 100 and up != down:
                gate_name = GATE_NAMES[cnt % len(GATE_NAMES)] if cnt < len(GATE_NAMES) else f"door_{cnt}"
                g = Gate(
                    name=gate_name,
                    pos=np.array([x, y], dtype=int),
                    ori=np.array([1, 0], dtype=int),
                    room_id=[int(up), int(down)],
                    ori_by_room={int(up): np.array([-1, 0], dtype=int), int(down): np.array([1, 0], dtype=int)},
                )
                gates.append(g); cnt += 1
        # horizontal doors (101, go through vertically): look left and right for room ids
        xs, ys = np.where(msk == 101)
        for x, y in zip(xs.tolist(), ys.tolist()):
            left, right = int(msk[x, y - 1]), int(msk[x, y + 1])
            if 1 <= left < 100 and 1 <= right < 100 and left != right:
                gate_name = GATE_NAMES[cnt % len(GATE_NAMES)] if cnt < len(GATE_NAMES) else f"door_{cnt}"
                g = Gate(
                    name=gate_name,
                    pos=np.array([x, y], dtype=int),
                    ori=np.array([0, 1], dtype=int),
                    room_id=[int(left), int(right)],
                    ori_by_room={int(left): np.array([0, -1], dtype=int), int(right): np.array([0, 1], dtype=int)},
                )
                gates.append(g); cnt += 1
        return gates

    @staticmethod
    @staticmethod
    def generate_room(
        room_size: Tuple[int, int],
        n_objects: int,
        np_random: np.random.Generator,
        room_name: str = 'room',
        candidate_objects: List[ObjectInfo] = CANDIDATE_OBJECTS,
        level: int = 0,
        main: Optional[int] = None,
        fixed_mask: bool = False,
        **kwargs
    ) -> Tuple[Room, Agent]:
        """Generate a multi-room layout, gates, objects, and agent.
        - Mask is generated via generate_room_layout; gates derived from mask.
        - Agent is sampled from main room (room id = 1).
        - Validates layout for rotation tasks and retries if needed.
        - If fixed_mask is True, uses seed=42 for mask generation while preserving original randomness for object/agent placement.
        """
        eval_tasks = kwargs.get('eval_tasks', [])
        max_retries = kwargs.get('max_retries', 30)
        fix_room_size = kwargs.get('fix_room_size', None)
        same_room_size = kwargs.get('same_room_size', False)
        
        # Store original random state for reproducibility
        original_state = np_random.bit_generator.state
        
        # Get the original seed to generate deterministic sub-seeds
        temp_random = np.random.default_rng()
        temp_random.bit_generator.state = original_state
        base_seed = temp_random.integers(0, 2**32 - 1)
        
        for attempt in range(max_retries + 1):
            try:
                # Use deterministic sub-seed based on base seed and attempt number
                sub_seed = (base_seed + attempt * 1000007) % (2**32)  # Use prime number to avoid patterns
                attempt_random = np.random.default_rng(sub_seed)
                n = int(max(room_size[0], room_size[1]))
                # Generate layout - use seed=42 for mask if fixed_mask is True, otherwise use attempt_random
                if fixed_mask:
                    # Use fixed seed=42 for mask generation to get consistent room layout
                    mask_random = np.random.default_rng(42)
                    mask = generate_room_layout(
                        n=n, level=int(level), main=main, 
                        np_random=mask_random, fix_room_size=fix_room_size,
                        same_room_size=same_room_size
                    )
                else:
                    # Use original random behavior
                    mask = generate_room_layout(
                        n=n, level=int(level), main=main, 
                        np_random=attempt_random, fix_room_size=fix_room_size,
                        same_room_size=same_room_size
                    )

                gates = RoomGenerator._gen_gates_from_mask(mask)
                
                # Object distribution strategies
                fix_object_n = kwargs.get('fix_object_n', None)
                proportional_to_area = kwargs.get('proportional_to_area', False)
                
                if proportional_to_area and not fix_object_n:
                    # Calculate proportional distribution
                    room_areas = []
                    num_rooms = level + 1
                    for room_id in range(1, num_rooms + 1):
                        room_area = np.sum(mask == room_id)
                        room_areas.append(room_area)
                    
                    # Distribute n_objects proportionally
                    if sum(room_areas) > 0:
                        proportions = [area / sum(room_areas) for area in room_areas]
                        fix_object_n = [max(1, round(n_objects * prop)) for prop in proportions]
                        # Adjust to ensure sum equals n_objects
                        diff = n_objects - sum(fix_object_n)
                        if diff != 0:
                            # Add/subtract from largest room
                            max_idx = room_areas.index(max(room_areas))
                            fix_object_n[max_idx] += diff
                
                room, agent = RoomGenerator._generate_objects_and_agent(
                    mask, n_objects, fix_object_n, attempt_random, candidate_objects, gates, room_name
                )
                
                # Validate layout for rotation tasks
                if RoomGenerator._validate_rotation_tasks(room, agent, eval_tasks):
                    return room, agent
                else:
                    if attempt == max_retries:
                        print(f"Warning: Failed to generate valid layout after {max_retries + 1} attempts. "
                              f"Using layout that may have insufficient angular separation for rotation tasks.")
                        return room, agent
                    # Continue to next attempt
                    
            except Exception as e:
                if attempt == max_retries:
                    raise e
                # Continue to next attempt
                
        # This should not be reached, but just in case
        raise RuntimeError(f"Failed to generate room after {max_retries + 1} attempts")

    @staticmethod
    def generate_multi_room(
        room_size: Tuple[int, int],
        n_objects: int,
        np_random: np.random.Generator,
        room_name: str = 'room',
        candidate_objects: List[ObjectInfo] = CANDIDATE_OBJECTS,
        room_num: int = 1,
        topology: int = 0,
        **kwargs
    ) -> Tuple[Room, Agent]:
        """Generate multiple rooms with specified topology.

        Args:
            room_size: Size of each individual room (width, height)
            n_objects: Number of objects per room
            np_random: Random number generator
            room_name: Base name for the room
            candidate_objects: List of candidate objects to place
            room_num: Total number of rooms to generate (MUST be 4)
            topology: Connection topology for main room (room_id=1)
                - 0: Main room at bottom edge, connects to only 1 room
                - 1: Main room connects to 2 rooms (only if room_num > 2)
                - 2: Main room connects to 3 rooms (only if room_num > 3)
                - 3: Ring/circular structure (e.g., 4 rooms: 1-2, 2-3, 3-4, 4-1)
            **kwargs: Additional arguments (max_retries, eval_tasks, etc.)

        Returns:
            Tuple of (Room, Agent) where Room contains all rooms and Agent is in main room
        """

        if room_num == 1:
            # Single room case - use default mask
            mask = RoomGenerator._default_mask(room_size)
            gates = []
            room, agent = RoomGenerator._generate_objects_and_agent(
                mask, n_objects, None, np_random, candidate_objects, gates, room_name
            )
            return room, agent

        # Multi-room case
        eval_tasks = kwargs.get('eval_tasks', [])
        # Use more retries for larger room counts
        default_retries = 30 if room_num > 3 else 10
        max_retries = kwargs.get('max_retries', default_retries)

        # Store original random state
        original_state = np_random.bit_generator.state
        temp_random = np.random.default_rng()
        temp_random.bit_generator.state = original_state
        base_seed = temp_random.integers(0, 2**32 - 1)

        for attempt in range(max_retries + 1):
            try:
                sub_seed = (base_seed + attempt * 1000007) % (2**32)
                attempt_random = np.random.default_rng(sub_seed)

                # Generate custom multi-room layout with topology constraint
                mask = RoomGenerator._generate_multi_room_layout(
                    room_size=room_size,
                    room_num=room_num,
                    topology=topology,
                    np_random=attempt_random
                )
                
                if mask is None:
                    continue
                # print(mask)

                corner_gap_strategy = kwargs.get('corner_gap_strategy', 'retry')  # 'retry' | 'off'
                if corner_gap_strategy == 'retry':
                    if RoomGenerator._has_corner_gap(mask):
                        continue  # switch to next attempt with a new sub-seed

                # Generate gates from mask
                gates = RoomGenerator._gen_gates_from_mask(mask)

                # Distribute objects: n_objects per room
                fix_object_n = [n_objects] * room_num
                total_objects = n_objects * room_num

                # Generate objects and agent
                room, agent = RoomGenerator._generate_objects_and_agent(
                    mask, total_objects, fix_object_n, attempt_random,
                    candidate_objects, gates, room_name
                )

                # Validate layout for rotation tasks
                if RoomGenerator._validate_rotation_tasks(room, agent, eval_tasks):
                    return room, agent
                else:
                    if attempt == max_retries:
                        print(f"Warning: Failed to generate valid layout after {max_retries + 1} attempts. "
                              f"Using layout that may have insufficient angular separation for rotation tasks.")
                        return room, agent

            except Exception as e:
                if attempt == max_retries:
                    raise e

        raise RuntimeError(f"Failed to generate multi-room after {max_retries + 1} attempts")

    @staticmethod
    def _generate_multi_room_layout(
        room_size: Tuple[int, int],
        room_num: int,
        topology: int,
        np_random: np.random.Generator
    ) -> Optional[np.ndarray]:
        """Generate a multi-room layout with topology constraints.

        Args:
            room_size: Size of each room (width, height)
            room_num: Number of rooms
            topology: 0 = main room connects to 1 room, 1 = main room connects to 2 rooms,
                     2 = main room connects to 3 rooms, 3 = ring/circular structure
            np_random: Random generator

        Returns:
            Mask array or None if generation failed
        """
        width, height = room_size

        # Calculate grid size needed to fit all rooms
        # Each room needs width+2 (room + 2 walls) in each dimension
        grid_size = max(30, (width + 2) * int(np.ceil(np.sqrt(room_num))) * 2 + 10)

        # Initialize grid
        grid = np.full((grid_size, grid_size), -1, dtype=np.int8)

        # Generate room positions (coordinates are for room interior, walls will be added around)
        rooms = []

        # Track custom connections for topology 2 (random connections)
        topology_2_connections = None

        if topology == 0:
            # Main room connects to only 1 room
            # Randomly place main room (room 1) with some offset from center
            center_x = (grid_size - width) // 2
            center_y = (grid_size - height) // 2

            # Add random offset to main room position
            main_offset_x = int(np_random.integers(-grid_size // 8, grid_size // 8 + 1))
            main_offset_y = int(np_random.integers(-grid_size // 8, grid_size // 8 + 1))
            main_x = max(2, min(grid_size - width - 2, center_x + main_offset_x))
            main_y = max(2, min(grid_size - height - 2, center_y + main_offset_y))
            rooms.append((main_x, main_y, main_x + width - 1, main_y + height - 1))

            # Place second room in a random direction from main room
            if room_num >= 2:
                # Randomly choose a direction for the second room
                # Directions: 0=right, 1=left, 2=up, 3=down
                direction = int(np_random.integers(0, 4))

                max_offset = min(1, min(width, height) // 4) if room_num > 3 else min(2, min(width, height) // 3)

                room2_x, room2_y = RoomGenerator._calculate_adjacent_room_position(
                    main_x, main_y, main_x + width - 1, main_y + height - 1,
                    direction, width, height, max_offset, np_random, grid_size
                )
                rooms.append((room2_x, room2_y, room2_x + width - 1, room2_y + height - 1))

            # Place remaining rooms in a chain
            # Each new room branches from the previous room in any valid direction
            for i in range(2, room_num):
                # Get the previous room in the chain
                prev_room = rooms[i - 1]  # Chain: room i connects to room i-1
                x1, y1, x2, y2 = prev_room

                # Try different directions and offsets to find a valid placement
                # Directions: 0=right, 1=left, 2=up, 3=down
                directions = list(range(4))
                np_random.shuffle(directions)

                placed = False
                for direction in directions:
                    # Calculate offset range based on room size
                    max_offset = min(2, min(width, height) // 3)
                    offset_range = list(range(-max_offset, max_offset + 1))
                    np_random.shuffle(offset_range)

                    for offset in offset_range:
                        if direction == 0:  # Right
                            new_x = x2 + 2
                            new_y = max(2, min(grid_size - height - 2, y1 + offset))
                        elif direction == 1:  # Left
                            new_x = x1 - width - 1
                            new_y = max(2, min(grid_size - height - 2, y1 + offset))
                        elif direction == 2:  # Up
                            new_x = max(2, min(grid_size - width - 2, x1 + offset))
                            new_y = y1 - height - 1
                        else:  # Down
                            new_x = max(2, min(grid_size - width - 2, x1 + offset))
                            new_y = y2 + 2

                        # Check if new room is within bounds
                        if new_x < 2 or new_y < 2 or new_x + width > grid_size - 2 or new_y + height > grid_size - 2:
                            continue

                        # Check if new room overlaps with any existing room
                        new_room = (new_x, new_y, new_x + width - 1, new_y + height - 1)
                        if not RoomGenerator._rooms_overlap_with_margin(new_room, rooms):
                            rooms.append(new_room)
                            placed = True
                            break

                    if placed:
                        break

                # If we couldn't place the room, return None to retry
                if not placed:
                    return None

        elif topology == 1 and room_num > 2:
            # Main room connects to 2 rooms
            # Randomly choose main room position (center with random offset)
            center_x = (grid_size - width) // 2
            center_y = (grid_size - height) // 2

            # Add random offset to main room position
            main_offset_x = int(np_random.integers(-grid_size // 8, grid_size // 8 + 1))
            main_offset_y = int(np_random.integers(-grid_size // 8, grid_size // 8 + 1))
            main_x = max(2, min(grid_size - width - 2, center_x + main_offset_x))
            main_y = max(2, min(grid_size - height - 2, center_y + main_offset_y))
            rooms.append((main_x, main_y, main_x + width - 1, main_y + height - 1))

            # Randomly choose two different directions for the two connected rooms
            # Directions: 0=right, 1=left, 2=up, 3=down
            available_directions = list(range(4))
            np_random.shuffle(available_directions)
            dir1, dir2 = available_directions[:2]

            max_offset = min(1, min(width, height) // 4) if room_num > 3 else min(2, min(width, height) // 3)

            # Place room 2 in direction dir1
            room2_x, room2_y = RoomGenerator._calculate_adjacent_room_position(
                main_x, main_y, main_x + width - 1, main_y + height - 1,
                dir1, width, height, max_offset, np_random, grid_size
            )
            rooms.append((room2_x, room2_y, room2_x + width - 1, room2_y + height - 1))

            # Place room 3 in direction dir2
            room3_x, room3_y = RoomGenerator._calculate_adjacent_room_position(
                main_x, main_y, main_x + width - 1, main_y + height - 1,
                dir2, width, height, max_offset, np_random, grid_size
            )
            rooms.append((room3_x, room3_y, room3_x + width - 1, room3_y + height - 1))

            # Place remaining rooms branching from rooms 2 and 3 with random directions
            for i in range(3, room_num):
                # Alternate between branching from room 2 and room 3
                if i % 2 == 1:
                    base_room = rooms[1]  # Room 2
                else:
                    base_room = rooms[2]  # Room 3

                x1, y1, x2, y2 = base_room

                # Try different random directions to find a valid placement
                directions = list(range(4))
                np_random.shuffle(directions)

                placed = False
                for direction in directions:
                    max_offset_val = min(1, min(width, height) // 4) if room_num > 3 else min(2, min(width, height) // 3)

                    new_x, new_y = RoomGenerator._calculate_adjacent_room_position(
                        x1, y1, x2, y2, direction, width, height, max_offset_val, np_random, grid_size
                    )

                    # Check if new room is within bounds
                    if new_x < 2 or new_y < 2 or new_x + width > grid_size - 2 or new_y + height > grid_size - 2:
                        continue

                    # Check if new room overlaps with any existing room
                    new_room = (new_x, new_y, new_x + width - 1, new_y + height - 1)
                    if not RoomGenerator._rooms_overlap_with_margin(new_room, rooms):
                        rooms.append(new_room)
                        placed = True
                        break

                # If we couldn't place the room, return None to retry
                if not placed:
                    return None

        elif topology == 2 and room_num >= 4:
            # Main room connects to 3 rooms
            # Track connections for topology 2: list of (parent_room_idx, child_room_idx)
            topology_2_connections = []

            # Randomly choose main room position (center with random offset)
            center_x = (grid_size - width) // 2
            center_y = (grid_size - height) // 2

            # Add random offset to main room position
            main_offset_x = int(np_random.integers(-grid_size // 8, grid_size // 8 + 1))
            main_offset_y = int(np_random.integers(-grid_size // 8, grid_size // 8 + 1))
            main_x = max(2, min(grid_size - width - 2, center_x + main_offset_x))
            main_y = max(2, min(grid_size - height - 2, center_y + main_offset_y))
            rooms.append((main_x, main_y, main_x + width - 1, main_y + height - 1))

            # Randomly choose three different directions for the three connected rooms
            # Directions: 0=right, 1=left, 2=up, 3=down
            available_directions = list(range(4))
            np_random.shuffle(available_directions)
            dir1, dir2, dir3 = available_directions[:3]

            max_offset = min(1, min(width, height) // 4) if room_num > 4 else min(2, min(width, height) // 3)

            # Place room 2 in direction dir1
            room2_x, room2_y = RoomGenerator._calculate_adjacent_room_position(
                main_x, main_y, main_x + width - 1, main_y + height - 1,
                dir1, width, height, max_offset, np_random, grid_size
            )
            rooms.append((room2_x, room2_y, room2_x + width - 1, room2_y + height - 1))
            topology_2_connections.append((0, 1))  # Main room (0) connects to room 2 (index 1)

            # Place room 3 in direction dir2
            room3_x, room3_y = RoomGenerator._calculate_adjacent_room_position(
                main_x, main_y, main_x + width - 1, main_y + height - 1,
                dir2, width, height, max_offset, np_random, grid_size
            )
            rooms.append((room3_x, room3_y, room3_x + width - 1, room3_y + height - 1))
            topology_2_connections.append((0, 2))  # Main room (0) connects to room 3 (index 2)

            # Place room 4 in direction dir3
            room4_x, room4_y = RoomGenerator._calculate_adjacent_room_position(
                main_x, main_y, main_x + width - 1, main_y + height - 1,
                dir3, width, height, max_offset, np_random, grid_size
            )
            rooms.append((room4_x, room4_y, room4_x + width - 1, room4_y + height - 1))
            topology_2_connections.append((0, 3))  # Main room (0) connects to room 4 (index 3)

            # Place remaining rooms randomly connecting to any existing room
            for i in range(4, room_num):
                # Randomly choose an existing room to connect to
                # Create a shuffled list of existing room indices to try
                existing_room_indices = list(range(len(rooms)))
                np_random.shuffle(existing_room_indices)

                placed = False
                connected_to_idx = -1
                # Try to place the new room adjacent to each existing room
                for base_idx in existing_room_indices:
                    base_room = rooms[base_idx]
                    x1, y1, x2, y2 = base_room

                    # Try different random directions to find a valid placement
                    directions = list(range(4))
                    np_random.shuffle(directions)

                    for direction in directions:
                        max_offset_val = min(1, min(width, height) // 4) if room_num > 4 else min(2, min(width, height) // 3)

                        new_x, new_y = RoomGenerator._calculate_adjacent_room_position(
                            x1, y1, x2, y2, direction, width, height, max_offset_val, np_random, grid_size
                        )

                        # Check if new room is within bounds
                        if new_x < 2 or new_y < 2 or new_x + width > grid_size - 2 or new_y + height > grid_size - 2:
                            continue

                        # Check if new room overlaps with any existing room
                        new_room = (new_x, new_y, new_x + width - 1, new_y + height - 1)
                        if not RoomGenerator._rooms_overlap_with_margin(new_room, rooms):
                            rooms.append(new_room)
                            connected_to_idx = base_idx
                            placed = True
                            break

                    if placed:
                        break

                # If we couldn't place the room, return None to retry
                if not placed:
                    return None

                # Record the connection
                topology_2_connections.append((connected_to_idx, i))

        elif topology == 3:
            # Ring/circular structure: rooms arranged in a ring
            # For 4 rooms: 1-2, 2-3, 3-4, 4-1
            # Strategy: Use a 2x2 grid layout to ensure all connections are possible
            # Layout:
            #   [Room 4] [Room 1]
            #   [Room 3] [Room 2]
            # Connections: 1-2 (vertical), 2-3 (horizontal), 3-4 (vertical), 4-1 (horizontal)

            if room_num != 4:
                raise ValueError(f"Ring topology (topology=3) only supports room_num=4, got {room_num}")

            # Calculate center position for the 2x2 grid
            center_x = (grid_size - 2 * width - 1) // 2  # -1 for wall between rooms
            center_y = (grid_size - 2 * height - 1) // 2

            # Add small random offset to avoid always centering
            offset_x = int(np_random.integers(-2, 3))
            offset_y = int(np_random.integers(-2, 3))
            center_x = max(2, min(grid_size - 2 * width - 3, center_x + offset_x))
            center_y = max(2, min(grid_size - 2 * height - 3, center_y + offset_y))

            # Place rooms in 2x2 grid pattern
            # Room 1 (main room): top-right
            room1_x = center_x + width + 1
            room1_y = center_y + height + 1
            rooms.append((room1_x, room1_y, room1_x + width - 1, room1_y + height - 1))

            # Room 2: bottom-right
            room2_x = center_x + width + 1
            room2_y = center_y
            rooms.append((room2_x, room2_y, room2_x + width - 1, room2_y + height - 1))

            # Room 3: bottom-left
            room3_x = center_x
            room3_y = center_y
            rooms.append((room3_x, room3_y, room3_x + width - 1, room3_y + height - 1))

            # Room 4: top-left
            room4_x = center_x
            room4_y = center_y + height + 1
            rooms.append((room4_x, room4_y, room4_x + width - 1, room4_y + height - 1))

            # Validate all rooms are within bounds
            for room in rooms:
                x1, y1, x2, y2 = room
                if x1 < 2 or y1 < 2 or x2 >= grid_size - 2 or y2 >= grid_size - 2:
                    return None

        else:
            # Fallback: topology 2 with room_num < 4, treat as topology 1
            # Fallback: topology 1 with room_num <= 2, treat as topology 0
            if topology == 2 and room_num < 4:
                return RoomGenerator._generate_multi_room_layout(room_size, room_num, 1, np_random)
            elif topology == 3 and room_num < 3:
                # Ring topology needs at least 3 rooms to make sense
                return RoomGenerator._generate_multi_room_layout(room_size, room_num, 0, np_random)
            else:
                return RoomGenerator._generate_multi_room_layout(room_size, room_num, 0, np_random)

        # Validate all rooms are within bounds
        for room in rooms:
            x1, y1, x2, y2 = room
            if x1 < 1 or y1 < 1 or x2 >= grid_size - 1 or y2 >= grid_size - 1:
                return None

        # Place rooms in grid
        for i, room in enumerate(rooms):
            room_id = i + 1
            x1, y1, x2, y2 = room
            grid[y1:y2+1, x1:x2+1] = room_id

        # Add walls around rooms
        RoomGenerator._add_walls_to_multi_room(grid, rooms)

        # check if there is a corner gap in the mask
        if RoomGenerator._has_corner_gap(grid):
            return None

        # Validate that no two rooms are directly adjacent (without a wall between them)
        if not RoomGenerator._validate_room_separation(grid, rooms):
            return None

        # Generate connections based on topology
        connections = RoomGenerator._generate_topology_connections(rooms, topology, topology_2_connections)

        # Add doors
        doors_added = RoomGenerator._add_doors_to_multi_room(grid, rooms, connections, np_random)

        if len(doors_added) != len(connections):
            return None

        # Trim the grid to minimum bounding box (remove excess -1 borders)
        grid = RoomGenerator._trim_mask_to_bounding_box(grid)

        return grid

    @staticmethod
    def _trim_mask_to_bounding_box(mask: np.ndarray) -> np.ndarray:
        """Trim mask to minimum bounding box containing all non--1 values.

        This ensures that the four edges of the mask contain at least one non--1 value.
        """
        # Find all positions that are not -1
        non_empty = np.argwhere(mask != -1)

        if len(non_empty) == 0:
            # If everything is -1, return a minimal mask
            return mask

        # Get bounding box
        y_coords = non_empty[:, 0]
        x_coords = non_empty[:, 1]

        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()

        # Extract the bounding box
        trimmed = mask[y_min:y_max+1, x_min:x_max+1]

        return trimmed

    @staticmethod
    def _find_corner_gaps(msk: np.ndarray) -> list:
        """
        Detect 2x2 diagonal-room corner gaps.
        Pattern A:
            a (room)   c (wall=0)
            b (wall=0) d (room), a!=d

        Pattern B (the other diagonal):
            a (wall=0) c (room)
            b (room)   d (wall=0), b!=c

        Doors (100/101) are NOT walls (also invalid separators).
        Returns a list of gap descriptors (tuples of coordinates).
        """
        H, W = msk.shape
        gaps = []
        is_room = lambda v: 1 <= int(v) < 100
        is_wall = lambda v: int(v) == 0

        for x in range(H - 1):
            for y in range(W - 1):
                a = int(msk[x, y])
                b = int(msk[x + 1, y])
                c = int(msk[x, y + 1])
                d = int(msk[x + 1, y + 1])

                # Pattern A: diagonal a (room) - d (room), with b,c as walls
                if is_room(a) and is_room(d) and a != d and is_wall(b) and is_wall(c):
                    gaps.append(((x, y), (x + 1, y + 1)))

                # Pattern B: diagonal b (room) - c (room), with a,d as walls
                if is_room(b) and is_room(c) and b != c and is_wall(a) and is_wall(d):
                    gaps.append(((x + 1, y), (x, y + 1)))
        return gaps

    @staticmethod
    def _has_corner_gap(msk: np.ndarray) -> bool:
        """Boolean shortcut."""
        return len(RoomGenerator._find_corner_gaps(msk)) > 0

    @staticmethod
    def _calculate_adjacent_room_position(
        x1: int, y1: int, x2: int, y2: int,
        direction: int, width: int, height: int,
        max_offset: int, np_random: np.random.Generator,
        grid_size: int
    ) -> Tuple[int, int]:
        """Calculate position for a room adjacent to another room in a given direction.

        Args:
            x1, y1, x2, y2: Coordinates of the base room
            direction: 0=right, 1=left, 2=up, 3=down
            width, height: Size of the new room
            max_offset: Maximum offset perpendicular to the direction
            np_random: Random generator
            grid_size: Size of the grid

        Returns:
            (new_x, new_y) coordinates for the new room
        """
        if max_offset > 0:
            offset = int(np_random.integers(-max_offset, max_offset + 1))
        else:
            offset = 0

        if direction == 0:  # Right
            new_x = x2 + 2
            new_y = max(2, min(grid_size - height - 2, y1 + offset))
        elif direction == 1:  # Left
            new_x = x1 - width - 1
            new_y = max(2, min(grid_size - height - 2, y1 + offset))
        elif direction == 2:  # Up
            new_x = max(2, min(grid_size - width - 2, x1 + offset))
            new_y = y1 - height - 1
        else:  # Down (direction == 3)
            new_x = max(2, min(grid_size - width - 2, x1 + offset))
            new_y = y2 + 2

        return new_x, new_y

    @staticmethod
    def _generate_topology_connections(
        rooms: List[Tuple[int, int, int, int]],
        topology: int,
        custom_connections: Optional[List[Tuple[int, int]]] = None
    ) -> List[Tuple[int, int]]:
        """Generate room connections based on topology.

        Args:
            rooms: List of room coordinates
            topology: Connection pattern
            custom_connections: For topology 2, pre-computed random connections

        Returns:
            List of (room_idx1, room_idx2) connections
        """
        connections = []
        num_rooms = len(rooms)

        if num_rooms <= 1:
            return connections

        if topology == 0:
            # Main room (0) connects to room 1 only
            connections.append((0, 1))
            # Chain remaining rooms
            for i in range(1, num_rooms - 1):
                connections.append((i, i + 1))

        elif topology == 1 and num_rooms > 2:
            # Main room (0) connects to rooms 1 and 2
            connections.append((0, 1))
            connections.append((0, 2))
            # Chain remaining rooms alternating between branches
            for i in range(3, num_rooms):
                if i % 2 == 1:
                    connections.append((1, i))  # Connect to room 2's branch
                else:
                    connections.append((2, i))  # Connect to room 3's branch

        elif topology == 2 and num_rooms >= 4:
            # Use custom connections if provided (random connections)
            if custom_connections is not None:
                connections = custom_connections
            else:
                # Fallback: Main room (0) connects to rooms 1, 2, and 3
                connections.append((0, 1))
                connections.append((0, 2))
                connections.append((0, 3))
                # Chain remaining rooms cycling between branches
                for i in range(4, num_rooms):
                    branch_idx = ((i - 4) % 3) + 1  # Cycles through 1, 2, 3 (rooms 2, 3, 4)
                    connections.append((branch_idx, i))

        elif topology == 3:
            # Ring/circular structure: 1-2, 2-3, 3-4, ..., n-1
            # For example, with 4 rooms: 0-1, 1-2, 2-3, 3-0
            for i in range(num_rooms - 1):
                connections.append((i, i + 1))
            # Close the ring by connecting the last room back to the first
            if num_rooms > 2:
                connections.append((num_rooms - 1, 0))

        return connections

    @staticmethod
    def _rooms_overlap_with_margin(new_room: Tuple[int, int, int, int], existing_rooms: List[Tuple[int, int, int, int]]) -> bool:
        """Check if a new room overlaps with any existing room (including wall margins).

        Two rooms overlap if they are too close (less than 2 cells apart in any direction).
        We need at least 1 cell for the wall between rooms.

        Args:
            new_room: (x1, y1, x2, y2) coordinates of the new room
            existing_rooms: List of existing room coordinates

        Returns:
            True if there is overlap, False otherwise
        """
        nx1, ny1, nx2, ny2 = new_room

        for room in existing_rooms:
            x1, y1, x2, y2 = room

            # Check if rooms are too close in x direction
            # Rooms need at least 1 cell gap for wall
            x_overlap = not (nx2 + 1 < x1 or nx1 > x2 + 1)

            # Check if rooms are too close in y direction
            y_overlap = not (ny2 + 1 < y1 or ny1 > y2 + 1)

            # If both x and y overlap, rooms are too close
            if x_overlap and y_overlap:
                return True

        return False

    @staticmethod
    def _validate_room_separation(grid: np.ndarray, rooms: List[Tuple[int, int, int, int]]) -> bool:
        """Validate that no two rooms are directly adjacent without a wall between them.

        This checks that every room cell is only adjacent to cells of the same room,
        walls (0), or empty space (-1), but never to cells of a different room.

        Args:
            grid: The grid with rooms placed
            rooms: List of room coordinates

        Returns:
            True if all rooms are properly separated, False otherwise
        """
        h, w = grid.shape

        for y in range(h):
            for x in range(w):
                cell_value = grid[y, x]

                # Only check room cells (1-99)
                if not (1 <= cell_value < 100):
                    continue

                # Check all 4 neighbors
                for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ny, nx = y + dy, x + dx

                    # Skip out of bounds
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue

                    neighbor_value = grid[ny, nx]

                    # Check if neighbor is a different room (not same room, not wall, not empty)
                    if 1 <= neighbor_value < 100 and neighbor_value != cell_value:
                        return False

        return True

    @staticmethod
    def _add_walls_to_multi_room(grid: np.ndarray, rooms: List[Tuple[int, int, int, int]]):
        """Add walls around rooms in multi-room layout.

        Walls are only added where there is -1 (empty space), so adjacent rooms
        will share walls instead of having double walls.
        """
        n = grid.shape[0]

        for room in rooms:
            x1, y1, x2, y2 = room

            # Add walls around room perimeter, but only where there's empty space (-1)
            # This ensures adjacent rooms share walls
            for x in range(max(0, x1-1), min(n, x2+2)):
                # Top wall
                if y1 - 1 >= 0 and grid[y1-1, x] == -1:
                    grid[y1-1, x] = 0
                # Bottom wall
                if y2 + 1 < n and grid[y2+1, x] == -1:
                    grid[y2+1, x] = 0

            for y in range(max(0, y1-1), min(n, y2+2)):
                # Left wall
                if x1 - 1 >= 0 and grid[y, x1-1] == -1:
                    grid[y, x1-1] = 0
                # Right wall
                if x2 + 1 < n and grid[y, x2+1] == -1:
                    grid[y, x2+1] = 0

    @staticmethod
    def _add_doors_to_multi_room(
        grid: np.ndarray,
        rooms: List[Tuple[int, int, int, int]],
        connections: List[Tuple[int, int]],
        np_random: np.random.Generator
    ) -> List[Tuple[int, int, int]]:
        """Add doors between connected rooms on walls.

        Rooms are separated by walls, and doors are placed on these walls.
        The gap between rooms is 2 cells: one for wall, one for door.
        """
        doors_added = []

        for room1_idx, room2_idx in connections:
            room1 = rooms[room1_idx]
            room2 = rooms[room2_idx]

            x1_1, y1_1, x2_1, y2_1 = room1
            x1_2, y1_2, x2_2, y2_2 = room2

            door_placed = False

            # Check if rooms are horizontally separated (room1 left of room2)
            # Gap is 2: x2_1 + 1 is wall, x2_1 + 2 is start of room2
            if x2_1 + 2 == x1_2:
                # Find overlapping y range
                y_start = max(y1_1, y1_2)
                y_end = min(y2_1, y2_2)

                if y_end >= y_start:
                    # Place door connecting the two rooms
                    # Door should be between the two rooms, replacing the wall
                    door_x = x2_1 + 1  # Wall position between rooms
                    door_y = int(np_random.integers(y_start, y_end + 1))
                    if 0 <= door_x < grid.shape[1] and 0 <= door_y < grid.shape[0]:
                        # Verify both sides have rooms
                        left_room = grid[door_y, door_x - 1] if door_x > 0 else -1
                        right_room = grid[door_y, door_x + 1] if door_x < grid.shape[1] - 1 else -1
                        if 1 <= left_room < 100 and 1 <= right_room < 100:
                            grid[door_y, door_x] = 101  # East-west door
                            doors_added.append((door_x, door_y, 101))
                            door_placed = True

            # Check if rooms are horizontally separated (room2 left of room1)
            elif x2_2 + 2 == x1_1:
                y_start = max(y1_1, y1_2)
                y_end = min(y2_1, y2_2)

                if y_end >= y_start:
                    door_x = x2_2 + 1  # Wall position between rooms
                    door_y = int(np_random.integers(y_start, y_end + 1))
                    if 0 <= door_x < grid.shape[1] and 0 <= door_y < grid.shape[0]:
                        # Verify both sides have rooms
                        left_room = grid[door_y, door_x - 1] if door_x > 0 else -1
                        right_room = grid[door_y, door_x + 1] if door_x < grid.shape[1] - 1 else -1
                        if 1 <= left_room < 100 and 1 <= right_room < 100:
                            grid[door_y, door_x] = 101  # East-west door
                            doors_added.append((door_x, door_y, 101))
                            door_placed = True

            # Check if rooms are vertically separated (room1 above room2)
            # Gap is 2: y2_1 + 1 is wall, y2_1 + 2 is start of room2
            elif y2_1 + 2 == y1_2:
                x_start = max(x1_1, x1_2)
                x_end = min(x2_1, x2_2)

                if x_end >= x_start:
                    door_y = y2_1 + 1  # Wall position between rooms
                    door_x = int(np_random.integers(x_start, x_end + 1))
                    if 0 <= door_x < grid.shape[1] and 0 <= door_y < grid.shape[0]:
                        # Verify both sides have rooms
                        up_room = grid[door_y - 1, door_x] if door_y > 0 else -1
                        down_room = grid[door_y + 1, door_x] if door_y < grid.shape[0] - 1 else -1
                        if 1 <= up_room < 100 and 1 <= down_room < 100:
                            grid[door_y, door_x] = 100  # North-south door
                            doors_added.append((door_x, door_y, 100))
                            door_placed = True

            # Check if rooms are vertically separated (room2 above room1)
            elif y2_2 + 2 == y1_1:
                x_start = max(x1_1, x1_2)
                x_end = min(x2_1, x2_2)

                if x_end >= x_start:
                    door_y = y2_2 + 1  # Wall position between rooms
                    door_x = int(np_random.integers(x_start, x_end + 1))
                    if 0 <= door_x < grid.shape[1] and 0 <= door_y < grid.shape[0]:
                        # Verify both sides have rooms
                        up_room = grid[door_y - 1, door_x] if door_y > 0 else -1
                        down_room = grid[door_y + 1, door_x] if door_y < grid.shape[0] - 1 else -1
                        if 1 <= up_room < 100 and 1 <= down_room < 100:
                            grid[door_y, door_x] = 100  # North-south door
                            doors_added.append((door_x, door_y, 100))
                            door_placed = True

            # If no door was placed with exact gap of 2, try to find nearby walls
            if not door_placed:
                # Try to find a wall between the two rooms
                door_placed = RoomGenerator._try_place_door_on_nearby_wall(
                    grid, room1, room2, np_random, doors_added
                )

        return doors_added

    @staticmethod
    def _try_place_door_on_nearby_wall(
        grid: np.ndarray,
        room1: Tuple[int, int, int, int],
        room2: Tuple[int, int, int, int],
        np_random: np.random.Generator,
        doors_added: List[Tuple[int, int, int]]
    ) -> bool:
        """Try to place a door on a wall between two rooms that may not be perfectly aligned."""
        x1_1, y1_1, x2_1, y2_1 = room1
        x1_2, y1_2, x2_2, y2_2 = room2

        # Check if rooms are roughly horizontally separated
        if abs(x2_1 - x1_2) <= 5 or abs(x2_2 - x1_1) <= 5:
            # Find overlapping y range
            y_start = max(y1_1, y1_2)
            y_end = min(y2_1, y2_2)

            if y_end >= y_start:
                # Determine which room is on the left
                if x2_1 < x1_2:
                    # Room1 is on the left, find wall between them
                    for door_x in range(x2_1 + 1, x1_2):
                        if grid[y_start, door_x] == 0:  # Found a wall
                            door_y = int(np_random.integers(y_start, y_end + 1))
                            grid[door_y, door_x] = 101
                            doors_added.append((door_x, door_y, 101))
                            return True
                else:
                    # Room2 is on the left
                    for door_x in range(x2_2 + 1, x1_1):
                        if grid[y_start, door_x] == 0:  # Found a wall
                            door_y = int(np_random.integers(y_start, y_end + 1))
                            grid[door_y, door_x] = 101
                            doors_added.append((door_x, door_y, 101))
                            return True

        # Check if rooms are roughly vertically separated
        if abs(y2_1 - y1_2) <= 5 or abs(y2_2 - y1_1) <= 5:
            # Find overlapping x range
            x_start = max(x1_1, x1_2)
            x_end = min(x2_1, x2_2)

            if x_end >= x_start:
                # Determine which room is on top
                if y2_1 < y1_2:
                    # Room1 is on top, find wall between them
                    for door_y in range(y2_1 + 1, y1_2):
                        if grid[door_y, x_start] == 0:  # Found a wall
                            door_x = int(np_random.integers(x_start, x_end + 1))
                            grid[door_y, door_x] = 100
                            doors_added.append((door_x, door_y, 100))
                            return True
                else:
                    # Room2 is on top
                    for door_y in range(y2_2 + 1, y1_1):
                        if grid[door_y, x_start] == 0:  # Found a wall
                            door_x = int(np_random.integers(x_start, x_end + 1))
                            grid[door_y, door_x] = 100
                            doors_added.append((door_x, door_y, 100))
                            return True

        return False

    @staticmethod
    def _gen_objects(
        n: int,
        random_generator: np.random.Generator,
        room_size: list[int] = [5, 5],
        perspective_taking: bool = False,
        candidate_list: list[ObjectInfo] = CANDIDATE_OBJECTS,
        mask: Optional[np.ndarray] = None,
        fix_object_n: Optional[List[int]] = None,
    ) -> List[Object]:
        """Sample objects (names, orientations) and positions from mask."""
        if mask is None:
            mask = RoomGenerator._default_mask((room_size[0], room_size[1]))
        
        objects = []
        ori_vectors = {0: [0, 1], 1: [1, 0], 2: [0, -1], 3: [-1, 0]}
        
        # Generate positions based on distribution strategy
        if fix_object_n is not None:
            positions = []
            for room_id, num_objects in enumerate(fix_object_n, start=1):
                if num_objects > 0:
                    room_positions = RoomGenerator._get_valid_positions(mask, room_id=room_id)
                    if len(room_positions) < num_objects:
                        raise ValueError(f"Room {room_id} needs {num_objects} objects but only has {len(room_positions)} positions")
                    random_generator.shuffle(room_positions)
                    positions.extend(room_positions[:num_objects])
        else:
            all_positions = RoomGenerator._get_valid_positions(mask)
            if len(all_positions) < n:
                raise ValueError(f"Need {n} objects but only {len(all_positions)} positions available")
            random_generator.shuffle(all_positions)
            positions = all_positions[:n]

        # Generate objects with selected positions
        indices = random_generator.choice(len(candidate_list), len(positions), replace=False)
        orientations = random_generator.integers(0, 4, len(positions))

        for label, (idx, pos, ori_idx) in enumerate(zip(indices, positions, orientations)):
            obj_info = candidate_list[idx]
            ori = np.array(ori_vectors[int(ori_idx)]) if obj_info.has_orientation and perspective_taking else np.array([0, 1])
            objects.append(Object(name=obj_info.name.replace('_', ' '), pos=np.array(pos, dtype=int), ori=ori, has_orientation=obj_info.has_orientation, label=label+1))
        
        return objects


class RoomPlotter:
    @staticmethod
    def get_symbol_definition() -> str:
        return (
            "Symbolic Map:\n"
            "- **North is Up.**\n"
            "- **Coordinates**: (0, 0) is at the bottom-left (last row, first column). X points Right (East), Y points Up (North).\n"
            "- Legend:\n"
            "  - '.': Floor (integer coordinates)\n"
            "  - '#': Wall\n"
            "  - '+': Door\n"
            "  - '*': Agent's current Position\n"
            "  - 'A-Z': Candidates"
        )

    @staticmethod
    def get_symbolic_map(room: Room, agent: Agent | None = None, include_object: bool = False, 
                          candidate_points: List[Tuple[int, int]] | None = None) -> str:
        """Convert room to symbolic map (N*M)."""
        if getattr(room, 'mask', None) is None: return ""
        x_size, y_size = room.mask.shape
        grid = [['#' for _ in range(x_size)] for _ in range(y_size)]

        for x in range(x_size):
            for y in range(y_size):
                v = int(room.mask[x, y])
                if 1 <= v < 100: grid[y_size - 1 - y][x] = '.'
                elif v in (100, 101): grid[y_size - 1 - y][x] = '+'

        if include_object:
            chars = "abcdefghijklmnopqrstuvwxyz"
            for i, o in enumerate(room.objects):
                x, y = int(o.pos[0]), int(o.pos[1])
                if 0 <= x < x_size and 0 <= y < y_size: grid[y_size - 1 - y][x] = chars[i % len(chars)]

        if candidate_points:
            if len(candidate_points) > 26: raise ValueError("Too many candidate points (max 26)")
            for i, (x, y) in enumerate(candidate_points):
                x, y = int(x), int(y)
                if 0 <= x < x_size and 0 <= y < y_size: grid[y_size - 1 - y][x] = chr(ord('A') + i)

        if agent:
            pos = agent.pos
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < x_size and 0 <= y < y_size: grid[y_size - 1 - y][x] = '*'

        return "\n".join(["".join(row) for row in grid])

    @staticmethod
    def plot(room: Room, agent: Agent | None, mode: str = 'text', save_path: str | None = None):
        """Render room.
        - mode='text': print symbolic map.
        - mode='img': save image to save_path if provided.
        """
        if mode == 'text':
            print(RoomPlotter.get_symbolic_map(room, agent, include_object=True))
        elif mode == 'img':

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_facecolor('white')
            RoomPlotter._draw_img(ax, room, agent)
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close(); return
        else:
            raise ValueError('mode must be text or img')

    @staticmethod
    def _draw_img(ax, room: Room, agent: Agent | None):
        has_mask = getattr(room, 'mask', None) is not None and isinstance(room, Room)
        gate_labels = {}
        if room.gates:
            gate_labels = {g.name: g.name for i, g in enumerate(room.gates)}
        if has_mask:
            h, w = room.mask.shape  # h: rows=x, w: cols=y
            min_x, max_x, min_y, max_y = 0, h - 1, 0, w - 1
            # build label map for background coloring
            label = np.zeros_like(room.mask, dtype=int)
            label[room.mask == 0] = 1
            label[(room.mask == 100) | (room.mask == 101)] = 2
            rids = sorted(int(r) for r in np.unique(room.mask) if 1 <= int(r) < 100)
            for i, rid in enumerate(rids, start=3):
                label[room.mask == rid] = i
            colors = ['#111111', '#888888', '#ffcc33', '#4e79a7', '#59a14f', '#f28e2b', '#e15759', '#76b7b2', '#edc949', '#af7aa1']
            # repeat colors if many rooms
            while len(colors) <= label.max():
                colors += colors[3:]
            
            # transpose so that imshow's x-axis corresponds to world x (rows)
            ax.imshow(label.T, origin='lower', cmap=ListedColormap(colors[:label.max()+1]),
                      extent=(min_x-0.5, max_x+0.5, min_y-0.5, max_y+0.5), interpolation='nearest', zorder=0, alpha=0.18)
            # room id labels
            for rid in rids:
                xs, ys = np.where(room.mask == rid)
                if len(xs) == 0:
                    continue
                cx, cy = float(np.mean(xs)), float(np.mean(ys)) # x, y
                ax.text(cx, cy, str(rid), color='white', ha='center', va='center', fontsize=8, zorder=1, alpha=0.85)
        else:
            min_x, max_x, min_y, max_y = room.get_boundary()
        # grid
        ax.set_xlim(min_x-0.5, max_x+0.5); ax.set_ylim(min_y-0.5, max_y+0.5)
        ax.set_xticks(np.arange(int(min_x), int(max_x)+1)); ax.set_yticks(np.arange(int(min_y), int(max_y)+1))
        ax.grid(True, color='#bdbdbd', linewidth=0.2)
        for s in ax.spines.values(): s.set_visible(False)
        # draw objects/gates + orientation (distinct markers/colors; legend, no text labels)
        palette = plt.get_cmap('tab10').colors
        markers = ['o','s','D','P','X','h','H','*','p','d']
        name_to_idx, seen_labels = {}, set()
        for obj in room.all_objects:
            x, y = float(obj.pos[0]), float(obj.pos[1])
            if isinstance(obj, Gate) and hasattr(room, 'gates'):
                color, marker, label = 'crimson', 'D', ('gate' if 'gate' not in seen_labels else None)
                seen_labels.add('gate')
            else:
                i = name_to_idx.setdefault(obj.name, len(name_to_idx))
                color, marker = palette[i % len(palette)], markers[i % len(markers)]
                label = obj.name if obj.name not in seen_labels else None
                seen_labels.add(obj.name)
            ax.scatter(x, y, c=[color], marker=marker, s=64, edgecolors='white', linewidths=0.7, zorder=3, label=label)
            if getattr(obj, 'has_orientation', True):
                dx, dy = float(obj.ori[0])*0.4, float(obj.ori[1])*0.4
                ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=0.5, color='grey', width=0.005)
            if isinstance(obj, Gate) and gate_labels:
                ax.text(x+0.10, y+0.12, gate_labels.get(obj.name, ''), color='crimson', fontsize=8, zorder=4)
        if agent is not None:
            ax.scatter(agent.pos[0], agent.pos[1], c='green', marker='s', s=70, edgecolors='white', linewidths=0.7, label='agent', zorder=5)
            ax.scatter(agent.init_pos[0], agent.init_pos[1], c='green', marker='x', s=60, label='agent_init', zorder=4)
            dx, dy = float(agent.ori[0])*0.4, float(agent.ori[1])*0.4
            idx, idy = float(agent.init_ori[0])*0.4, float(agent.init_ori[1])*0.4
            ax.quiver(agent.pos[0], agent.pos[1], dx, dy, angles='xy', scale_units='xy', scale=0.5, color='grey', width=0.005)
            ax.quiver(agent.init_pos[0], agent.init_pos[1], idx, idy, angles='xy', scale_units='xy', scale=0.5, color='grey', width=0.005)
        h,l = ax.get_legend_handles_labels();
        if l:
            d = dict(zip(l,h)); ax.legend(d.values(), d.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
        ax.set_aspect('equal'); ax.set_title(room.name)

    @staticmethod
    def plot_to_image(room: Room, agent: Agent | None, observe: bool = False, dpi: int = 120):
        import io, imageio
        from matplotlib import patches
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_facecolor('white')
        RoomPlotter._draw_img(ax, room, agent)
        if observe and agent is not None:
            # Draw FOV boundary lines centered on agent orientation
            fov = float(BaseAction.get_field_of_view())
            theta = float(np.arctan2(float(agent.ori[1]), float(agent.ori[0])))
            half = np.deg2rad(fov / 2.0)
            lengths = 0.9
            for ang in (theta - half, theta + half):
                dx, dy = lengths * np.cos(ang), lengths * np.sin(ang)
                ax.plot([float(agent.pos[0]), float(agent.pos[0]) + dx], [float(agent.pos[1]), float(agent.pos[1]) + dy], c='tab:green', lw=2.0, alpha=0.9)
        buf = io.BytesIO()
        plt.tight_layout(); plt.savefig(buf, format='png', dpi=int(dpi))
        plt.close(fig)
        buf.seek(0)
        return imageio.v2.imread(buf)

def get_room_description(room: Room, agent: Agent) -> str:
    # Get room information
    if hasattr(room, 'mask') and room.mask is not None:
        room_ids = [int(rid) for rid in np.unique(room.mask) if 1 <= int(rid) < 100]
        num_rooms = len(room_ids)
        room_names = [f"room {rid}" for rid in sorted(room_ids)]
    else:
        num_rooms = 1
        room_names = ["room 1"]

    room_type = f"{num_rooms} rooms connected by doors" if room.gates else "a single room"
    assert isinstance(agent.room_id, int), f"Agent room id must be an integer, got {agent.room_id}"

    # Separate objects and gates for clearer description
    # objects = [o.label + ": " + o.name + "(has facing direction)" if o.has_orientation else "" for o in room.all_objects if not isinstance(o, Gate)]
    # TODO, currently only for cognitive map
    objects = [str(o.label) + ": " + o.name  for o in room.all_objects if not isinstance(o, Gate)]
    # objects = [o.name for o in sorted([o for o in room.all_objects if not isinstance(o, Gate)], key=lambda x: x.label) ]
    gates = [o.name for o in room.all_objects if isinstance(o, Gate)]

    desc = f"Imagine {room_type}: {', '.join(room_names)}. You current (initial) position is in room {agent.room_id} and you face north. Unless otherwise specified, treat your initial position as the origin (0, 0), and north is +y."
    desc += f"\nObjects: {', '.join(objects)}" if objects else ""

    if gates:
        desc += f"\nDoors: {', '.join(gates)}"


    return desc

def initialize_room_from_json(json_data: Dict[str, Any]) ->  Tuple[Room, Agent]:
    """
    Initialize a Room from your metadata JSON, which now has:
      - objects: list of {oid, model, pos:{x,y,z}, rot:{x,y,z}, size:[w,h]}
      - cameras: list of {id, label, position:{x,y,z}, rotation:{y}}
      - room_size, screen_size, etc.
    """
    # Rotation to orientation vector mapping
    rotation_map = {0: np.array([0, 1]), 90: np.array([1, 0]), 180: np.array([0, -1]), 270: np.array([-1, 0])}
    offset = np.array(json_data['offset'])
    # 1) Parse all objects
    objects = []
    for obj in json_data['objects']:
        if 'door' not in obj['name']:
            objects.append(Object(
                name=obj['name'].replace('_', ' '),
                pos=np.array([obj["pos"]["x"], obj["pos"]["z"]]),
                ori=rotation_map.get(obj["rot"]["y"]) if obj["attributes"]["has_orientation"] else np.array([1, 0]),
                has_orientation=obj["attributes"]["has_orientation"],
                label=obj['label'],
            ))
        
    # 2) Room size metadata
    room_name = json_data.get("name", "room_from_json")

    # 3) Build and return
    agent_pos = [camera['position'] for camera in json_data['cameras'] if camera['id'] == 'agent'][0]
    agent_pos = np.array([agent_pos["x"], agent_pos["z"]])
    for obj in objects:
        obj.pos +=offset
    agent_pos +=offset
    mask = np.array(json_data['mask'])
    gates = RoomGenerator._gen_gates_from_mask(mask)

    # Update gate room_ids to match the connected rooms from JSON data
    door_objects = [obj for obj in json_data['objects'] if 'door' in obj['name']]
    for gate in gates:
        for door in door_objects:
            if set(gate.room_id) == set(door["attributes"]['connected_rooms']):
                gate.name = door['name'].replace('_', ' ')

    return Room(objects=objects, mask=mask, name=room_name, gates=gates), Agent(pos=agent_pos,room_id=1,init_room_id=1)


if __name__ == '__main__':
    import json
    import os
    
    # Simple debug for symbolic map
    json_path = "vagen/env/spatial/room_data_3_room/run00/meta_data.json"
    with open(json_path, 'r') as f:
        data = json.load(f)
    room, agent = initialize_room_from_json(data)

    RoomPlotter.plot(room, agent, mode='img', save_path='room.png')
    
    print("Symbolic Map (Init Pos):")
    print(RoomPlotter.get_symbolic_map(room, agent, include_object=False, use_agent_pos=False))
    
    print("\nSymbolic Map (Candidate Points):")
    candidates = [(1, 7), (3, 2), (6, 14)]
    print(RoomPlotter.get_symbolic_map(room, agent, include_object=False, use_agent_pos=False, candidate_points=candidates))
