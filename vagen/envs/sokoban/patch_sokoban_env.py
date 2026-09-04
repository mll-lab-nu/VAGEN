import hashlib
from gym_sokoban.envs.sokoban_env import SokobanEnv
from gym.utils import seeding
from gym_sokoban.envs.room_utils import generate_room
from .utils.seeding import set_seed
import numpy as np
from collections import deque
import marshal
import copy


def _next_retry_seed(seed: int | None) -> int | None:
    """Advance a seed reproducibly when a room misses the difficulty band.

    Python's ``hash(str(seed))`` is salted independently in every interpreter. Ray
    workers therefore generated different fallback maps for the same dataset seed, and
    paired validation comparisons silently stopped being paired whenever reset retried.
    This full-period 32-bit LCG is only a deterministic walk through candidate seeds;
    ``set_seed`` still owns room-generation randomness.
    """
    if seed is None:
        return None
    return (1664525 * int(seed) + 1013904223) % (2**32)


def _room_partition_bucket(
    room_fixed: np.ndarray, room_state: np.ndarray, modulus: int
) -> int:
    if modulus < 2:
        raise ValueError("map_partition_modulus must be at least 2")
    fixed = np.asarray(room_fixed, dtype="<i8", order="C")
    state = np.asarray(room_state, dtype="<i8", order="C")
    digest = hashlib.sha256(fixed.tobytes() + state.tobytes()).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def _room_matches_partition(
    room_fixed: np.ndarray,
    room_state: np.ndarray,
    partition: str | None,
    modulus: int,
    eval_bucket: int,
) -> bool:
    if partition is None:
        return True
    if partition not in {"train", "eval"}:
        raise ValueError("map_partition must be 'train', 'eval', or null")
    if not 0 <= eval_bucket < modulus:
        raise ValueError("map_partition_eval_bucket must be within the modulus")
    held_out = _room_partition_bucket(room_fixed, room_state, modulus) == eval_bucket
    return held_out if partition == "eval" else not held_out


def get_shortest_action_path(room_fixed: np.ndarray, room_state: np.ndarray, MAX_DEPTH: int = 100) -> list[int]:
    """
    BFS shortest solution in action space (up/down/left/right).
    Returns [] if not found within MAX_DEPTH.
    """
    queue = deque([(copy.deepcopy(room_state), [])])
    explored_states = set()

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    actions = [1, 2, 3, 4]

    H, W = room_fixed.shape

    while queue:
        state, path = queue.popleft()
        if len(path) > MAX_DEPTH:
            return []

        key = marshal.dumps(state)
        if key in explored_states:
            continue
        explored_states.add(key)

        player_pos = tuple(np.argwhere(state == 5)[0])
        boxes_on_target = set(map(tuple, np.argwhere(state == 3)))
        boxes_not_on_target = set(map(tuple, np.argwhere(state == 4)))
        boxes = boxes_on_target | boxes_not_on_target

        if not boxes_not_on_target:
            return path

        for (dr, dc), act in zip(moves, actions):
            new_state = copy.deepcopy(state)
            nr, nc = player_pos[0] + dr, player_pos[1] + dc

            # bounds + wall
            if nr < 0 or nr >= H or nc < 0 or nc >= W or room_fixed[nr, nc] == 0:
                continue

            new_player_pos = (nr, nc)

            # push box?
            if new_player_pos in boxes:
                br, bc = nr, nc
                nbr, nbc = br + dr, bc + dc

                # bounds first (avoid index error), then wall/box
                if nbr < 0 or nbr >= H or nbc < 0 or nbc >= W:
                    continue
                if room_fixed[nbr, nbc] == 0 or (nbr, nbc) in boxes:
                    continue

                # move box away from (br, bc)
                new_state[br, bc] = room_fixed[br, bc]
                new_state[nbr, nbc] = 3 if room_fixed[nbr, nbc] == 2 else 4

            # move player
            pr, pc = player_pos
            new_state[pr, pc] = room_fixed[pr, pc]
            new_state[nr, nc] = 5

            queue.append((new_state, path + [act]))

    return []

class PatchedSokobanEnv(SokobanEnv):
    def reset(
        self,
        second_player=False,
        render_mode="rgb_array",
        seed=0,
        min_solution_steps=None,
        reset_seed_max_tries=10000,
        min_solution_bfs_max_depth=200,
        map_partition=None,
        map_partition_modulus=4,
        map_partition_eval_bucket=0,
    ):
        
        find_solution = False
        action_seq_len = 0
        for _try in range(reset_seed_max_tries):
            try:
                with set_seed(seed):
                    self.room_fixed, self.room_state, self.box_mapping = generate_room(
                        dim=self.dim_room,
                        num_steps=self.num_gen_steps,
                        num_boxes=self.num_boxes,
                        second_player=second_player
                    )
                    action_seq=get_shortest_action_path(self.room_fixed,self.room_state,MAX_DEPTH=min_solution_bfs_max_depth)
                    action_seq_len = len(action_seq)
                    difficulty_matches = (
                        min_solution_steps is None
                        or min_solution_steps[0] <= action_seq_len <= min_solution_steps[1]
                    )
                    partition_matches = _room_matches_partition(
                        self.room_fixed,
                        self.room_state,
                        map_partition,
                        int(map_partition_modulus),
                        int(map_partition_eval_bucket),
                    )
                    if difficulty_matches and partition_matches:
                        find_solution=True
                        break
            except (RuntimeError, RuntimeWarning) as e:
                print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
                print("[SOKOBAN] Retry . . .")
            seed = _next_retry_seed(seed)
        if not find_solution:
            if map_partition is not None:
                raise RuntimeError(
                    f"failed to generate a Sokoban room in map partition {map_partition!r} "
                    f"after {reset_seed_max_tries} attempts"
                )
            print(f"Max tries reached: {reset_seed_max_tries}, using map with action seq len {action_seq_len}")
                
        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = self.render(render_mode)
        return starting_observation
