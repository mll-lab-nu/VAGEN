from gym_sokoban.envs.sokoban_env import SokobanEnv
from gym.utils import seeding
from gym_sokoban.envs.room_utils import generate_room
from .utils.seeding import set_seed
import numpy as np
from collections import deque
import marshal
import copy
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
    def reset(self, second_player=False, render_mode='rgb_array',seed=0,min_solution_steps=None,reset_seed_max_tries=10000,min_solution_bfs_max_depth=200):
        
        find_solution = False if min_solution_steps is not None else True
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
                    if find_solution or len(action_seq)>=min_solution_steps[0] and len(action_seq)<=min_solution_steps[1]:
                        find_solution=True
                        break
            except (RuntimeError, RuntimeWarning) as e:
                print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
                print("[SOKOBAN] Retry . . .")
            seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
        if not find_solution:
            print(f"Max tries reached: {reset_seed_max_tries}, using map with action seq len {action_seq_len}")
                
        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = self.render(render_mode)
        return starting_observation