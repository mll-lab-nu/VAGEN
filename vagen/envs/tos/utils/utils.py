import re
from collections import deque
from typing import Tuple
import hashlib
import numpy as np

from ..actions.base import BaseAction
from ..core.object import Agent, Object

# Reusable labels for formatting and parsing
THINK_LABEL = "THINK:"
ANSWER_LABEL = "FINAL ANSWER:"

_DEG_TO_VEC = {
    0: (0, 1),
    90: (1, 0),
    180: (0, -1),
    270: (-1, 0),
}
_VEC_TO_DEG = {v: k for k, v in _DEG_TO_VEC.items()}

def get_model_name(model_name: str) -> str:
    """Generate a unique directory name for the model configuration"""
    model_name = model_name.replace("\\", "/").rstrip("/").split("/")[-1]
    return model_name

def numpy_to_python(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def parse_llm_response(text: str, enable_think: bool = True) -> Tuple[str, str, bool]:
    """Parse LLM response for optional THINK and required FINAL ANSWER blocks.

    Expected format (labels on their own line). Parser tolerates content on the same line too:
    THINK:\n
    <think_content>    or    THINK: <think_content>
    FINAL ANSWER:\n
    <answer_content>  or    FINAL ANSWER: <answer_content>

    Returns: (think_content, answer_content, parsed_ok). parsed_ok is True if an answer was extracted.
    """
    if not isinstance(text, str):
        return "", "", False

    # Prefer new header-style format (labels can be mid-line or on their own line)
    think_re = re.compile(rf"(?is){re.escape(THINK_LABEL)}\s*(.*?)(?={re.escape(ANSWER_LABEL)}|\Z)")
    answer_re = re.compile(rf"(?is){re.escape(ANSWER_LABEL)}\s*(.*)\Z")

    think_match = think_re.search(text)
    answer_match = answer_re.search(text)
    think_content = (think_match.group(1).strip() if think_match else "")
    answer_content = (answer_match.group(1).strip() if answer_match else "")

    # Backward-compat: fall back to legacy <think>/<answer> tags if headers not found
    if not answer_content:
        legacy_ans = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if legacy_ans:
            answer_content = legacy_ans.group(1).strip()
    if not think_content:
        legacy_think = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        if legacy_think:
            think_content = legacy_think.group(1).strip()

    # If answer still missing, treat whole text as the answer (answer is critical)
    if not answer_content:
        answer_content = text.strip()

    # Clean GLM-4.5V box tokens if present
    if answer_content:
        answer_content = (
            answer_content.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "").strip()
        )

    if not enable_think:
        return "", answer_content, bool(answer_content)

    parsed_ok = bool(answer_content)
    return think_content, answer_content, parsed_ok

def hash(input_str: str) -> str:
    """Generate a stable hash for the given input string."""
    return hashlib.sha256(input_str.encode('utf-8')).hexdigest()[:16]

def format_llm_output(think_content: str, answer_content: str, enable_think: bool = True) -> str:
    """Format output with headers based on enable_think.

    THINK:\n<think_content>\nFINAL ANSWER:\n<answer_content>
    """
    if enable_think:
        return f"{THINK_LABEL}\n{think_content}\n{ANSWER_LABEL}\n{answer_content}"
    return f"{ANSWER_LABEL}\n{answer_content}"


def compute_shortest_path(
    room,
    start_pos: Tuple[int, int],
    start_ori: Tuple[int, int],
    target_pos: Tuple[int, int],
    target_ori: Tuple[int, int] = None,
) -> list:
    """Return shortest action list using rotate and jumpto actions.
    
    Returns a list of actions like [('rotate', 90), ('jumpto', 'lamp'), ...].
    If target_ori is provided, the path will include final rotation to match the orientation.
    """
    start_pos = tuple(map(int, start_pos))
    start_ori = tuple(map(int, start_ori))
    target_pos = tuple(map(int, target_pos))
    if target_ori is not None:
        target_ori = tuple(map(int, target_ori))

    # State: (pos, ori), value: (steps, parent_state, action_to_reach_here)
    queue = deque([(start_pos, start_ori, [])])  # (pos, ori, action_list)
    visited = {(start_pos, start_ori)}

    temp_agent = Agent(name="temp", pos=np.array(start_pos), ori=np.array(start_ori))
    tmp_room = room.copy()
    target_info = room.get_cell_info(int(target_pos[0]), int(target_pos[1]))
    initial_stub = Object(name="initial_pos", pos=np.array(target_pos))
    initial_stub.room_id = target_info.get("room_id")
    tmp_room.add_object(initial_stub)

    while queue:
        pos, ori, actions = queue.popleft()
        
        # Check if we've reached the target
        if np.allclose(pos, target_pos):
            if target_ori is None:
                # No target orientation specified, we're done
                return actions
            elif tuple(int(x) for x in ori) == target_ori:
                # Position and orientation both match
                return actions
            # Position matches but orientation doesn't - continue searching
            # (rotations will be explored from this state)

        temp_agent.pos = np.array(pos)
        temp_agent.ori = np.array(ori)
        cell = room.get_cell_info(int(pos[0]), int(pos[1]))
        temp_agent.room_id = cell.get("room_id")

        cur_deg = _VEC_TO_DEG.get(tuple(int(x) for x in ori), 0)
        for delta in (90, -90, 180):
            new_deg = (cur_deg + int(delta)) % 360
            new_ori = _DEG_TO_VEC.get(new_deg, ori)
            key = (pos, new_ori)
            if key not in visited:
                visited.add(key)
                new_actions = actions + [('rotate', int(delta))]
                queue.append((pos, new_ori, new_actions))

        for obj in tmp_room.all_objects:
            if np.allclose(obj.pos, pos):
                continue
            if BaseAction._is_visible(temp_agent, obj):
                new_pos = tuple(map(int, obj.pos))
                key = (new_pos, ori)
                if key not in visited:
                    visited.add(key)
                    new_actions = actions + [('jumpto', obj.name)]
                    queue.append((new_pos, ori, new_actions))

    raise ValueError("No path found")

if __name__ == "__main__":
    from .room_utils import RoomPlotter
    # Simple parser sanity checks
    llm_response = f"{THINK_LABEL}\nI will move then observe.{ANSWER_LABEL}\nActions: [JumpTo(table), Observe()]"
    t, a, ok = parse_llm_response(llm_response, enable_think=True)
    print('think:', t)
    print('answer:', a)
    print('ok:', ok)

    # Shortest-path sanity checks on generated three-room layout
    from .room_utils import RoomGenerator

    rng = np.random.default_rng(1)
    room, agent = RoomGenerator.generate_multi_room(room_size=[6, 6], room_num=4, n_objects=3, np_random=rng, topology=1)

    start_pos = tuple(map(int, agent.init_pos))
    start_ori = tuple(map(int, agent.init_ori))

    path = compute_shortest_path(room, start_pos, start_ori, start_pos)
    print(f'path (start->start): {path}, steps: {len(path)}')

    candidates = [
        obj for obj in room.all_objects
        if not np.allclose(obj.pos, agent.pos)
    ]
    RoomPlotter.plot(room, agent, mode='img', save_path='room.png')
    if candidates:
        target = candidates[-4]
        target_pos = tuple(map(int, target.pos))
        path = compute_shortest_path(room, start_pos, start_ori, target_pos)
        print(f'path (start->{target.name}): {path}, steps: {len(path)}')
        try:
            path = compute_shortest_path(room, target_pos, start_ori, start_pos)
            print(f'path ({target.name}->start): {path}, steps: {len(path)}')
        except ValueError as err:
            print('return path failed:', err)