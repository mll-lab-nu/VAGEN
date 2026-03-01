"""Localization task: infer your 2D coordinate from a new view."""

from typing import List, Tuple, Dict
import numpy as np
import json

from .tasks import BaseEvaluationTask, retry_generate_question
from ..core.object import Object, Gate
from ..core.relationship import PairwiseRelationshipDiscrete
from ..actions import BaseAction
from ..actions import ObserveAction
from ..utils.utils import hash

"""
Task Overview:
1. View2LocationEvaluationTask (Backward Loc): Infer (x,y) and orientation from observations.
   - Evaluated by: coordinate similarity (position + orientation) vs ground truth. (TODO: debug for metrics)
2. Location2ViewEvaluationTask (Forward Loc): Infer ego relation after movement to (x,y).
   - Evaluated by: direction and distance match.
"""

LOC_2_ACTION_TEMPLATE = (
    "You move to a new location and face {orientation}.\n"
    "{observations}\n"
    "{origin_instruction}"
    "What is your new 2D coordinate (x, y)?\n\n"
    "Answer format: (x, y)\n"
    "Example: (2, -1)\n"
)

ACTION_2_LOC_TEMPLATE = (
    "{origin_instruction}"
    "You move to {loc} and face {direction}.\n"
    "What is the egocentric relation of {target}?\n\n"
    "Answer format: <direction>, <distance>\n"
    "Example: front, near\n"
)


def _ori_to_name(ori: Tuple[int, int]) -> str:
    mapping = {(0, 1): "north", (1, 0): "east", (0, -1): "south", (-1, 0): "west"}
    return mapping.get(tuple(int(x) for x in ori), "north")

class BaseLocEvaluationTask(BaseEvaluationTask):
    """Base class for localization tasks."""
    def _pick_room(self) -> int:
        rids = [r for r in self.room.objects_by_room if isinstance(r, int) and r > 0]
        self.np_random.shuffle(rids)
        for rid in rids:
            objs = [self.room.get_object_by_name(n) for n in self.room.objects_by_room[rid]]
            if len(objs) < 2: continue
            if any(np.linalg.norm(objs[i].pos - objs[j].pos) > 1.0 + 1e-6 
                   for i in range(len(objs)) for j in range(i + 1, len(objs))):
                return rid
        return 1

    def _pick_best_pose_and_obs(self) -> Tuple[np.ndarray, np.ndarray, int, List[Dict], Dict]:
        """Sample poses and pick best based on visibility."""
        def pose_generator():
            for _ in range(20):
                rid = self._pick_room()
                xmin, xmax, ymin, ymax = self.room.get_boundary(room_id=rid)
                for _ in range(10):
                    x, y = self.np_random.integers(xmin, xmax + 1), self.np_random.integers(ymin, ymax + 1)
                    if not self.room.get_cell_info(x, y)['object_name']:
                        ori = [(0,1), (1,0), (0,-1), (-1,0)][self.np_random.integers(0, 4)]
                        yield (np.array([x, y]), np.array(ori), int(rid))
                        break

        def make_agent(cand):
             tmp = self.agent.copy()
             tmp.pos, tmp.ori, tmp.room_id = cand
             return tmp

        best_cand, obs, oris = self._select_best_candidate(
            candidates=pose_generator(),
            get_agent_func=make_agent,
            min_visible=2, max_obs=3
        )
        return best_cand[0], best_cand[1], best_cand[2], obs, oris

    def _get_origin(self) -> Tuple[Tuple[int, int], str]:
        """Determine origin position and name."""
        init_pos = self.agent.init_pos
        init_room_info = self.room.get_cell_info(int(init_pos[0]), int(init_pos[1]))
        init_room_id = init_room_info.get('room_id')
        
        current_room_id = self.agent.room_id
        
        if current_room_id == init_room_id:
            return tuple(map(int, init_pos)), "your starting position"
        
        # Find a door in the current room
        names = self.room.objects_by_room.get(int(current_room_id), [])
        if hasattr(self.room, 'gates_by_room'):
            names.extend(self.room.gates_by_room.get(int(current_room_id), []))
            
        gates = []
        for name in names:
            obj = self.room.get_object_by_name(name)
            if isinstance(obj, Gate):
                gates.append(obj)
        
        if gates:
            gate = self.np_random.choice(gates)
            return tuple(map(int, gate.pos)), f"the {gate.name}"
            
        return tuple(map(int, init_pos)), "your starting position"


class BaseView2LocationEvaluationTask(BaseLocEvaluationTask):
    """Base class for Location2Action (Backward Localization) tasks."""
    QUESTION_TEMPLATE = LOC_2_ACTION_TEMPLATE

    @retry_generate_question
    def generate_question(self) -> dict:
        pos, ori, rid, observations, _ = self._pick_best_pose_and_obs()
        self.agent.pos, self.agent.ori, self.agent.room_id = pos, ori, rid
        
        origin_pos, origin_name = self._get_origin()
        origin_instruction = (
            "Still treat your initial position as origin (0, 0)\n"
            if origin_name == "your starting position"
            else f"Treat {origin_name} as the new 'origin' (0, 0).\n"
        )
        obs_text = self._format_observations_custom(observations)

        correct_coord = (
            int(self.agent.pos[0]) - int(origin_pos[0]),
            int(self.agent.pos[1]) - int(origin_pos[1]),
        )
        correct_orientation = _ori_to_name(tuple(self.agent.ori))

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            orientation=correct_orientation,
            observations=obs_text,
            origin_instruction=origin_instruction,
        )
        
        # Collect object info
        object_positions = {}
        all_orientations = {}
        gate_info = {}
        object_rooms = {}
        
        for obj in self.room.all_objects:
            name = obj.name.lower()
            object_positions[name] = tuple(map(int, obj.pos))
            
            if isinstance(obj.room_id, (list, tuple, np.ndarray)):
                object_rooms[name] = [int(x) for x in obj.room_id]
            else:
                object_rooms[name] = int(obj.room_id)
                
            if obj.has_orientation:
                all_orientations[name] = tuple(map(int, obj.ori))
            if isinstance(obj, Gate):
                 gate_info[name] = {
                     'room_ids': [int(x) for x in obj.room_id] if isinstance(obj.room_id, (list, tuple, np.ndarray)) else [int(obj.room_id)],
                     'ori_by_room': {int(k): tuple(map(int, v)) for k, v in obj.ori_by_room.items()}
                 }

        self.eval_data.answer = {
            'coord': correct_coord,
            'final_pos': tuple(map(int, self.agent.pos)),
            'final_ori': tuple(map(int, self.agent.ori)),
            'room_id': int(rid),
            'object_positions': object_positions,
            'object_orientations': all_orientations,
            'final_observation': observations,
            'gate_info': gate_info,
            'object_rooms': object_rooms
        }
        
        self.eval_data.choices = []
        self.eval_data.id = hash(json.dumps(self.eval_data.answer, sort_keys=True, default=str) + self.eval_data.question)
        return self.eval_data.question

    def _format_observations_custom(self, observations: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class View2LocationTextEvaluationTask(BaseView2LocationEvaluationTask):
    """Localize your own coordinate (x, y) and orientation using text observations."""
    def _format_observations_custom(self, observations: List[Dict[str, str]]) -> str:
        obs_parts = []
        for v in observations:
            txt = f"{v['name']} is at {v['direction']}, {v['distance']}"
            if v.get('orientation'):
                txt += f", {v['orientation']}"
            obs_parts.append(txt)
        return "You observe: " + "; ".join(obs_parts)


class View2LocationVisionEvaluationTask(BaseView2LocationEvaluationTask):
    """Localize your own coordinate (x, y) and orientation using vision."""
    def _format_observations_custom(self, observations: List[Dict[str, str]]) -> str:
         return "You observe: <image>"

class Location2ViewEvaluationTask(BaseLocEvaluationTask):
    
    QUESTION_TEMPLATE = ACTION_2_LOC_TEMPLATE

    @retry_generate_question
    def generate_question(self) -> dict:
        pos, ori, rid, rels, _ = self._pick_best_pose_and_obs()
        self.agent.pos, self.agent.ori, self.agent.room_id = pos, ori, rid
        
        origin_pos, origin_name = self._get_origin()
        origin_instruction = (
            "Still treat your initial position as origin (0, 0)\n"
            if origin_name == "your starting position"
            else f"Treat {origin_name} as the new 'origin' (0, 0).\n"
        )

        # question fields
        loc_rel = (int(self.agent.pos[0]) - origin_pos[0], int(self.agent.pos[1]) - origin_pos[1])
        dir_name = _ori_to_name(tuple(self.agent.ori))

        # compute correct observation text (pairwise-only, compact)
        # compute correct observation text (pairwise-only, compact)
        if not rels:
            raise ValueError("No visible relations found")
        first = rels[0]
        target_name = first['name']
        direction = first['direction']
        distance = first['distance']

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            origin_instruction=origin_instruction,
            loc=f"({int(loc_rel[0])}, {int(loc_rel[1])})",
            direction=dir_name,
            target=target_name,
        )
        self.eval_data.answer = f"{direction}, {distance}"
        self.eval_data.choices = []
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question


if __name__ == "__main__":
    from ..utils.eval_utilities import create_and_plot_room, manual_test_loop
    from .task_types import EvalTaskType

    # Robustness test suggestions:
    # 1. Coordinate format: "(x, y)", "x,y", "x y".
    # 2. Coordinate list spacing: "(1,2)" vs "(1, 2)".
    # 3. List delimiters: semicolons vs newlines.
    # 4. Mixed Types: Handling when answer expects dict vs string input (auto-parsing).

    task_names = ['fwd_loc', 'bwd_loc_text', 'bwd_loc_vision']
    # task_names = ['fwd_loc']

    for task_name in task_names:
        print(f"\nTesting task: {task_name}")
        try:
            room, agent, np_random = create_and_plot_room(seed=2)
            task = EvalTaskType.create_task(task_name, np_random=np_random, room=room, agent=agent)
            
            manual_test_loop(task_name, task, EvalTaskType.evaluate_prediction)

        except ValueError as e:
            print(f"Skipping {task_name}: {e}")

