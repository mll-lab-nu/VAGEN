"""Direction and POV evaluation tasks."""

from typing import Iterable, Tuple, List, Dict
import json
import numpy as np
from .tasks import BaseEvaluationTask, retry_generate_question
from ..core.relationship import (
    PairwiseRelationshipDiscrete,
    CardinalBinsAllo,
    EgoFrontBins,
    StandardDistanceBins,
    OrientationRel,
)
from ..core.object import Gate
from ..actions.base import BaseAction
from ..utils.eval_utilities import _is_visible_from
from ..utils.utils import hash

"""
Task Overview:
1. DirectionEvaluationTask: Allo relation between two objects.
   - Evaluated by: direction and distance match.
2. PovEvaluationTask: Ego relation from oriented anchor.
   - Evaluated by: direction and distance match.
3. BackwardPovEvaluationTask: Identify which perspective from Ego relation description.
   - Evaluated by: simulate fov (superset of ground truth).
4. DirectionPov: Ego relation using anchor's facing as North.
   - Evaluated by: direction and distance match.
"""

DIRECTION_EVAL_TEMPLATE = (
    "You return to your starting position and face north.\n"
    "From a Top-Down map, describe where {obj_name} is relative to {anchor_name}.\n"
    "Answer format: <cardinal direction>, <distance>\n"
    "Example: north-west, near\n"
)

POV_EVAL_TEMPLATE = (
    "Now you jump to {anchor_name}'s direction, facing its direction.\n"
    "Describe where {obj_name} is relative to you.\n"
    "Answer format: <ego direction>, <distance>\n"
    "Example: front-left, near\n"
)

BACKWARD_POV_EVAL_TEMPLATE = (
    "Now you jump to an object's position, facing its direction.\n"
    "You observe that {observation}.\n"
    "Which object are you standing at?\n"
    "Answer format: <object_name>\n"
    "Example: lamp\n"
)

DIRECTION_POV_TEMPLATE = (
    "Assume the {anchor_name}'s facing defines 'north' (not true north).\n"
    "Where is {obj_name} relative to {anchor_name}?\n"
    "Answer format: <cardinal direction>, <distance>\n"
    "Example: north-west, near\n"
)

# ---- shared helpers ----
def _store_relation(task: BaseEvaluationTask, question: str, rel: PairwiseRelationshipDiscrete) -> str:
    """Persist relation answer in eval_data."""
    task.eval_data.question = question
    task.eval_data.answer = f"{rel.direction.bin_label}, {rel.dist.bin_label}"
    task.eval_data.choices = []
    task.eval_data.id = hash(question)
    return question


def _visible_relations(room, anchor, rng) -> Iterable[Tuple[object, PairwiseRelationshipDiscrete]]:
    """Yield visible objects in random order with their ego relations."""
    candidates: List = [obj for obj in room.objects if obj is not anchor]
    rng.shuffle(candidates)
    for obj in candidates:
        if not _is_visible_from(
            anchor.pos,
            anchor.ori,
            obj.pos,
            agent_room_id=anchor.room_id,
            target_room_id=obj.room_id,
        ):
            continue
        rel = PairwiseRelationshipDiscrete.relationship(
            tuple(obj.pos),
            tuple(anchor.pos),
            anchor_ori=tuple(anchor.ori),
            bin_system=EgoFrontBins(),
            distance_bin_system=StandardDistanceBins(),
        )
        yield obj, rel


class DirectionEvaluationTask(BaseEvaluationTask):
    """Ask allocentric relation between two objects."""

    QUESTION_TEMPLATE = DIRECTION_EVAL_TEMPLATE

    @retry_generate_question
    def generate_question(self) -> str:
        total = len(self.room.objects)
        if total < 2:
            raise ValueError("Need at least two objects to form a relation")
        objects = list(self.room.objects)
        self.np_random.shuffle(objects)
        obj, anchor = objects[0], objects[1]
        rel = PairwiseRelationshipDiscrete.relationship(
            tuple(obj.pos),
            tuple(anchor.pos),
            bin_system=CardinalBinsAllo(),
        )
        question = self.QUESTION_TEMPLATE.format(obj_name=obj.name, anchor_name=anchor.name)
        return _store_relation(self, question, rel)


class PovEvaluationTask(BaseEvaluationTask):
    """Ask egocentric relation from an oriented anchor's perspective."""

    QUESTION_TEMPLATE = POV_EVAL_TEMPLATE

    @retry_generate_question
    def generate_question(self) -> str:
        oriented = [obj for obj in self.room.objects if obj.has_orientation]
        if not oriented:
            raise ValueError("Need at least one oriented object for POV task")
        self.np_random.shuffle(oriented)
        anchor = None
        visibles: List[Tuple[object, PairwiseRelationshipDiscrete]] = []
        for candidate in oriented:
            visibles = list(_visible_relations(self.room, candidate, self.np_random))
            if visibles:
                anchor = candidate
                break
        if not anchor:
            raise ValueError("No visible objects from available anchors")
        target, rel = self.np_random.choice(visibles)
        question = self.QUESTION_TEMPLATE.format(anchor_name=anchor.name, obj_name=target.name)
        return _store_relation(self, question, rel)


class BaseBackwardPovEvaluationTask(BaseEvaluationTask):
    """Identify which oriented object matches the described egocentric relation."""

    QUESTION_TEMPLATE = BACKWARD_POV_EVAL_TEMPLATE

    def _format_observations(self, obs_list: List[Dict]) -> str:
        raise NotImplementedError

    @retry_generate_question
    def generate_question(self) -> str:
        oriented = [obj for obj in self.room.objects if obj.has_orientation]
        if not oriented:
            raise ValueError("Need an oriented object for backward POV task")
        self.np_random.shuffle(oriented)
        
        # Use shared helper to pick best anchor
        anchor, observations, _ = self._select_best_candidate(
            candidates=oriented,
            get_agent_func=lambda x: x,
            min_visible=1,
            max_obs=3
        )
            
        observation_text = self._format_observations(observations)
        question = self.QUESTION_TEMPLATE.format(observation=observation_text)
        
        self.eval_data.question = question
        
        # Store detailed answer for validation
        object_positions = {obj.name.lower(): tuple(map(float, obj.pos)) for obj in self.room.all_objects}
        
        all_orientations = {obj.name.lower(): tuple(map(int, obj.ori)) for obj in self.room.all_objects if obj.has_orientation}
        
        gate_info = {}
        object_rooms = {}
        for obj in self.room.all_objects:
            name = obj.name.lower()
            if isinstance(obj.room_id, (list, tuple, np.ndarray)):
                object_rooms[name] = [int(x) for x in obj.room_id]
            else:
                object_rooms[name] = int(obj.room_id)

            if isinstance(obj, Gate):
                 gate_info[name] = {
                     'room_ids': [int(x) for x in obj.room_id] if isinstance(obj.room_id, (list, tuple, np.ndarray)) else [int(obj.room_id)],
                     'ori_by_room': {int(k): tuple(map(int, v)) for k, v in obj.ori_by_room.items()}
                 }

        self.eval_data.answer = {
            'answer': anchor.name,
            'final_pos': tuple(map(int, anchor.pos)),
            'final_ori': tuple(map(int, anchor.ori)),
            'final_observation': observations,
            'object_positions': object_positions,
            'object_orientations': all_orientations,
            'room_id': [int(x) for x in anchor.room_id] if isinstance(anchor.room_id, (list, tuple, np.ndarray)) else int(anchor.room_id),
            'gate_info': gate_info,
            'object_rooms': object_rooms
        }
        self.eval_data.choices = []
        # Hash based on question + answer
        self.eval_data.id = hash(json.dumps(anchor.name) + question)
        return question


class BackwardPovTextEvaluationTask(BaseBackwardPovEvaluationTask):
    """Identify which oriented object matches the described egocentric relation (Text)."""

    def _format_observations(self, obs_list: List[Dict]) -> str:
        descriptions = []
        for obs in obs_list:
            parts = [f"{obs['direction']}, {obs['distance']}"]
            if obs.get('orientation'):
                parts.append(obs['orientation'])
            relation_text = ", ".join(parts)
            descriptions.append(f"{obs['name']} is {relation_text}")
        return "; ".join(descriptions)


class BackwardPovVisionEvaluationTask(BaseBackwardPovEvaluationTask):
    """Identify which oriented object matches the described egocentric relation (Vision)."""

    def _format_observations(self, obs_list: List[Dict]) -> str:
        return "You observe: <image>"

class DirectionPov(BaseEvaluationTask):
    """Allocentric relation treating the anchor's facing as north."""

    QUESTION_TEMPLATE = DIRECTION_POV_TEMPLATE

    @retry_generate_question
    def generate_question(self) -> str:
        oriented = [obj for obj in self.room.objects if obj.has_orientation]
        if not oriented:
            raise ValueError("Need an oriented object for DirectionPov task")
        self.np_random.shuffle(oriented)
        anchor = oriented[0]
        others = [obj for obj in self.room.objects if obj is not anchor]
        if not others:
            raise ValueError("DirectionPov task requires another object besides the anchor")
        self.np_random.shuffle(others)
        target = others[0]
        rel = PairwiseRelationshipDiscrete.relationship(
            tuple(target.pos),
            tuple(anchor.pos),
            anchor_ori=tuple(anchor.ori),
            bin_system=CardinalBinsAllo(),
        )
        question = self.QUESTION_TEMPLATE.format(anchor_name=anchor.name, obj_name=target.name)
        return _store_relation(self, question, rel)



if __name__ == "__main__":
    from ..utils.eval_utilities import create_and_plot_room, manual_test_loop
    from .task_types import EvalTaskType
    import numpy as np

    # Robustness test suggestions:
    # 1. Case insensitivity: Ensure answers like "North, Near" and "north, near" are equivalent.
    # 2. Extra whitespace: "north,  near" should be valid.
    # 3. Component swapping: "near, north" should be valid if order doesn't matter (check specific task logic).
    # 4. Partial matching: Verify strict vs loose matching requirements.

    # task_names = ['dir', 'pov', 'bwd_pov_text', 'bwd_pov_vision', 'dir_anchor']
    task_names = ['bwd_pov_text'] # Uncomment to run only one

    room, agent, np_random = create_and_plot_room(seed=3)
    # scanner = room.get_object_by_name('scanner')
    # cabinet = room.get_object_by_name('cabinet')
    # cabinet.ori = scanner.ori
    # cabinet.pos = np.array([12.1, 15])
    # print(room)
    for task_name in task_names:
        print(f"\nTesting task: {task_name}")
        try:
            task = EvalTaskType.create_task(task_name, np_random=np_random, room=room, agent=agent)
            manual_test_loop(task_name, task, EvalTaskType.evaluate_prediction)

        except ValueError as e:
            print(f"Skipping {task_name}: {e}")
