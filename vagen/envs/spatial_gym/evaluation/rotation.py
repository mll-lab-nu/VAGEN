"""Rotation-related evaluation tasks."""

from typing import List, Tuple
import numpy as np
from typing_extensions import override

from .tasks import BaseEvaluationTask, retry_generate_question
from ..core.object import Object
from ..core.relationship import PairwiseRelationship
from ..utils.utils import hash

"""
Task Overview:
1. RotEvaluationTask: List objects appearing during 360° rotation.
   - Evaluated by: exact sequence match.
2. RotDualEvaluationTask: Infer rotation direction from object sequence.
   - Evaluated by: correct direction (CW/CCW).
"""

ROT_EVAL_TEMPLATE = (
    "You return to your starting position and face north.\n"
    "You will perform a full 360-degree rotation by continuously turning {turn_direction} in place.\n"
    "Assume all walls are removed (you can see through walls), so every object is visible.\n"
    "Focus on this set of objects: {object_pool}.\n"
    "List them in the exact order they appear directly ahead while you rotate.\n"
    "If two objects share a bearing, place the nearer one first.\n\n"
    "Answer format: <object_name1>, <object_name2>, ...\n"
    "Example: mug, sofa, plant\n"
)

class RotEvaluationTask(BaseEvaluationTask):
    """Ask the sequence of objects appearing when rotating in place."""

    QUESTION_TEMPLATE = ROT_EVAL_TEMPLATE

    # ---------- helpers ----------
    def _get_object_info(self, obj: Object, turn_dir: str) -> Tuple[float, float]:
        """Get angle and distance for an object relative to agent position and turn direction."""
        bearing = float(PairwiseRelationship.get_bearing_degree(tuple(obj.pos), tuple(self.agent.pos), anchor_ori=tuple(self.agent.ori)))
        distance = float(PairwiseRelationship.get_distance(tuple(obj.pos), tuple(self.agent.pos)).value)
        angle = (bearing % 360.0) if turn_dir == "clockwise" else ((-bearing) % 360.0)
        return angle, distance

    def _gen_valid_sequence(self, turn_dir: str, start_eps: float) -> List[str]:
        candidates = []
        for o in self.room.objects:
            if np.array_equal(o.pos, self.agent.pos):
                continue
            ang, dist = self._get_object_info(o, turn_dir)
            if ang < 1e-3 or ang > 360.0 - 1e-3:
                continue
            candidates.append((o.name, ang, dist))

        if len(candidates) < 3:
             raise ValueError(f"Too few objects available ({len(candidates)}) for rotation task")

        eps_schedule = list(dict.fromkeys(max(v,1) for v in [start_eps, start_eps/2, start_eps/4, 1.0]))

        for eps in eps_schedule:
            for _ in range(5):
                self.np_random.shuffle(candidates)
                selected = []

                for item in candidates:
                    _, ang, _ = item
                    # Check distance from ALL currently selected
                    if all(abs(ang - s[1]) > eps for s in selected):
                        selected.append(item)

                if len(selected) >= 3:
                    selected.sort(key=lambda x: (x[1], x[2]))
                    count = self.np_random.integers(3, min(len(selected), 5) + 1)
                    return [x[0] for x in selected[:count]]
        
        raise ValueError("Failed to generate valid rotation sequence")

    # ---------- main ----------
    @retry_generate_question
    def generate_question(self) -> str:
        self.turn_direction = self.np_random.choice(["clockwise", "counterclockwise"])
        self.angle_eps = float(self.config.get("angle_eps", 30.0))

        correct_seq = self._gen_valid_sequence(self.turn_direction, self.angle_eps)
        object_pool = ", ".join(sorted(set(correct_seq)))

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            turn_direction=self.turn_direction,
            object_pool=object_pool,
        )
        self.eval_data.answer = correct_seq
        self.eval_data.choices = []
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question

    @override
    def to_string(self) -> str:
        return f"{self.__class__.__name__}({self.turn_direction})"


if __name__ == "__main__":
    from ..utils.eval_utilities import create_and_plot_room, manual_test_loop
    from .task_types import EvalTaskType

    # Robustness test suggestions:
    # 1. Sequence delimiters: Commas, semicolons, or newlines.
    # 2. Sequence order: Verify strict ordering vs set matching (rotation tasks usually require order).
    # 3. Synonym matching: "CW" for "clockwise", "CCW" for "counterclockwise".
    # 4. Starting point shift: (A, B, C) might be equivalent to (B, C, A) depending on task definition (usually not for this task).

    task_names = ['rot']
    
    for task_name in task_names:
        print(f"\nTesting task: {task_name}")
        try:
            room, agent, np_random = create_and_plot_room(seed=2, plot=True)
            task = EvalTaskType.create_task(task_name, np_random=np_random, room=room, agent=agent)
            
            manual_test_loop(task_name, task, EvalTaskType.evaluate_prediction)

        except ValueError as e:
            print(f"Skipping {task_name}: {e}")