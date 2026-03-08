"""E2A: object coordinates and orientations identification task."""

from typing import List, Tuple
import numpy as np

from .tasks import BaseEvaluationTask, retry_generate_question
from ..core.object import Object
from ..utils.utils import hash

"""
Task Overview:
1. AlloMappingEvaluationTask (E2A): Report allocentric coordinates for objects.
   - Evaluated by: coordinate similarity (position + direction) vs ground truth.
"""

ALLO_MAPPING_TEMPLATE = (
    "Treat your starting position as the origin (0, 0) while facing north.\n"
    "Report allocentric coordinates using (x right/east, y up/north).\n"
    "Objects: {object_list}.\n"
    "Answer format: (x0, y0); (x1, y1); ... in the same order.\n"
    "Example: (1, 0); (-2, 3); (0, -1)\n"
)


class AlloMappingEvaluationTask(BaseEvaluationTask):
    """Report allocentric coordinates for selected objects."""

    QUESTION_TEMPLATE = ALLO_MAPPING_TEMPLATE

    @retry_generate_question
    def generate_question(self) -> str:
        self._selected_objects = self._pick_objects()
        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            object_list=", ".join(obj.name for obj in self._selected_objects)
        )
        self.eval_data.answer = self._collect_coordinates(self._selected_objects)
        self.eval_data.choices = []
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question

    def _pick_objects(self) -> List[Object]:
        pool = list(self.room.objects)
        self.np_random.shuffle(pool)
        count = int(self.np_random.integers(3, min(5, len(pool)) + 1))
        return pool[:count]

    def _collect_coordinates(self, objects: List[Object]) -> List[Tuple[int, int]]:
        ox, oy = map(int, self.agent.init_pos)
        coords: List[Tuple[int, int]] = []
        for obj in objects:
            x, y = map(int, obj.pos)
            coords.append((x - ox, y - oy))
        return coords


if __name__ == "__main__":
    from ..utils.eval_utilities import create_and_plot_room, manual_test_loop
    from .task_types import EvalTaskType

    # Robustness test suggestions:
    # 1. Delimiter variations: Test semicolon vs newline delimiters for coordinate lists.
    # 2. Coordinate format: Test (x,y) vs x,y vs x y.
    # 3. Extra text: Ensure "Object A is at (1, 2)" is parsed correctly if only coordinates are expected.
    # 4. Partial answers: Check behavior when fewer coordinates are provided than requested.

    task_name = 'e2a'
    print(f"\nTesting task: {task_name}")
    try:
        room, agent, np_random = create_and_plot_room(seed=2)
        task = EvalTaskType.create_task(task_name, np_random=np_random, room=room, agent=agent)
        
        manual_test_loop(task_name, task, EvalTaskType.evaluate_prediction)

    except ValueError as e:
        print(f"Skipping {task_name}: {e}")
