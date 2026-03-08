from enum import Enum
from typing import Any, Dict, Optional, Tuple, Type, TYPE_CHECKING
import numpy as np

from ..core.room import Room
from ..core.object import Agent
from ..utils.eval_utilities import evaluate_task_answer
if TYPE_CHECKING:
    from .tasks import BaseEvaluationTask

class EvalTaskType(Enum):
    """Enum for all available evaluation task types."""
    
    # Task type definitions: (short_name, class_name)
    DIR = ("dir", "DirectionEvaluationTask")
    ROT = ("rot", "RotEvaluationTask")
    POV = ("pov", "PovEvaluationTask")
    BWD_POV_TEXT = ("bwd_pov_text", "BackwardPovTextEvaluationTask")
    E2A = ("e2a", "AlloMappingEvaluationTask")
    FWD_LOC = ("fwd_loc", "Location2ViewEvaluationTask")
    BWD_LOC_TEXT = ("bwd_loc_text", "View2LocationTextEvaluationTask")
    FWD_FOV = ("fwd_fov", "Action2ViewEvaluationTask")
    BWD_NAV_TEXT = ("bwd_nav_text", "View2ActionTextEvaluationTask")

    # vision
    BWD_POV_VISION = ("bwd_pov_vision", "BackwardPovVisionEvaluationTask")
    BWD_LOC_VISION = ("bwd_loc_vision", "View2LocationVisionEvaluationTask")
    BWD_NAV_VISION = ("bwd_nav_vision", "View2ActionVisionEvaluationTask")

    # useless
    BWD_NAV_VISION_OLD = ("bwd_nav_vision", "Location2ActionVisionEvaluationTask")  # deprecated
    ROT_DUAL = ("rot_dual", "RotDualEvaluationTask")
    BWD_NAV_REV = ("bwd_nav_rev", "View2ActionRevEvaluationTask")
    DIR_ANCHOR = ("dir_anchor", "DirectionPov")
    
    def __init__(self, short_name: str, class_name: str):
        self.short_name = short_name
        self.class_name = class_name
    
    @classmethod
    def get_short_names(cls) -> list[str]:
        """Get all short names for task types."""
        return [task.short_name for task in cls]
    
    @classmethod
    def get_class_names(cls) -> list[str]:
        """Get all class names for task types."""
        return [task.class_name for task in cls]

    @classmethod
    def excluded_from_average(cls) -> set[str]:
        """Task identifiers (short or class name) excluded from overall evaluation averages."""
        excluded = (cls.ROT_DUAL, cls.BWD_POV_VISION, cls.BWD_NAV_VISION, cls.BWD_LOC_VISION, cls.BWD_NAV_VISION_OLD)
        return {t.short_name for t in excluded} | {t.class_name for t in excluded}
    
    @classmethod
    def get_task_map(cls) -> Dict[str, 'Type[BaseEvaluationTask]']:
        """Get mapping from short names to task classes."""
        # Import here to avoid circular imports
        from .direction import DirectionEvaluationTask, PovEvaluationTask, BackwardPovTextEvaluationTask, BackwardPovVisionEvaluationTask, DirectionPov
        from .rotation import RotEvaluationTask
        from .e2a import AlloMappingEvaluationTask
        from .localization import Location2ViewEvaluationTask, View2LocationTextEvaluationTask, View2LocationVisionEvaluationTask
        from .navigation_tasks import Action2ViewEvaluationTask, View2ActionTextEvaluationTask, View2ActionVisionEvaluationTask, View2ActionRevEvaluationTask
        
        task_map = {
            cls.DIR.short_name: DirectionEvaluationTask,
            cls.ROT.short_name: RotEvaluationTask,
            cls.POV.short_name: PovEvaluationTask,
            cls.DIR_ANCHOR.short_name: DirectionPov,
            cls.E2A.short_name: AlloMappingEvaluationTask,
            cls.FWD_LOC.short_name: Location2ViewEvaluationTask,
            cls.BWD_LOC_TEXT.short_name: View2LocationTextEvaluationTask,
            cls.BWD_LOC_VISION.short_name: View2LocationVisionEvaluationTask,
            cls.FWD_FOV.short_name: Action2ViewEvaluationTask,
            cls.BWD_NAV_TEXT.short_name: View2ActionTextEvaluationTask,
            cls.BWD_NAV_VISION.short_name: View2ActionVisionEvaluationTask,
            cls.BWD_NAV_REV.short_name: View2ActionRevEvaluationTask,
            cls.BWD_POV_TEXT.short_name: BackwardPovTextEvaluationTask,
            cls.BWD_POV_VISION.short_name: BackwardPovVisionEvaluationTask,
        }
        return task_map
    
    @classmethod
    def get_class_map(cls) -> Dict[str, 'Type[BaseEvaluationTask]']:
        """Get mapping from class names to task classes."""
        task_map = cls.get_task_map()
        return {task.class_name: task_class for task, task_class in 
                zip(cls, task_map.values())}

    @classmethod
    def resolve_class_name(cls, task_name: str) -> str:
        """Resolve short or long task identifier to class name."""
        task_name = cls.migrate_legacy_name(task_name)
        if task_name in cls.get_short_names():
            return cls.from_short_name(task_name).class_name
        if task_name in cls.get_class_names():
            return task_name
        raise ValueError(f"Unknown task identifier: {task_name}")

    @classmethod
    def migrate_legacy_name(cls, task_name: str) -> str:
        """Migrate legacy task names to current ones."""
        mapping = {
            # Forward: Action2Location (Old) -> Location2View (New)
            "Action2LocationEvaluationTask": "Location2ViewEvaluationTask",
            "Action2LocationTextEvaluationTask": "Location2ViewEvaluationTask",
            "Action2LocationVisionEvaluationTask": "Location2ViewEvaluationTask",

            # Backward: Location2Action (Old) -> View2Location (New)
            "Location2ActionEvaluationTask": "View2LocationTextEvaluationTask", 
            "Location2ActionTextEvaluationTask": "View2LocationTextEvaluationTask",
            "Location2ActionVisionEvaluationTask": "View2LocationVisionEvaluationTask",
        }
        return mapping.get(task_name, task_name)

    @classmethod
    def evaluate_prediction(
        cls,
        task_name: str,
        pred: Any,
        answer: Any,
        choices: Optional[list[str]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate a prediction for the given task identifier."""
        class_name = cls.resolve_class_name(task_name)
        return evaluate_task_answer(class_name, pred, answer, choices or [])
    
    @classmethod
    def from_short_name(cls, short_name: str) -> 'EvalTaskType':
        """Get task type from short name."""
        for task in cls:
            if task.short_name == short_name:
                return task
        raise ValueError(f"Unknown task short name: {short_name}")
    
    @classmethod
    def from_class_name(cls, class_name: str) -> 'EvalTaskType':
        """Get task type from class name."""
        for task in cls:
            if task.class_name == class_name:
                return task
        raise ValueError(f"Unknown task class name: {class_name}")
    
    @classmethod
    def create_task(cls, task_name: str, np_random: np.random.Generator, room: 'Room', agent: 'Agent', config: dict = None, history_manager = None) -> 'BaseEvaluationTask':
        """Create an evaluation task instance from task name."""
        task_map = cls.get_task_map()
        if task_name in task_map:
            task_class = task_map[task_name]
            # By default, evaluation questions assume the agent returns to the *initial* state.
            # Only View2ActionRevEvaluationTask ("bwd_nav_rev") starts from the *final* pose.
            a = agent.copy()
            if task_name != cls.BWD_NAV_REV.short_name:
                a.pos = a.init_pos.copy()
                a.ori = a.init_ori.copy()
                if a.init_room_id is not None:
                    a.room_id = a.init_room_id
            return task_class(np_random, room, a, config or {}, history_manager)
        else:
            raise ValueError(f"Unknown evaluation task: {task_name}")
