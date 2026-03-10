"""
Base evaluation definitions (data and abstract base classes).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Iterable, Callable
import numpy as np
from dataclasses import dataclass
from functools import wraps

from ..core.room import Room
from ..core.object import Agent, Gate
from ..utils.eval_utilities import evaluate_task_answer, resolve_gate_orientation
from ..actions import RotateAction, ObserveAction
from ..utils.action_utils import action_results_to_text
from ..core.relationship import (PairwiseRelationshipDiscrete, OrientationRel)
from ..utils.utils import hash

# Helper for orientation
def _agent_relative_orientation(agent_ori: np.ndarray, target_ori: np.ndarray) -> str:
    from ..core.relationship import OrientationRel
    return OrientationRel.to_string(
        OrientationRel.get_relative_orientation(tuple(target_ori), tuple(agent_ori)),
        perspective='ego'
    )

@dataclass
class EvaluationData:
    id: str
    question: str
    answer: Any
    task_type: str
    action: Optional[str] = None
    choices: Optional[List[str]] = None
    kwargs: Optional[Dict] = None

    def __post_init__(self):
        # Lazy import to avoid circular dependency during module import
        from .task_types import EvalTaskType  # type: ignore
        valid_task_types = EvalTaskType.get_class_names()
        assert self.task_type in valid_task_types, f"Invalid task type: {self.task_type}"
        if self.choices is None:
            self.choices = []
    
    def evaluate(self, pred: Any) -> Tuple[bool, Dict[str, Any]]:
        return evaluate_task_answer(self.task_type, pred, self.answer, self.choices)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the evaluation data to a dictionary"""
        return {
            'id': self.id,
            'question': self.question,
            'action': self.action,
            'answer': self.answer,
            'task_type': self.task_type,
            'choices': self.choices,
            'kwargs': self.kwargs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationData':
        """Initialize the evaluation data from a dictionary"""
        return cls(**data)


# ---- Lightweight evaluation helper for offline/builder usage ----
def evaluate_from_dict(eval_data_dict: Dict[str, Any], user_answer: Any) -> tuple[bool, Dict[str, Any]]:
    """Evaluate a user's answer given a serialized EvaluationData dict.

    Keeps behavior identical to EvaluationData.evaluate used in env runtime.
    """
    try:
        data = EvaluationData.from_dict(eval_data_dict)
    except Exception:
        # Fallback: minimal dict support
        data = EvaluationData(
            id=str(eval_data_dict.get('id', '')),
            question=str(eval_data_dict.get('question', '')),
            answer=eval_data_dict.get('answer', ''),
            action=eval_data_dict.get('action'),
            task_type=str(eval_data_dict.get('task_type', '')),
            choices=list(eval_data_dict.get('choices', []) or []),
            kwargs=dict(eval_data_dict.get('kwargs', {}) or {}),
        )
    return data.evaluate(user_answer)


class BaseEvaluationTask(ABC):
    """Abstract base class for all spatial evaluation tasks."""
    
    def __init__(self, np_random: np.random.Generator, room: Room, agent: Agent, config: Dict[str, Any] = None, history_manager=None):
        """Initialize the evaluation task"""
        self.config = config or {}
        self.np_random = np_random
        self.room = room.copy()
        self.agent = agent.copy()
        self.history_manager = history_manager
        self.eval_data = EvaluationData(
            id="",
            question="",
            answer="",
            action=None,
            task_type=self.__class__.__name__,
            choices=[],
            kwargs={},
        )

    @property
    def answer(self) -> Any:
        return self.eval_data.answer
    
    @property
    def question(self) -> str:
        return self.eval_data.question
    
    @property
    def choices(self) -> List[str]:
        return self.eval_data.choices
    
    @abstractmethod
    def generate_question(self) -> str:
        """Generate evaluation question based on the current room/agent state."""
        if self.room is None:
            raise ValueError("Room must be set before generating question")
        raise NotImplementedError
    
    def evaluate(self, pred: Any) -> Tuple[bool, Dict[str, Any]]:
        return self.eval_data.evaluate(pred)
    
    def to_string(self) -> str:
        return f"{self.__class__.__name__}()"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the evaluation task to a dictionary"""
        return {
            'question': self.question,
            'answer': self.answer,
            'choices': self.choices,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEvaluationTask':
        raise NotImplementedError

    @classmethod
    def create_task_from_dict(cls, data: Dict[str, Any]) -> 'BaseEvaluationTask':
        from .task_types import EvalTaskType  # Lazy import
        task_types = EvalTaskType.get_class_map()
        task_type = data.get('type', cls.__name__)
        return task_types.get(task_type, cls).from_dict(data)

    def _get_ground_truth_observations(self, agent: Agent, limit: int = None) -> Tuple[List[Dict[str, str]], Dict[str, Tuple[int, int]]]:
        """
        Get ground truth observations and object orientations from a specific agent state.
        Returns:
            - List of observation dicts [{'name', 'direction', 'distance', 'orientation'}]
            - Dict of object orientations {name: (x, y)}
        """
        room_copy = self.room.copy()
        agent_copy = agent.copy()
        
        obs = ObserveAction().execute(room_copy, agent_copy, free_position=True)
        triples = obs.data.get('relation_triples', [])
        
        observations = []
        obj_orientations = {}
        
        for tr in triples:
            rel = getattr(tr, "relation", None)
            if isinstance(rel, PairwiseRelationshipDiscrete):
                obj_name = tr.subject
                target_obj = self.room.get_object_by_name(obj_name)
                
                # Check orientation
                ori_label = None
                if target_obj and target_obj.has_orientation:
                    # Compute relative orientation
                    if isinstance(target_obj, Gate):
                        gate_ori = resolve_gate_orientation(
                            gate_room_ids=target_obj.room_id,
                            gate_ori_by_room={k: tuple(v) for k, v in target_obj.ori_by_room.items()},
                            gate_base_ori=tuple(target_obj.ori),
                            agent_room_ids=agent.room_id
                        )
                        ori_pair = OrientationRel.get_relative_orientation(tuple(gate_ori), tuple(agent.ori))
                        ori_label = OrientationRel.to_string(ori_pair, 'ego', 'orientation', if_gate=True)
                    else:
                        ori_label = _agent_relative_orientation(agent.ori, target_obj.ori)
                    
                    obj_orientations[obj_name] = tuple(int(x) for x in target_obj.ori)

                observations.append({
                    'name': obj_name,
                    'direction': rel.direction.bin_label,
                    'distance': rel.dist.bin_label,
                    'orientation': ori_label
                })
        
        if limit is not None and len(observations) > limit:
            self.np_random.shuffle(observations)
            observations = observations[:limit]

        return observations, obj_orientations

    def _select_best_candidate(
        self,
        candidates: Iterable[Any],
        get_agent_func: Callable[[Any], Agent],
        min_visible: int = 1,
        max_obs: int = 3
    ) -> Tuple[Any, List[Dict], Dict]:
        """
        Select candidate with weighted random sampling based on visible object count.
        Returns (best_candidate, observations, obj_orientations).
        """
        results = []
        for cand in candidates:
            agent = get_agent_func(cand)
            # Use limit=None to get full count
            obs, oris = self._get_ground_truth_observations(agent, limit=None)
            if len(obs) >= min_visible:
                results.append((len(obs), cand, obs, oris))
        
        if not results:
             raise ValueError(f"No candidate found with at least {min_visible} visible objects")
             
        # Weighted random selection: more objects -> higher chance
        weights = np.array([r[0] for r in results], dtype=float)
        idx = self.np_random.choice(len(results), p=weights/weights.sum() if weights.sum() > 0 else None)
        _, best_cand, best_obs_full, best_oris = results[idx]
        
        # Apply limit to observations
        final_obs = best_obs_full
        force_shuffle = bool(getattr(self, "_force_shuffle_final_obs", False))
        if force_shuffle:
            self._force_shuffle_final_obs = False
        if force_shuffle or (max_obs is not None and len(final_obs) > max_obs):
            self.np_random.shuffle(final_obs)
        final_obs = final_obs[:max_obs]
            
        return best_cand, final_obs, best_oris



# ---- Decorator: retry question generation with max retries and history de-dup ----
def retry_generate_question(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retry = int(self.config.get('max_retry', 10))
        for _ in range(max_retry):
            q = func(self, *args, **kwargs)
            # Ensure question field is populated
            if isinstance(q, str) and q:
                self.eval_data.question = q
            q = self.eval_data.question
            # Ensure ID exists for history checks
            if not getattr(self.eval_data, 'id', None) or not self.eval_data.id:
                self.eval_data.id = hash(q)
            # Accept if no history manager or not seen
            hm = getattr(self, 'history_manager', None)
            if not hm or not hm.has_question(self.eval_data.id):
                return q
        raise Exception('Failed to generate question')
    return wrapper