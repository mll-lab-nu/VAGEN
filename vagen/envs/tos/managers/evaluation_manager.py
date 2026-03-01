"""Simple Evaluation Manager for SpatialGym Environment"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..evaluation.task_types import EvalTaskType
from ..core.room import Room
from ..core.object import Agent
from ..evaluation.tasks import BaseEvaluationTask, EvaluationData

@dataclass
class EvaluationTurnLog:
    """Log data for a single evaluation turn."""
    task_type: str
    user_answer: str
    score: bool
    evaluation_info: Dict[str, Any]
    evaluation_data: EvaluationData
    room_state: Optional['Room'] = None
    agent_state: Optional['Agent'] = None

    def to_dict(self):
        evaluation_data = self.evaluation_data.to_dict()
        if "question" in evaluation_data:
            evaluation_data.pop("question")
        return {
            "task_type": self.task_type,
            "user_answer": self.user_answer,
            "score": self.score,
            "room_state": self.room_state.to_dict() if self.room_state else {},
            "agent_state": self.agent_state.to_dict() if self.agent_state else {},
            "evaluation_info": self.evaluation_info,
            "evaluation_data": evaluation_data
        }


class EvaluationManager:
    """
    Manages evaluation tasks for the SpatialGym environment.
    
    Handles task initialization, question generation, answer evaluation,
    and tracking of evaluation results across multiple tasks.
    """
    def __init__(
        self,
        eval_tasks: List[Dict[str, Any]],
        np_random: np.random.Generator,
        room: Room,
        agent: Agent,
        history_manager=None,
        seed: int | None = None,
        render_mode: str = 'text'
    ):
        # In current implementation, only one evaluation task is allowed
        assert len(eval_tasks) == 1, "Only one evaluation task is supported"
        self.history_manager = history_manager
        self.eval_tasks = []
        spec = eval_tasks[0]
        ttype = spec['task_type']

        if render_mode == 'text':
            assert 'vision' not in ttype, f"Cannot run vision task {ttype} in text mode"
        num = int(spec.pop('num', 1))
        counts = self.history_manager.get_eval_counts() if self.history_manager else {}
        done_count = 0
        if counts:
            class_name = EvalTaskType.from_short_name(ttype).class_name
            done_count = counts.get(class_name, 0)

        if num > done_count:
            self.eval_tasks = [spec]
        self.np_random = np_random
        self.results = []
        self.turn_logs: List[EvaluationTurnLog] = []
        self.seed = seed
        
        # Initialize tasks
        self.tasks = []
        for idx, task_spec in enumerate(self.eval_tasks):
            task_type = task_spec['task_type']
            rng_seed = int(self.seed) if self.seed is not None else None
            task = EvalTaskType.create_task(task_type, np.random.default_rng(rng_seed), room.copy(), agent.copy(), {}, history_manager)
            self.tasks.append(task)
            self.results.append({
                "task_type": task.__class__.__name__,
                "score": 0,
                "info": {}
            })
        self.current_index = 0
    
    def _get_current_eval_task(self) -> Optional[BaseEvaluationTask]:
        """Get current evaluation task."""
        assert self.current_index < len(self.tasks), "No more tasks"
        return self.tasks[self.current_index]

    # ---------------- Aggregations ----------------
    @staticmethod
    def aggregate_per_sample(env_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate evaluation metrics within one sample (counts and accuracy)."""
        tasks = env_data.get('evaluation_tasks') or {}
        per_task = {}
        for task_type, questions in tasks.items():
            if not questions:
                continue
            n_total = len(questions)
            task_score = sum(float(q.get('evaluation_log', {}).get('score', 0)) for q in questions.values())
            per_task[task_type] = {
                'n_total': n_total,
                'task_score': task_score,
                'avg_accuracy': (task_score / n_total) if n_total else None,
            }

        excluded = EvalTaskType.excluded_from_average()
        total = sum(v['n_total'] for k, v in per_task.items() if k not in excluded)
        total_score = sum(v['task_score'] for k, v in per_task.items() if k not in excluded)
        return {
            'overall': {'n_total': total, 'total_score': total_score, 'avg_accuracy': (total_score / total) if total else None},
            'per_task': per_task,
        }

    @staticmethod
    def aggregate_group_performance(env_data_list: List[Dict] = None) -> Dict[str, Any]:
        """Group aggregation; use precomputed per-sample metrics or reuse per-sample aggregation."""
        if not env_data_list:
            return {'avg_accuracy': 0.0, 'task_metrics': {}}

        per_samples = [EvaluationManager.aggregate_per_sample(s) for s in env_data_list]
        total_count = sum(int(m.get('overall',{}).get('n_total', 0)) for m in per_samples)
        total_score = sum(float(m.get('overall',{}).get('total_score', 0)) for m in per_samples)
        agg_task: Dict[str, Dict[str, int]] = {}
        for m in per_samples:
            for t, tm in (m.get('per_task') or {}).items():
                d = agg_task.setdefault(t, {'total': 0, 'task_score': 0})
                d['total'] += int(tm.get('n_total', 0))
                d['task_score'] += float(tm.get('task_score', 0))
        task_metrics = {t: {
            'accuracy': (v['task_score'] / v['total']) if v['total'] else 0.0,
            'total_count': v['total'],
            'task_score': v['task_score'],
        } for t, v in agg_task.items()}
        return {'avg_accuracy': (total_score / total_count) if total_count else 0.0, 'task_metrics': task_metrics}
    
    def reset(self):
        """Reset to start."""
        self.current_index = 0
        self.results = []
    



if __name__ == "__main__":
    pass