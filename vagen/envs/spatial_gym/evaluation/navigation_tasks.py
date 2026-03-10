"""Forward/Backward navigation tasks with shared helpers.

ForwardFOVEvaluationTask: predict final observation from an action sequence.
BackwardNavEvaluationTask: infer action sequence from a final observation.
BackwardNavRevEvaluationTask: navigate back to starting point from termination location.
"""

from typing import Any, List, Tuple, Dict
import numpy as np
import json

from .tasks import BaseEvaluationTask, retry_generate_question
from ..core.object import Agent, Gate
from ..core.relationship import PairwiseRelationshipDiscrete, EgoFrontBins, StandardDistanceBins
from ..actions import ObserveAction, RotateAction, MoveAction
from ..managers.exploration_manager import ExplorationManager
from ..utils.utils import hash, compute_shortest_path

"""
Task Overview:
1. Action2ViewEvaluationTask (Forward FOV): Predict final observation (of target object) after actions.
   - Evaluated by: direction and distance match of target object.
2. View2ActionEvaluationTask (Backward Nav): Infer action sequence from final view.
   - Evaluated by: simulating actions and checking final state/view match.
3. View2ActionRevEvaluationTask: Navigate back to start from end location.
   - Evaluated by: success of reaching start and visibility.
"""

ACTION_2_VIEW_TEMPLATE = (
    "You return to your starting position and face north.\n"
    "You will execute the following action sequence:\n"
    "{actions}\n\n"
    "After executing the actions, what is the ego relation of {target} relative to you?\n\n"
    "Answer format: <ego direction>, <distance>\n"
    "Example: front, near\n"
)

VIEW_2_ACTION_TEMPLATE = (
    "You return to your starting position and face north.\n"
    "Then you have executed an action sequence and changed to a new location and facing direction.\n"
    "You observe the following:\n"
    "{final_obs}\n\n"
    "What action sequence led to this final view? The action sequence must be valid and only contain move actions.\n\n"
    "Answer format: <sequence of move actions>\n"
    "Example: JumpTo(lamp), Rotate(90)\n"
)

VIEW_2_ACTION_REV_TEMPLATE = (
    "You are currently at the termination location.\n"
    "What action sequence will navigate you back to your starting position? The action sequence must be valid and only contain move actions.\n\n"
    "You must end with a JumpTo(initial_pos) action.\n"
    "Answer format: <sequence of move actions>\n"
    "Example: JumpTo(lamp), Rotate(90), JumpTo(initial_pos)\n"
)

# Nav action descriptor: ('rotate', degrees) or ('jumpto', object_name)
NavAction = Tuple[str, Any]

# ---- Small helpers ----
def _closest_cardinal(vec: np.ndarray) -> np.ndarray:
    """Return closest cardinal direction vector."""
    basis = [np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0])]
    dots = [float(np.dot(vec, b)) for b in basis]
    return basis[int(np.argmax(dots))]


def _ordinal(n: int) -> str:
    n = int(n)
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _nearfar_phrase(index: int, total: int) -> str:
    index = int(index)
    total = int(total)
    if index == 1:
        return "nearest"
    if index == total:
        return "farthest"
    return f"{_ordinal(index)} nearest"


def _ori_to_deg(ori: Tuple[int, int]) -> int:
    """Convert orientation tuple to degrees."""
    mapping = {(0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270}
    return mapping[tuple(int(x) for x in ori)]


def _rotate_ori(ori: Tuple[int, int], degrees: int) -> Tuple[int, int]:
    mapping = {0: (0, 1), 90: (1, 0), 180: (0, -1), 270: (-1, 0)}
    cur = _ori_to_deg(ori)
    new_deg = (cur + degrees) % 360
    return mapping.get(new_deg, ori)


def _rotation_delta(current: Tuple[int, int], desired: Tuple[int, int]) -> int:
    cur = _ori_to_deg(current)
    des = _ori_to_deg(desired)
    return (des - cur + 540) % 360 - 180


class BaseNavEvaluationTask(BaseEvaluationTask):
    """Shared navigation helpers for both tasks."""

    def _agent_from_init(self) -> Agent:
        """Reset agent to initial position and orientation."""
        a = self.agent.copy()
        a.pos = self.agent.init_pos.copy()
        a.ori = self.agent.init_ori.copy()
        a.room_id = getattr(self.agent, 'init_room_id', None) or a.room_id
        if a.room_id is None:
            info = self.room.get_cell_info(int(a.pos[0]), int(a.pos[1]))
            a.room_id = info.get('room_id', a.room_id)
        return a

    def _move_simple(self, agent: Agent, name: str) -> None:
        """Move agent to object position."""
        obj = self.room.get_object_by_name(name)
        agent.pos = obj.pos.copy()
        agent.room_id = obj.room_id

    def _current_rooms(self, agent: Agent) -> List[int]:
        """Get current room IDs for agent."""
        rid = getattr(agent, 'room_id', None)
        if isinstance(rid, list):
            return [int(x) for x in rid]
        if rid is None:
            info = self.room.get_cell_info(int(agent.pos[0]), int(agent.pos[1]))
            rid = info.get('room_id')
        return [int(rid)] if rid is not None else []

    def _candidates_in_rooms(self, rooms: List[int]) -> List[str]:
        """Get all object names in given rooms."""
        # If no rooms specified (e.g., BaseRoom without mask), return all objects
        if not rooms:
            return [obj.name for obj in self.room.all_objects]

        names: List[str] = []
        for rid in rooms:
            names.extend(self.room.objects_by_room.get(int(rid), []))
            if hasattr(self.room, 'gates_by_room'):
                names.extend(self.room.gates_by_room.get(int(rid), []))
        return list(dict.fromkeys(names))

    def _generate_plan(self, steps: int = 3) -> List[NavAction]:
        """Generate navigation plan with exactly 'steps' jumpto actions (and necessary rotations)."""
        a = self._agent_from_init()
        plan: List[NavAction] = []
        last_was_gate = False
        other_rooms_after_gate: List[int] = []

        move_count = 0
        while move_count < int(steps):
            rooms = self._current_rooms(a)
            cand = [n for n in self._candidates_in_rooms(rooms)
                   if not np.allclose(self.room.get_object_by_name(n).pos, a.pos)]
            if not cand:
                raise ValueError(f"Cannot generate {steps} moves: no candidates available after {move_count} moves")

            gate_cand = [n for n in cand if isinstance(self.room.get_object_by_name(n), Gate)]
            non_gate = [n for n in cand if n not in gate_cand]

            if last_was_gate:
                objects_in_other_rooms = [n for n in non_gate
                                         if self.room.get_object_by_name(n).room_id in other_rooms_after_gate]
                pool = objects_in_other_rooms or non_gate or gate_cand
                name = str(self.np_random.choice(pool))
            else:
                if gate_cand and int(self.np_random.integers(0, 10)) < 6:
                    name = str(self.np_random.choice(gate_cand))
                else:
                    name = str(self.np_random.choice(non_gate or cand))

            target = self.room.get_object_by_name(name)
            desired_ori = _closest_cardinal(target.pos - a.pos)
            delta = _rotation_delta(tuple(a.ori), tuple(desired_ori))

            # Add rotation only if needed
            if int(delta) != 0:
                plan.append(('rotate', int(delta)))
                a.ori = np.array(_rotate_ori(tuple(a.ori), int(delta)))

            plan.append(('jumpto', name))
            self._move_simple(a, name)
            move_count += 1

            if isinstance(self.room.get_object_by_name(name), Gate):
                last_was_gate = True
                gobj = self.room.get_object_by_name(name)
                other_rooms_after_gate = [int(r) for r in list(gobj.room_id) if int(r) not in rooms]
            else:
                last_was_gate = False
                other_rooms_after_gate = []

        # Align to a valid viewing orientation with at least one visible object
        valid_oris = []
        for ori in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            tmp = a.copy()
            tmp.ori = np.array(ori)
            if ObserveAction().execute(self.room, tmp).data.get('visible_objects', []):
                valid_oris.append(ori)
        final_ori = tuple(valid_oris[int(self.np_random.integers(0, len(valid_oris)))] if valid_oris else (0, 1))
        delta = _rotation_delta(tuple(a.ori), tuple(final_ori))
        if int(delta) != 0:
            plan.append(('rotate', int(delta)))
            a.ori = np.array(_rotate_ori(tuple(a.ori), int(delta)))

        return plan

    def _execute_plan(self, plan: List[NavAction]) -> Agent:
        """Execute navigation plan and return final agent state."""
        mgr = ExplorationManager(self.room.copy(), self._agent_from_init())
        for action_type, value in plan:
            if action_type == 'rotate':
                mgr.execute_success_action(RotateAction(int(value)))
            elif action_type == 'jumpto':
                target = mgr.exploration_room.get_object_by_name(str(value))
                assert MoveAction._is_visible(mgr.agent, target), f"Target '{value}' must be visible before JumpTo."
                mgr.observed_items.add(str(value))
                mgr.execute_success_action(MoveAction(str(value)), move_anyway=True)
            else:
                raise ValueError(f"Unknown navigation action: {action_type}")
        return mgr.agent.copy()

    def _describe_target(self, mgr: ExplorationManager, target_name: str) -> str:
        bin_sys = EgoFrontBins()
        dist_sys = StandardDistanceBins()
        target = self.room.get_object_by_name(target_name)
        rel_t = PairwiseRelationshipDiscrete.relationship(
            tuple(target.pos),
            tuple(mgr.agent.pos),
            anchor_ori=tuple(mgr.agent.ori),
            bin_system=bin_sys,
            distance_bin_system=dist_sys,
        )
        dir_label = rel_t.direction.bin_label
        dist_label = rel_t.dist.bin_label

        # Find all objects that match BOTH direction AND distance bins
        same_bin_group = []
        visible = mgr.execute_success_action(ObserveAction()).data.get('visible_objects', [])
        for name in visible:
            obj = self.room.get_object_by_name(name)
            rel = PairwiseRelationshipDiscrete.relationship(
                tuple(obj.pos),
                tuple(mgr.agent.pos),
                anchor_ori=tuple(mgr.agent.ori),
                bin_system=bin_sys,
                distance_bin_system=dist_sys,
            )
            # Only include objects that match BOTH direction and distance bins
            if (int(rel.direction.bin_id) == int(rel_t.direction.bin_id) and 
                int(rel.dist.bin_id) == int(rel_t.dist.bin_id)):
                actual_dist = float(np.linalg.norm(np.array(obj.pos) - np.array(mgr.agent.pos)))
                same_bin_group.append((obj, float(rel.direction.degree), actual_dist))

        dir_phrase = None
        dist_phrase = None
        
        if len(same_bin_group) > 1:
            # Sort by angular degree (left to right)
            dir_sorted = sorted(same_bin_group, key=lambda item: item[1])
            dir_idx = 1 + next(i for i, (obj, _, _) in enumerate(dir_sorted) if obj.name == target.name)
            dir_phrase = f"{_ordinal(dir_idx)} from left"
            
            # Sort by distance (near to far)
            dist_sorted = sorted(same_bin_group, key=lambda item: item[2])
            dist_idx = 1 + next(i for i, (obj, _, _) in enumerate(dist_sorted) if obj.name == target.name)
            dist_phrase = f"{_nearfar_phrase(dist_idx, len(same_bin_group))} one"

        if dir_phrase and dist_phrase:
            descriptors = " also ".join(filter(None, (dir_phrase, dist_phrase)))
            return f"Among objects which are {dir_label}, {dist_label} to you, you jump to the {descriptors}."

        return f"Jump to the object at {dir_label}, {dist_label}."

    def _plan_to_text(self, plan: List[NavAction]) -> str:
        mgr = ExplorationManager(self.room.copy(), self._agent_from_init())
        steps = []
        for idx, (act, value) in enumerate(plan, start=1):
            if act == 'rotate':
                mgr.execute_success_action(RotateAction(int(value)))
                label = f"Rotate({int(value)})"
            elif act == 'jumpto':
                name = str(value)
                target = mgr.exploration_room.get_object_by_name(name)
                description = self._describe_target(mgr, name)
                assert MoveAction._is_visible(mgr.agent, target), f"Target '{name}' must be visible before JumpTo."
                mgr.observed_items.add(name)
                mgr.execute_success_action(MoveAction(name), move_anyway=True)
                label = description
            else:
                raise ValueError(f"Unknown navigation action: {act}")
            steps.append(f"{idx}. {label}")
        return "\n".join(steps)

    def _sample_plan_with_visible(self, steps: int, max_attempts: int = 5) -> Tuple[List[NavAction], Agent, List[Dict[str, str]], Dict[str, Tuple[int, int]]]:
        attempts = max(1, min(int(self.config.get('plan_retry', max_attempts)), max_attempts))
        
        def plan_candidate_generator():
            for _ in range(attempts):
                plan = self._generate_plan(steps)
                agent = self._execute_plan(plan)
                yield (plan, agent)

        try:
            best_cand, visible, obj_orientations = self._select_best_candidate(
                candidates=plan_candidate_generator(),
                get_agent_func=lambda c: c[1],
                min_visible=1,
                max_obs=3
            )
            return best_cand[0], best_cand[1], visible, obj_orientations
        except ValueError as e:
            # Re-raise with specific message to match expectations or keep original
            raise ValueError(f"Failed to generate navigation plan with visible objects: {e}")

class Action2ViewEvaluationTask(BaseNavEvaluationTask):
    """Predict final observation from an action sequence."""

    QUESTION_TEMPLATE = ACTION_2_VIEW_TEMPLATE

    @retry_generate_question
    def generate_question(self) -> str:
        steps = int(self.config.get('steps', 2))
        plan, end_agent, visible, _ = self._sample_plan_with_visible(steps)
        actions_str = self._plan_to_text(plan)

        self.np_random.shuffle(visible)
        # visible is list of dicts now
        first = visible[0]
        target_name = first['name']
        direction = first['direction']
        distance = first['distance']
        
        answer = f"{direction}, {distance}"

        self.eval_data.question = self.QUESTION_TEMPLATE.format(actions=actions_str, target=target_name)
        self.eval_data.answer = answer
        self.eval_data.choices = []
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question

class BaseView2ActionEvaluationTask(BaseNavEvaluationTask):
    """Base class for View2Action (Backward Navigation) tasks."""

    QUESTION_TEMPLATE = VIEW_2_ACTION_TEMPLATE

    def _get_final_obs(self, visible: List[Dict[str, str]]) -> str:
        raise NotImplementedError

    @retry_generate_question
    def generate_question(self) -> str:
        steps = int(self.config.get('steps', 3))
        plan, end_agent, visible, obj_orientations = self._sample_plan_with_visible(steps)
        
        final_obs = self._get_final_obs(visible)

        # Store expected final state and object positions for evaluation
        init_agent = self._agent_from_init()
        object_positions = {obj.name.lower(): tuple(map(int, obj.pos)) for obj in self.room.all_objects}
        
        # Build answer with orientations
        # Merge orientations: local visible ones + global ones if needed.
        # Ideally satisfy _eval_backward_nav which needs orientations for checking.
        # We assume room.all_objects has them, so we can re-extract global map of orientations
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

        answer = {
            'final_pos': tuple(map(int, end_agent.pos)),
            'final_ori': tuple(map(int, end_agent.ori)),
            'room_id': (list(end_agent.room_id) if isinstance(end_agent.room_id, (list, tuple)) else int(end_agent.room_id)) if end_agent.room_id is not None else None,
            'init_pos': tuple(map(int, init_agent.pos)),
            'init_ori': tuple(map(int, init_agent.ori)),
            'object_positions': object_positions,
            'object_orientations': all_orientations,
            'gate_info': gate_info,
            'object_rooms': object_rooms,
            "minimal_plan": compute_shortest_path(
                self.room,
                init_agent.pos,
                init_agent.ori,
                end_agent.pos,
                end_agent.ori,
            ),
            'final_observation': visible, # List of dicts
        }

        self.eval_data.question = self.QUESTION_TEMPLATE.format(final_obs=final_obs)
        self.eval_data.answer = answer
        self.eval_data.choices = []
        self.eval_data.id = hash(self.eval_data.question + json.dumps(answer, sort_keys=True, default=str)) # Use default=str for numpy types
        return self.eval_data.question


class View2ActionTextEvaluationTask(BaseView2ActionEvaluationTask):
    """Infer action sequence from final observation (text)."""
    def _get_final_obs(self, visible: List[Dict[str, str]]) -> str:
        obs_parts = []
        for v in visible:
            txt = f"{v['name']} is at {v['direction']}, {v['distance']}"
            if v.get('orientation'):
                txt += f", {v['orientation']}"
            obs_parts.append(txt)
        return "; ".join(obs_parts)


class View2ActionVisionEvaluationTask(BaseView2ActionEvaluationTask):
    """Infer action sequence from final observation (vision)."""
    def _get_final_obs(self, visible: List[Dict[str, str]]) -> str:
        return "You observe: <image>"

class View2ActionRevEvaluationTask(BaseNavEvaluationTask):
    """Navigate back to starting point from termination location."""

    QUESTION_TEMPLATE = VIEW_2_ACTION_REV_TEMPLATE

    @retry_generate_question
    def generate_question(self) -> str:
        # Store initial and final states for evaluation
        # Current position is self.agent.pos (termination location)
        # Initial position is self.agent.init_pos
        start_pos = tuple(map(int, self.agent.pos))
        start_ori = tuple(map(int, self.agent.ori))
        target_pos = tuple(map(int, self.agent.init_pos))
        target_ori = tuple(map(int, self.agent.init_ori))
        object_positions = {obj.name: tuple(map(int, obj.pos)) for obj in self.room.all_objects}

        # Compute shortest path
        minimal_plan = compute_shortest_path(
            self.room,
            start_pos,
            start_ori,
            target_pos,
        )

        answer = {
            'start_pos': start_pos,  # Starting from termination location
            'start_ori': start_ori,
            'target_pos': target_pos,  # Target is the initial position
            'target_ori': target_ori,
            'object_positions': object_positions,
            'minimal_plan': minimal_plan,  # Action list for shortest path
        }

        self.eval_data.question = self.QUESTION_TEMPLATE.format()
        self.eval_data.answer = answer
        self.eval_data.choices = []
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question


def _plan_to_action_str(plan: List[NavAction]) -> str:
    """Convert a plan list to action string format.
    
    Example: [('rotate', -180), ('jumpto', 'television')] -> "Rotate(-180), JumpTo(television)"
    """
    parts = []
    for action_type, value in plan:
        if action_type == 'rotate':
            parts.append(f"Rotate({value})")
        elif action_type == 'jumpto':
            parts.append(f"JumpTo({value})")
    return ", ".join(parts)


if __name__ == "__main__":
    from ..utils.eval_utilities import create_and_plot_room, manual_test_loop
    from .task_types import EvalTaskType

    # Robustness test suggestions:
    # 1. Action formats: "JumpTo(obj)", "jumpto obj", "JumpTo( obj )".
    # 2. Case insensitivity for actions and arguments.
    # 3. Delimiters: Comma vs semicolon vs newline for action sequences.
    # 4. Action abbreviations: "Rotate" vs "Rot" (if supported).

    # task_names = ['fwd_fov', 'bwd_nav_text', 'bwd_nav_vision']
    task_names = ['bwd_nav_text']

    for task_name in task_names:
        print(f"\nTesting task: {task_name}")
        try:
            room, agent, np_random = create_and_plot_room(seed=2)
            task = EvalTaskType.create_task(task_name, np_random=np_random, room=room, agent=agent)
            
            # Helper to display friendly answer for navigation tasks
            if isinstance(task.answer, dict) and 'minimal_plan' in task.answer:
                friendly_pred = _plan_to_action_str(task.answer['minimal_plan'])
                print(f"Computed Minimal Plan (Ground Truth): {friendly_pred}")
            
            manual_test_loop(task_name, task, EvalTaskType.evaluate_prediction)

        except ValueError as e:
            print(f"Skipping {task_name}: {e}")
