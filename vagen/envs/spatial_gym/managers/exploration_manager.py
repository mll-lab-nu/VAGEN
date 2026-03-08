import copy
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional, Set, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
 
import random

from ..core.object import Agent
from ..core.room import Room
from .spatial_solver import SpatialSolver
from ..utils.cogmap.unexplored import distances_to_explored

if TYPE_CHECKING:
    from ..actions import ActionSequence
    
from ..actions import *

@dataclass
class ExplorationTurnLog:
    """Log data for a single exploration turn."""
    node_coverage: float
    edge_coverage: float
    step: int
    action_counts: Dict[str, int]
    observed_items: List[str]
    visible_objects: List[str]
    is_action_fail: bool = False
    room_state: Optional['Room'] = None
    agent_state: Optional['Agent'] = None
    information_gain: Optional[float] = None  # Information gain (uses exploration quality metric)
    possible_positions: Optional[Dict[str, List[List[int]]]] = None  # Sampled possible positions per object
    unexplored_positions_by_room: Optional[Dict[str, List[List[int]]]] = None  # Unexplored positions per room (room_id -> [[x,y], ...])
    all_candidate_coords: Optional[List[Tuple[int, int]]] = None  # All candidate coordinates (unexplored + distractors)
    all_correct_coords: Optional[List[Tuple[int, int]]] = None  # All correct unexplored coordinates
    all_candidate_dists: Optional[List[float]] = None  # Distance of each candidate coord to explored region

    def to_dict(self):
        return {
            "node_coverage": self.node_coverage,
            "edge_coverage": self.edge_coverage,
            "step": self.step,
            "observed_items": self.observed_items,
            "visible_objects": self.visible_objects,
            "is_action_fail": self.is_action_fail,
            "action_counts": dict(self.action_counts),
            "room_state": self.room_state.to_dict() if self.room_state else {},
            "agent_state": self.agent_state.to_dict() if self.agent_state else {},
            "information_gain": self.information_gain or 0.0,
            "possible_positions": self.possible_positions or {},
            "unexplored_positions_by_room": self.unexplored_positions_by_room or {},
            "all_candidate_coords": [[int(x), int(y)] for x, y in (self.all_candidate_coords or [])],
            "all_correct_coords": [[int(x), int(y)] for x, y in (self.all_correct_coords or [])],
            "all_candidate_dists": [float(d) for d in (self.all_candidate_dists or [])],
        }

class ExplorationManager:
    """Minimal exploration manager without graphs.

    - Keeps copies of `room` and `agent` for simulation.
    - Executes actions and logs turns.
    - Graph-related metrics default to safe zeros.
    """
    MAX_POSSIBLE_POSITIONS_PER_OBJECT: int = 200
    DEFAULT_ACTION_COUNTS = {'move': 0, 'rotate': 0, 'return': 0, 'observe': 0, 'term': 0, 'forced_term': 0}
    DEFAULT_EXP_SUMMARY = {
        "node_coverage": 0.0, "edge_coverage": 0.0, "n_exploration_steps": 0,
        "action_counts": {}, "action_cost": 0, "exploration_cost": 0,
        "info_gain_list": [], "acc_info_gain": 0.0, "avg_info_gain": 0.0,
    }
    def __init__(self, room: Room, agent: Agent, grid_size: int | None = None, seed: int | None = None):
        self.base_room = room.copy()
        self.exploration_room = room.copy()
        self.agent = agent.copy()
        self.seed = seed
        self._rng = random.Random(seed)

        self.turn_logs: List[ExplorationTurnLog] = []
        # History now stores ActionResult for each executed action (in order)
        self.history: List['ActionResult'] = []

        # For uncertainty modeling: per-room unexplored/explored positions
        self._unexplored_by_room: Dict[str, Set[Tuple[int, int]]] = {}
        self._explored_by_room: Dict[str, Set[Tuple[int, int]]] = {}
        self._visited_rooms: Set[str] = set()  # Rooms that have been observed at least once
        self._init_unexplored()
        
        # Coverage tracking (exclude gates)
        self._init_node_name = "initial_pos"
        self.init_pos = self.agent.init_pos.copy()
        self._init_room_id = int(self.agent.init_room_id)

        # Node names: all objects in the exploration room
        self.node_names: List[str] = [o.name for o in self.exploration_room.all_objects]

        # Edge targets: per-room object pairs + (init, object-in-init-room)
        self.target_edges: Set[frozenset] = set()
        for rid, names in self.exploration_room.objects_by_room.items():
            names += self.exploration_room.gates_by_room.get(rid, [])
            if not names:
                continue
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    self.target_edges.add(frozenset({a, b}))
        for name in self.exploration_room.objects_by_room[self._init_room_id] + self.exploration_room.gates_by_room.get(self._init_room_id, []):
            self.target_edges.add(frozenset({self._init_node_name, name}))
        
        self.observed_nodes: Set[str] = set()
        self.known_edges: Set[frozenset] = set()

        # Action counts and costs
        self.action_counts: Dict[str, int] = self.DEFAULT_ACTION_COUNTS.copy()
        self.action_cost: int = 0
        # Observed names (objects and gates) to gate JumpTo() eligibility
        self.observed_items: Set[str] = set()
        self.visible_objects: List[str] = []
        # Grid size for solver metrics (use provided or infer from mask; fallback 10)
        inferred_g = (max(self.exploration_room.mask.shape) if getattr(self.exploration_room, 'mask', None) is not None else 10)
        self.grid_size: int = int(inferred_g if grid_size is None else grid_size)
        # Spatial solver for info gain / quality
        self.spatial_solver = SpatialSolver(self.node_names + ['initial_pos'], self.grid_size)
        self.spatial_solver.set_initial_position('initial_pos', (0, 0))
        
    def _execute_and_update(self, action: BaseAction, **kwargs) -> ActionResult:
        """Execute action and update exploration state."""
        # Inject observed-items context for actions that require observed gating.
        if action.requires_observed_items():
            kwargs['observed_items'] = list(self.observed_items)
        result = action.execute(self.exploration_room, self.agent, **kwargs)
        # Log every action result to history immediately
        self.history.append(result)
        if not result.success:
            return result
        # Count action, cost, and update coverage
        self.action_counts[result.action_type] = self.action_counts.get(result.action_type, 0) + 1
        self.action_cost += int(action.cost)
        if isinstance(action, ObserveAction):
            room_ids = self.agent.room_id if isinstance(self.agent.room_id, list) else [self.agent.room_id]
            for rid in room_ids:
                self._subtract_fov(
                    int(self.agent.pos[0]), int(self.agent.pos[1]),
                    int(self.agent.ori[0]), int(self.agent.ori[1]),
                    str(rid)
                )
            self._update_coverage_from_observe(result)
        
        return result


    def execute_action(self, action: BaseAction) -> ActionResult:
        """Execute single action and return result."""
        return self._execute_and_update(action)
    
    def execute_success_action(self, action: BaseAction, **kwargs) -> ActionResult:
        """Execute single action and return result (must be successful)."""
        result = self._execute_and_update(action, **kwargs)
        assert result.success, f"Action {action} with kwargs {kwargs} failed: {result.message}"
        return result

    def execute_action_sequence(self, action_sequence: 'ActionSequence') -> List[ActionResult]:
        """
        Execute a sequence of motion actions followed by a final action.
        If any motion action fails, execute an observe action and end.
        Returns list of action results.
        """
        assert action_sequence.final_action, "Action sequence requires a final action."

        action_results = []
        is_action_fail = False
        # Execute motion actions
        for action in action_sequence.motion_actions:
            result = self._execute_and_update(action)
            action_results.append(result)
            if not result.success:
                is_action_fail = True
                # On failure, perform an observe action and end
                obs_result = self._execute_and_update(ObserveAction())
                obs_result.message = f"Subsequent actions are skipped due to failure, instead an observe is executed: {obs_result.message}"
                action_results.append(obs_result)
                assert obs_result.success, f"Observe action failed: {obs_result.message}"
                self._log_exploration(action_results, is_action_fail)
                return action_results

        # Execute final action
        final_action = action_sequence.final_action
        result = self._execute_and_update(final_action)
        action_results.append(result)
        if not result.success:
            is_action_fail = True
        # Always log before return
        self._log_exploration(action_results, is_action_fail)
        return action_results
    
    def get_exp_summary(self) -> Dict[str, Any]:
        """Get exploration summary."""
        node_cov = len(self.observed_nodes) / len(self.node_names)
        edge_cov = len(self.known_edges) / len(self.target_edges)
        info_gain_list = [turn_log.information_gain for turn_log in self.turn_logs] if self.turn_logs else []
        acc_info_gain = sum(info_gain_list)
        avg_info_gain = acc_info_gain / len(self.turn_logs) if self.turn_logs else 0.0
        return {
            "node_coverage": node_cov,
            "edge_coverage": edge_cov,
            "n_exploration_steps": len(self.turn_logs),
            "action_counts": dict(self.action_counts),
            "action_cost": int(self.action_cost),
            "exploration_cost": int(self.action_cost),
            "info_gain_list": info_gain_list,
            "acc_info_gain": acc_info_gain,
            "avg_info_gain": avg_info_gain,
        }
    
    @staticmethod
    def aggregate_group_performance(env_data_list: List[Dict] = None) -> Dict[str, Any]:
        """Calculate exploration performance for a group from env_data_list.
        Prefer using precomputed per-sample metrics if present.
        """
        if not env_data_list:
            return {}

        pre = [((s.get('metrics') or {}).get('exploration') or {}) for s in env_data_list]

        def _avg_key(k: str) -> float | None:
            vals = [p.get(k) for p in pre if isinstance(p.get(k), (int, float))]
            return (sum(vals) / len(vals)) if vals else None

        result = {
            'avg_node_coverage': _avg_key('last_node_coverage'),
            'avg_edge_coverage': _avg_key('last_edge_coverage'),
            'avg_exploration_steps': _avg_key('n_exploration_steps'),
            'avg_action_cost': _avg_key('action_cost'),
            'avg_action_fail_ratio': _avg_key('action_fail_ratio'),
            'avg_valid_action_ratio': _avg_key('valid_action_ratio'),
            'avg_final_information_gain': _avg_key('final_information_gain'),
            'infogain_per_turn': ExplorationManager._avg_lists_carry_forward([p.get('infogain_per_turn') or [] for p in pre]),
        }

        # Average action counts
        agg_counts: Dict[str, float] = {}
        n = 0
        for p in pre:
            ac = p.get('action_counts')
            if not isinstance(ac, dict) or not ac:
                continue
            n += 1
            for a, c in ac.items():
                agg_counts[a] = agg_counts.get(a, 0.0) + float(c)
        if agg_counts and n:
            for a in list(agg_counts.keys()):
                agg_counts[a] /= n
            result['avg_action_counts'] = agg_counts
        return result
    
    @staticmethod
    def _avg_lists_carry_forward(list_of_lists: List[List[float]]) -> List[float]:
        list_of_lists = [lst for lst in (list_of_lists or []) if isinstance(lst, list) and lst]
        if not list_of_lists:
            return []
        max_len = max((len(lst) for lst in list_of_lists), default=0)
        if max_len == 0:
            return []
        padded: List[List[float]] = []
        for lst in list_of_lists:
            last = lst[-1]
            if len(lst) < max_len:
                lst = lst + [last] * (max_len - len(lst))
            padded.append(lst)
        out: List[float] = []
        for i in range(max_len):
            vals = [lst[i] for lst in padded if isinstance(lst[i], (int, float))]
            out.append((sum(vals) / len(vals)) if vals else 0.0)
        return out

    @staticmethod
    def aggregate_per_sample(env_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate exploration metrics within a single sample.
        - node/edge coverage (last)
        - total action cost
        - action counts
        - per-turn information gain and final information gain
        - exploration steps
        - is_action_fail and is_valid_action proportions
        """
        env_turn_logs = env_data.get('env_turn_logs', [])
        last_exp = None
        for t in reversed(env_turn_logs):
            if t.get('is_exploration_phase', False) and t.get('exploration_log'):
                last_exp = t['exploration_log']
                break
        has_exp = bool(last_exp)
        node_cov = float(last_exp.get('node_coverage')) if has_exp and isinstance(last_exp.get('node_coverage'), (int, float)) else None
        edge_cov = float(last_exp.get('edge_coverage')) if has_exp and isinstance(last_exp.get('edge_coverage'), (int, float)) else None
        steps = int(last_exp.get('step')) if has_exp and isinstance(last_exp.get('step'), (int, float)) else None
        # Approximate action counts and cost: derive from last turn summary fields if present
        action_counts = None
        if has_exp:
            default_counts = ExplorationManager.DEFAULT_ACTION_COUNTS.copy()
            action_counts = (last_exp.get('action_counts') if ('action_counts' in last_exp) else {}) or {}
            # ensure default keys exist
            for k in default_counts:
                action_counts[k] = int(action_counts.get(k, 0))
        # Compute action cost if not present using known costs
        action_cost = last_exp.get('action_cost') if has_exp and ('action_cost' in last_exp) else None
        if action_cost is None and action_counts:
            # default costs aligned with action classes
            default_costs = {
                'move': 0,
                'rotate': 0,
                'return': 0,
                'observe': 1,
                'term': 0,
                'forced_term': 0,
            }
            action_cost = 0
            for k, v in action_counts.items():
                c = default_costs.get(k.lower(), 0)
                try:
                    action_cost += int(v) * int(c)
                except Exception:
                    continue
        if action_cost is None and has_exp:
            action_cost = 0
        info_gain_list = last_exp.get('info_gain_list') if has_exp else None
        if info_gain_list is None:
            # rebuild from per-turn logs
            info_gain_list = []
            for t in env_turn_logs:
                if t.get('is_exploration_phase', False):
                    ig = (t.get('exploration_log') or {}).get('information_gain')
                    if ig is not None:
                        info_gain_list.append(ig)
        final_infogain = (float(info_gain_list[-1]) if info_gain_list else None)

        # Calculate proportions of is_action_fail and is_valid_action across all turns
        total_turns = len(env_turn_logs)
        action_fail_count = 0
        valid_action_count = 0

        for t in env_turn_logs:
            # Count is_action_fail from exploration_log
            if t.get('is_exploration_phase', False) and t.get('exploration_log'):
                if t['exploration_log'].get('is_action_fail', False):
                    action_fail_count += 1

            # Count is_valid_action from info
            if t.get('info', {}).get('is_valid_action', True):  # Default to True if not present
                valid_action_count += 1

        action_fail_ratio = (action_fail_count / total_turns) if total_turns > 0 else None
        valid_action_ratio = (valid_action_count / total_turns) if total_turns > 0 else None

        return {
            'last_node_coverage': node_cov,
            'last_edge_coverage': edge_cov,
            'n_exploration_steps': steps,
            'action_counts': action_counts,
            'action_cost': action_cost,
            'infogain_per_turn': info_gain_list or [],
            'final_information_gain': final_infogain,
            'action_fail_ratio': action_fail_ratio,
            'valid_action_ratio': valid_action_ratio,
        }
    
    # No passive history generation here; proxies produce text histories directly.
    
    # === Coverage helpers ===
    def _anchor_name(self) -> Optional[str]:
        # If standing on an object position, use that object as anchor (exclude gates)
        for obj in self.exploration_room.all_objects:
            if np.allclose(obj.pos, self.agent.pos):
                return obj.name
        # Initial position anchor
        if np.allclose(self.agent.pos, self.init_pos):
            return self._init_node_name
        raise ValueError("No anchor found")

    def _update_coverage_from_observe(self, observe_result: 'ActionResult') -> None:
        visible = observe_result.data.get('visible_objects', []) or []
        # node coverage
        for name in visible:
            self.observed_items.add(name)
            if name in self.node_names:
                self.observed_nodes.add(name)
        # edge coverage: observe A from B (B is anchor)
        anchor = self._anchor_name()
        for name in visible:
            if name == anchor:
                continue
            pair = frozenset({anchor, name})
            if pair in self.target_edges:
                self.known_edges.add(pair)


    
    def _log_exploration(self, action_results: List['ActionResult'], is_action_fail = False) -> None:
        """Log exploration history and efficiency."""
        # First ingest latest observations, then compute info gain as exploration quality
        for ar in action_results:
            if getattr(ar, 'action_type', None) in ('observe'):
                self.visible_objects = ar.data.get('visible_objects', []) or []
            self._calculate_single_action_information_gain(ar)
        turn_quality = self._compute_exploration_quality()
        # Snapshot possible positions per object (only initialized/observed ones), with sampling
        possible_positions = self._get_possible_positions_snapshot(self.MAX_POSSIBLE_POSITIONS_PER_OBJECT)
        
        # For uncertainty modeling
        # Compute unexplored positions per room based on FOV history
        unexplored_positions_by_room = self._compute_unexplored_positions_by_room()
        # Generate all candidate and correct coordinates for unexplored areas
        all_candidate_coords, all_correct_coords = self._generate_all_correct_coords(unexplored_positions_by_room)

        # Distance-to-explored for each candidate coord (global frame)
        explored_global: Set[Tuple[int, int]] = set()
        for rid, pts in (self._explored_by_room or {}).items():
            if rid not in self._visited_rooms:
                continue
            for p in (pts or set()):
                explored_global.add(p)
        all_candidate_dists = distances_to_explored(all_candidate_coords, explored_global)
        
        step_idx = len(self.turn_logs) + 1
        turn_log = ExplorationTurnLog(
            node_coverage=len(self.observed_nodes) / len(self.node_names),
            edge_coverage=len(self.known_edges) / len(self.target_edges),
            observed_items=list(self.observed_items),
            visible_objects=self.visible_objects,
            step=step_idx,
            is_action_fail=is_action_fail,
            action_counts=self.action_counts,
            room_state=self.exploration_room.copy(),
            agent_state=self.agent.copy(),
            information_gain=turn_quality if turn_quality is not None else (self.turn_logs[-1].information_gain if self.turn_logs else 0.0),
            possible_positions=possible_positions,
            unexplored_positions_by_room=unexplored_positions_by_room,
            all_candidate_coords=all_candidate_coords,
            all_correct_coords=all_correct_coords,
            all_candidate_dists=all_candidate_dists,
        )
        self.turn_logs.append(turn_log)
    

    def _get_possible_positions_snapshot(self, max_per_obj: Optional[int] = None) -> Dict[str, List[List[int]]]:
        """Return sampled possible positions for each observed object (exclude 'initial_pos')."""
        try:
            variables = self.spatial_solver.solver.variables if self.spatial_solver else {}
        except Exception:
            return {}
        snapshot: Dict[str, List[List[int]]] = {}
        for name, var in variables.items():
            if name == 'initial_pos':
                continue
            # Only include objects that have been observed at least once
            if name not in self.observed_items:
                continue
            dom = getattr(var, 'domain', None)
            if dom is None or len(dom) == 0:
                continue
            pts = list(dom)
            if max_per_obj is not None and len(pts) > int(max_per_obj):
                pts = self._rng.sample(pts, int(max_per_obj))
            snapshot[name] = [list(p) for p in pts]
        return snapshot
    
    def _calculate_single_action_information_gain(self, action_result: 'ActionResult') -> float:
        """Ingest observation triples into the solver; return 0.0.
        Keep simple: we use exploration quality as info gain elsewhere.
        """
        if getattr(action_result, 'action_type', None) == 'observe':
            triples = action_result.data.get('relation_triples', []) if hasattr(action_result, 'data') else []
            if triples:
                keep = set(self.spatial_solver.solver.variables.keys())
                filt = [tr for tr in triples if tr.subject in keep and tr.anchor in keep]
                if filt:
                    self.spatial_solver.add_observation(filt)

    # === Exploration quality helpers ===
    def _init_unexplored(self) -> None:
        unique_vals = np.unique(self.exploration_room.mask)
        for val in unique_vals:
            if 1 <= val < 100:  # Valid room IDs are 1-99
                rid = int(val)
                room_coords = np.argwhere(self.exploration_room.mask == rid)
                self._unexplored_by_room[str(rid)] = set((int(c[0]), int(c[1])) for c in room_coords)
                self._explored_by_room[str(rid)] = set()

    def _subtract_fov(
        self, px: int, py: int, ox: int, oy: int, room_id: str, fov_angle: int = 90
    ) -> None:
        """Remove visible positions from unexplored for the given room."""
        unexplored = self._unexplored_by_room.get(room_id)
        if not unexplored:
            return
        explored = self._explored_by_room.setdefault(room_id, set())
        
        # Mark this room as visited
        self._visited_rooms.add(room_id)
        
        # Normalize orientation
        ori_len = np.sqrt(ox**2 + oy**2)
        if ori_len == 0:
            return
        ox_n, oy_n = ox / ori_len, oy / ori_len
        
        # Agent's current position is observed
        unexplored.discard((px, py))
        explored.add((px, py))
        
        # Robustness: treat points within (fov_angle + eps_deg) as visible.
        eps_deg = 2e-3
        half_fov = np.radians((fov_angle + eps_deg) / 2.0)
        cos_half = float(np.cos(half_fov))
        to_remove = []
        
        for (tx, ty) in unexplored:
            dx, dy = tx - px, ty - py
            dist = np.sqrt(dx**2 + dy**2)
            if dist == 0:
                continue
            dot = ox_n * (dx / dist) + oy_n * (dy / dist)
            # Robust boundary handling: dot and cos can be numerically noisy near edges.
            dot = max(-1.0, min(1.0, float(dot)))
            if dot >= cos_half:
                to_remove.append((tx, ty))
        
        for pos in to_remove:
            unexplored.discard(pos)
            explored.add(pos)
    
    def _compute_unexplored_positions_by_room(self) -> Dict[str, List[List[int]]]:
        """Get unexplored positions for each visited room.
        
        Only returns rooms that have been observed at least once.
        Rooms that have never been visited are not included.
        
        Returns:
            Dict mapping room_id (str) to list of [x, y] unexplored positions
        """
        return {
            rid: [[x, y] for x, y in sorted(positions)]
            for rid, positions in self._unexplored_by_room.items()
            if rid in self._visited_rooms
        }
    
    def _generate_all_correct_coords(
        self,
        unexplored_positions_by_room: Dict[str, List[List[int]]]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Generate candidate and correct coordinates.

        We sample per room from both:
        - unexplored (correct)
        - explored (distractors)

        Constraints:
        - equal number per room (unexplored == explored)
        - max 3 per room
        - if a room has either side empty, skip that room
        - if no available points across rooms, return empty lists (caller should skip)
        - POINTS MUST BE INSIDE ROOMS (mask > 0), excluding walls and outside
        
        Args:
            unexplored_positions_by_room: Dict mapping room_id to unexplored positions
            
        Returns:
            Tuple of (all_candidate_coords, all_correct_coords)
        """
        all_candidate_coords: List[Tuple[int, int]] = []
        all_correct_coords: List[Tuple[int, int]] = []
        
        # Get agent's current position to exclude it
        agent_pos = (int(self.agent.pos[0]), int(self.agent.pos[1]))
        
        # Helper to check if a point is inside a valid room (mask > 0)
        # Note: Gates might also be valid depending on mask, but usually gates have separate IDs or are part of room. 
        # Here we assume mask > 0 means "inside room or gate".
        # If user strictly wanted "not gate", we'd check if it overlaps a gate object.
        # Given "exclude gates" requirement, we check if point is in any gate pos.
        gate_positions = set()
        if hasattr(self.exploration_room, 'gates'):
            for g in self.exploration_room.gates:
                gate_positions.add((int(g.pos[0]), int(g.pos[1])))

        def is_valid_room_point(p: Tuple[int, int]) -> bool:
            x, y = p
            # Check boundaries
            if not (0 <= x < self.exploration_room.mask.shape[0] and 0 <= y < self.exploration_room.mask.shape[1]):
                return False
            # Check mask (inside room)
            if self.exploration_room.mask[x, y] <= 0:
                return False
            # Check not a gate
            if p in gate_positions:
                return False
            return True

        for rid_str, unexplored_list in unexplored_positions_by_room.items():
            if not unexplored_list:
                continue
            
            # Filter unexplored points
            unexplored_set = set()
            for p in unexplored_list:
                pt = (int(p[0]), int(p[1]))
                if is_valid_room_point(pt):
                    unexplored_set.add(pt)

            # Filter explored points
            explored_raw = self._explored_by_room.get(rid_str, set()) or set()
            explored_set = set()
            for p in explored_raw:
                pt = (int(p[0]), int(p[1]))
                if is_valid_room_point(pt):
                    explored_set.add(pt)
            
            # Remove agent's current position from both sets
            unexplored_set.discard(agent_pos)
            explored_set.discard(agent_pos)
            
            if not unexplored_set or not explored_set:
                continue

            k = min(3, len(unexplored_set), len(explored_set))
            if k <= 0:
                continue

            unexplored_samples = [p for p in self._rng.sample(list(unexplored_set), k)]
            explored_samples = [p for p in self._rng.sample(list(explored_set), k)]

            all_correct_coords.extend(unexplored_samples)
            all_candidate_coords.extend(unexplored_samples)
            all_candidate_coords.extend(explored_samples)
        
        # Randomize candidate order so prompts don't leak structure (keep correct coords unchanged).
        self._rng.shuffle(all_candidate_coords)
        return all_candidate_coords, all_correct_coords

    def _full_grid_cell_count(self) -> int:
        return int(self.grid_size) * int(self.grid_size)

    def _final_position_counts(self) -> Dict[str, int]:
        """Counts of possible positions per variable at the end of exploration.
        Uses existing solver if available, otherwise rebuilds a solver from history.
        """
        if self.spatial_solver is not None:
            return self.spatial_solver.get_num_possible_positions()
        # Build a temporary solver and ingest history triples
        solver = SpatialSolver(self.node_names + ['initial_pos'], self.grid_size)
        solver.set_initial_position('initial_pos', (0, 0))
        for ar in self.history:
            try:
                if getattr(ar, 'action_type', None) == 'observe':
                    triples = ar.data.get('relation_triples', []) if hasattr(ar, 'data') else []
                    if triples:
                        solver.add_observation(triples)
            except Exception:
                continue
        return solver.get_num_possible_positions()

    def _compute_exploration_quality(self) -> float | None:
        """Compute quality = sum_i log2(M/Ci) / (N * log2(M)). Exclude 'initial_pos'. Include gates.
        Returns None if computation is not applicable.
        """
        try:
            counts = self._final_position_counts()
            M = self._full_grid_cell_count()
            if M <= 1:
                return 0.0
            names = [n for n in counts.keys() if n != 'initial_pos']
            if not names:
                return 0.0
            denom = len(names) * np.log2(M)
            if denom <= 0:
                return 0.0
            total = 0.0
            for n in names:
                Ci = max(1, int(counts.get(n, M)))
                total += float(np.log2(M / Ci))
            return float(total / denom)
        except Exception:
            return None

if __name__ == "__main__":
    pass