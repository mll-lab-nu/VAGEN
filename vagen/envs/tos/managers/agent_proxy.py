import copy
import random
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Set, Any
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm

from ..core.room import Room
from ..actions.base import BaseAction
from ..actions.actions import MoveAction, RotateAction, ObserveAction, TermAction
from .spatial_solver import SpatialSolver
from ..core.relationship import CardinalBinsAllo
from .exploration_manager import ExplorationManager
from ..core.object import Agent
from ..utils.action_utils import action_results_to_text
from ..utils.room_utils import initialize_room_from_json


@dataclass
class Turn:
    actions: List
    pos: tuple
    ori: tuple


def _ori_to_deg(ori: np.ndarray) -> int:
    return int(round(BaseAction._ori_to_angle(ori) / 45.0) * 45) % 360


def _closest_octant(vec: np.ndarray) -> np.ndarray:
    basis = [BaseAction._angle_to_ori(45 * k) for k in range(8)]
    dots = [float(np.dot(vec, b)) for b in basis]
    return basis[int(np.argmax(dots))]


class AgentProxy:
    """Base proxy that executes actions via ExplorationManager and logs a simple history."""

    def __init__(self, room: Room, agent: Agent, grid_size: int | None = None):
        self.mgr = ExplorationManager(room, agent, grid_size=grid_size)
        self.room, self.agent = self.mgr.exploration_room, self.mgr.agent
        # Focused room id for planning when at gates (agent may belong to two rooms at a door)
        self.room_focus = self.agent.room_id

        self.gates_by_room: Dict[int, Set[str]] = {int(r): set(glist) for r, glist in self.room.gates_by_room.items()}

        # nodes with and without gates
        self.object_nodes_by_room: Dict[int, Set[str]] = {}
        for o in self.room.objects:
            self.object_nodes_by_room.setdefault(int(o.room_id), set()).add(o.name)
        # Include rooms that have only gates (no objects)
        room_ids = set(self.object_nodes_by_room.keys()) | {int(r) for r in self.gates_by_room.keys()}
        self.nodes_by_room: Dict[int, Set[str]] = {
            int(rid): set(self.object_nodes_by_room.get(int(rid), set())) | self.gates_by_room.get(int(rid), set())
            for rid in room_ids
        }
        self.known_nodes_by_room: Dict[int, Set[str]] = {int(rid): set() for rid in self.nodes_by_room}

        self.all_edges_by_room: Dict[int, Set[frozenset]] = {}
        for rid, names in self.nodes_by_room.items():
            pairs: Set[frozenset] = set()
            for a in names:
                for b in names:
                    if a != b:
                        pairs.add(frozenset({a, b}))
            self.all_edges_by_room[int(rid)] = pairs
        self.known_edges_by_room: Dict[int, Set[frozenset]] = {int(rid): set() for rid in self.all_edges_by_room}

        # add edges from initial position to objects in the starting room
        self.initial_anchor: str = "initial_pos"
        start_rid = self._current_room()
        for obj_name in self.nodes_by_room.get(start_rid, set()):
            self.all_edges_by_room[start_rid].add(frozenset({self.initial_anchor, obj_name}))

        self.total_nodes: set = set().union(*self.nodes_by_room.values())

        self.turns: List[Turn] = []
        self.visited: Set[int] = set()
        self.current_gate: str | None = None
        self.anchor: str | None = None

    # --- helpers ---
    def _current_room(self) -> int:
        return int(self.room_focus)

    def _entry_anchor_name(self, is_initial: bool) -> str | None:
        return self.initial_anchor if is_initial else self.current_gate

    def _add_turn(self, actions: List) -> None:
        ori = tuple(float(np.round(x, 6)) for x in self.agent.ori.tolist())
        self.turns.append(
            Turn(
                actions=list(actions),
                pos=tuple(int(x) for x in self.agent.pos.tolist()),
                ori=ori,
            )
        )

    def _update_known_from_observe(self, last_result) -> None:
        rid = self._current_room()
        vis = last_result.data.get("visible_objects", [])
        for n in vis:
            self.known_nodes_by_room.setdefault(rid, set()).add(n)
        # Also record edges if an anchor (gate or node) is set
        if self.anchor:
            known = self.known_edges_by_room.setdefault(rid, set())
            for n in vis:
                if n != self.anchor:
                    known.add(frozenset({self.anchor, n}))

    def _adopt_state_from(self, other: 'AgentProxy') -> None:
        """Adopt exploration knowledge/state from another proxy instance."""
        fields = [
            'gates_by_room', 'object_nodes_by_room', 'nodes_by_room',
            'known_nodes_by_room', 'all_edges_by_room', 'known_edges_by_room',
            'visited', 'current_gate', 'anchor', 'room_focus', 'total_nodes'
        ]
        for f in fields:
            setattr(self, f, copy.deepcopy(getattr(other, f)))

    def _rotate_by(self, delta: int) -> List:
        assert delta is not None, "Invalid rotation delta"
        delta = int(delta) % 360
        assert delta % 45 == 0, "Invalid rotation delta"
        if delta > 180:
            delta -= 360
        return [] if delta == 0 else [self.mgr.execute_success_action(RotateAction(delta))]


    def _rotate_to_face(self, target_pos: np.ndarray) -> List:
        vec = target_pos - self.agent.pos
        if np.allclose(vec, 0):
            return []
        desired = _closest_octant(vec)
        cur, des = _ori_to_deg(self.agent.ori), _ori_to_deg(desired)
        return self._rotate_by(des - cur)

    def _rotate_to_ori(self, desired_ori: np.ndarray) -> List:
        cur, des = _ori_to_deg(self.agent.ori), _ori_to_deg(desired_ori)
        return self._rotate_by(des - cur)

    def _move_to(self, name: str) -> List:
        target = self.room.get_object_by_name(name)
        acts = []
        acts += self._rotate_to_face(target.pos)
        acts.append(self.mgr.execute_success_action(MoveAction(name)))
        return acts

    def _move_to_visible(self, name: str) -> List:
        """Rotate up to 8 headings to make target visible, then move."""
        target = self.room.get_object_by_name(name)
        acts: List = []
        base = _closest_octant(target.pos - self.agent.pos)
        for d in (0, 45, 90, 135, 180, 225, 270, 315):
            R = BaseAction._get_rotation_matrix(d)
            desired = base @ R
            acts += self._rotate_to_ori(desired)
            if BaseAction._is_visible(self.agent, target):
                # Only move if not already at the target
                if not np.allclose(self.agent.pos, target.pos):
                    acts.append(self.mgr.execute_success_action(MoveAction(name)))
                return acts
        # Fallback: if still not visible, do not attempt move
        return acts

    def _observe_at_anchor(self, anchor: str, desired_ori: np.ndarray) -> None:
        """Observe at anchor with minimal actions: move only if needed, rotate only if needed."""
        self.anchor = anchor
        at = self.room.get_object_by_name(anchor)
        if np.allclose(self.agent.pos, at.pos):
            self._observe(self._rotate_to_ori(desired_ori))
        else:
            acts = self._move_to_visible(anchor)
            assert np.allclose(self.agent.pos, at.pos), f"Agent pos {self.agent.pos} is not same as anchor pos {at.pos}"
            self._observe(acts + self._rotate_to_ori(desired_ori))
        self.anchor = None

    def _room_node_names(self) -> List[str]:
        rid = self._current_room()
        return sorted(self.nodes_by_room.get(rid, set()))

    def _ingest_last_into_solver(self, solver) -> None:
        if solver is None or not self.turns or not self.turns[-1].actions:
            return
        last = self.turns[-1].actions[-1]
        triples = last.data.get('relation_triples', []) if hasattr(last, 'data') else []
        if not triples:
            return
        keep = set(solver.solver.variables.keys())
        filt = [tr for tr in triples if tr.subject in keep and tr.anchor in keep]
        if filt:
            solver.add_observation(filt)

    def _observe(self, prefix_actions: List = None) -> None:
        acts = list(prefix_actions or [])
        obs = self.mgr.execute_success_action(ObserveAction())
        acts.append(obs)
        self._update_known_from_observe(obs)
        self._add_turn(acts)
        # Log this turn for info gain metrics
        try:
            self.mgr._log_exploration(acts)
        except Exception:
            pass

    def _unknown_nodes_in_room(self, rid: int) -> Set[str]:
        return self.nodes_by_room.get(rid, set()) - self.known_nodes_by_room.get(rid, set())

    def _unknown_edges_in_room(self, rid: int) -> Set[frozenset]:
        return self.all_edges_by_room.get(rid, set()) - self.known_edges_by_room.get(rid, set())

    def _score_rotation_nodes(self, rot: int, unknown_nodes: Set[str]) -> int:
        """How many unknown nodes become visible after rot."""
        if not unknown_nodes:
            return 0
        objects = {o.name: o for o in self.room.all_objects}
        R = BaseAction._get_rotation_matrix(rot)
        tmp = self.agent.copy(); tmp.ori = self.agent.ori @ R
        return sum(1 for n in unknown_nodes if n in objects and BaseAction._is_visible(tmp, objects[n]))

    def _score_rotation_edges(self, rot: int, anchor: str, unknown_edges: Set[frozenset]) -> int:
        """How many unknown edges incident to anchor become visible after rot."""
        anchored = {e for e in unknown_edges if anchor in e}
        if not anchored:
            return 0
        objects = {o.name: o for o in self.room.all_objects}
        R = BaseAction._get_rotation_matrix(rot)
        tmp = self.agent.copy(); tmp.ori = self.agent.ori @ R
        visible = {n for n, o in objects.items() if BaseAction._is_visible(tmp, o)}
        return sum(1 for e in anchored if next(iter(e - {anchor})) in visible)

    def _score_rotation(self, rot: int) -> int:
        """Unified rotation score: edges if anchor set, otherwise nodes."""
        rid = self._current_room()
        if self.anchor:
            return self._score_rotation_edges(rot, self.anchor, self._unknown_edges_in_room(rid))
        return self._score_rotation_nodes(rot, self._unknown_nodes_in_room(rid))

    def _gate_between(self, a: int, b: int) -> str:
        for gate_name, rooms in self.room.rooms_by_gate.items():
            if set(rooms) == {int(a), int(b)}:
                return gate_name
        raise AssertionError(f"No gate between rooms {a} and {b}")

    def _traverse_to(self, next_rid: int) -> List:
        # Move to the connecting gate and face next room; at gate, agent can see both rooms.
        cur = self._current_room()
        gate_name = self._gate_between(cur, next_rid)
        gate_obj = self.room.get_object_by_name(gate_name)
        acts = []
        if not np.allclose(self.agent.pos, gate_obj.pos):
            acts += self._rotate_to_face(gate_obj.pos)
            acts.append(self.mgr.execute_success_action(MoveAction(gate_name)))
        # face next room; TODO may not be needed
        acts += self._rotate_to_ori(-gate_obj.get_ori_for_room(int(cur)))
        self.current_gate = gate_name
        self.room_focus = int(next_rid)
        self.known_nodes_by_room.setdefault(next_rid, set()).add(gate_name)
        return acts

    def _allowed_rotations(self, is_initial: bool, continuous_rotation: bool = True) -> tuple:
        fov = BaseAction.get_field_of_view()
        if continuous_rotation:
            # note continuous rotation: 0 -> (+90) -> 90 -> (+90) -> 180 -> (+90) -> 270
            return (0, 90, 90, 90) if is_initial else ((0,) if fov == 180 else (0, 90, 180))
        else:
            # not continuous: 0, 0 -> 90, 0 -> 180, 0 -> 270
            return (0, 90, 180, 270) if is_initial else ((0,) if fov == 180 else (0, 90, 270))

    def _all_nodes_known_globally(self) -> bool:
        # early stop when all objects (exclude gates) are known
        known_total = set().union(*(objs & self.known_nodes_by_room.get(int(rid), set()) for rid, objs in self.nodes_by_room.items()))
        return known_total == self.total_nodes

    def _subtree_has_nodes(self, start_rid: int) -> bool:
        """check if the subtree has any nodes. Start from start_rid"""
        stack = [int(start_rid)]
        seen = set(copy.deepcopy(self.visited))
        while stack:
            r_id = int(stack.pop())
            if r_id in seen:
                continue
            seen.add(r_id)
            if len(self.nodes_by_room.get(r_id, set())) > 0:
                return True
            stack.extend(int(adj_r_id) for adj_r_id in self.room.adjacent_rooms_by_room.get(r_id, []))
        return False

    # hooks for subclasses
    def _on_entry_observe(self, is_initial: bool, prefix_actions: List = None) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def _explore_room(self, is_initial: bool, prefix_actions: List = None) -> None:
        # default: entry observes only
        self._on_entry_observe(is_initial=is_initial, prefix_actions=prefix_actions)
    
    def _prune_dfs(self, rid: int): # if prune subtree from rid
        return False

    def _dfs(self, rid: int, is_initial: bool, pre_actions: List = None) -> List:
        """DFS over rooms.
        - pre_actions: actions before first observe in this room
        - carry_actions: if no observe here, carry pre_actions into child
        - visited: expanded rooms; stop when all objects are known
        """
        self.visited.add(rid)
        before = len(self.turns)
        self._explore_room(is_initial=is_initial, prefix_actions=pre_actions or [])
        observed_here = len(self.turns) > before
        carry_actions: List = [] if observed_here else list(pre_actions or [])
        if self._all_nodes_known_globally():
            assert not carry_actions, "Carry must be empty if all objects are known"
            return []
        for child_rid in sorted(self.room.adjacent_rooms_by_room.get(rid, [])):
            if child_rid in self.visited:
                continue
            if self._prune_dfs(int(child_rid)):
                continue
            to_child = carry_actions + self._traverse_to(int(child_rid))
            carry_actions = self._dfs(int(child_rid), is_initial=False, pre_actions=to_child)
            # Stop here if exploration is complete; avoid returning to parent.
            if self._all_nodes_known_globally():
                return carry_actions
            carry_actions = carry_actions + self._traverse_to(int(rid))
        return carry_actions

    def run(self) -> List[Turn]:  # to be used by subclasses too
        start_rid = self._current_room()
        _ = self._dfs(start_rid, is_initial=True, pre_actions=[])
        # final turn contains only Term()
        self._add_turn([self.mgr.execute_success_action(TermAction())])
        return self.turns

    def to_text(self, image_placeholder = None) -> str:
        lines: List[str] = []
        for i, t in enumerate(self.turns, 1):
            lines.append(f"{i}. {action_results_to_text(t.actions, image_placeholder, include_action_sequence=True)}")
        return "\n".join(lines)
    
class OracleAgentProxy(AgentProxy):
    """Oracle Agent: greedy rotations to reveal all nodes (knows everything about nodes)."""

    def _on_entry_observe(self, is_initial: bool, prefix_actions: List = None) -> None:
        rid = self._current_room()
        allowed = self._allowed_rotations(is_initial, continuous_rotation=False)
        self.anchor = self._entry_anchor_name(is_initial)
        while self._unknown_nodes_in_room(rid):
            targets = self._unknown_nodes_in_room(rid)
            scores = {d: self._score_rotation_nodes(d, targets) for d in allowed}
            best, best_score = max(scores.items(), key=lambda kv: kv[1]) if scores else (0, 0)
            if best_score == 0:
                break
            self._observe((prefix_actions or []) + self._rotate_by(best))
            prefix_actions = []
        self.anchor = None

    def _prune_dfs(self, rid: int):
        return not self._subtree_has_nodes(rid)

# TODO fix unknown objects in room

class StrategistAgentProxy(AgentProxy):
    """NodeSweeper: simple sweep rotations; may not be optimal or complete."""

    def _on_entry_observe(self, is_initial: bool, prefix_actions: List = None) -> None:
        self.anchor = self._entry_anchor_name(is_initial)
        for d in self._allowed_rotations(is_initial):
            if self._all_nodes_known_globally():
                break
            self._observe((prefix_actions or []) + self._rotate_by(d))
            prefix_actions = []
        self.anchor = None


class InquisitorAgentProxy(AgentProxy):
    """Inquisitor Agent: visit/confirm all edges between nodes (know nothing in prior)"""

    def _on_entry_observe(self, is_initial: bool, prefix_actions: List = None) -> None:
        # treat entry (initial or gate) as anchor so edges are recorded
        self.anchor = self._entry_anchor_name(is_initial)
        for d in self._allowed_rotations(is_initial):
            if self._all_nodes_known_globally(): # early stop if see all objects at entry
                break
            self._observe((prefix_actions or []) + self._rotate_by(d))
            prefix_actions = []
        self.anchor = None

    # move to each node with unknown edges and observe
    def _resolve_edges(self) -> None:
        rid = self._current_room()
        unknown = self._unknown_edges_in_room(rid)
        if not unknown:
            return
        objects_with_unknown_edges = self._anchors_with_unknown_edges(rid)
        while objects_with_unknown_edges:
            self.anchor = objects_with_unknown_edges.pop()
            self._observe(self._move_to(self.anchor))
            for d in (90, 90, 90):
                self._observe(self._rotate_by(d))
                if not any(self.anchor in p for p in self._unknown_edges_in_room(rid)): # no unknown edges from anchor
                    break
            objects_with_unknown_edges = self._anchors_with_unknown_edges(rid)

    # how to choose anchor node with unknown edges
    def _anchors_with_unknown_edges(self, rid: int) -> List[str]:
        unknown = self._unknown_edges_in_room(rid)
        if not unknown:
            return []
        anchors = sorted(self.nodes_by_room.get(rid, set()))
        return [a for a in anchors if any(a in p for p in unknown)]

    def _explore_room(self, is_initial: bool, prefix_actions: List = None) -> None:
        self._on_entry_observe(is_initial=is_initial, prefix_actions=prefix_actions)
        self._resolve_edges()

class GreedyInquisitorAgentProxy(InquisitorAgentProxy):
    """Greedy Inquisitor Agent: pick best rotation each step."""

    def _on_entry_observe(self, is_initial: bool, prefix_actions: List = None) -> None:
        self.anchor = self._entry_anchor_name(is_initial)
        rid = self._current_room()
        while any(self.anchor in p for p in self._unknown_edges_in_room(rid)):
            scores = {d: self._score_rotation(d) for d in (0, 90, 180, 270)}
            best, best_score = max(scores.items(), key=lambda kv: kv[1]) if scores else (0, 0)
            if best_score == 0:
                break
            self._observe((prefix_actions or []) + self._rotate_by(best))
            prefix_actions = []
        self.anchor = None

    def _resolve_edges(self) -> None:
        # move to each node with unknown edges; at each node, rotate greedily by edge score
        rid = self._current_room()
        unknown = self._unknown_edges_in_room(rid)
        if not unknown:
            return
        objects_with_unknown_edges = self._anchors_with_unknown_edges(rid)
        while objects_with_unknown_edges:
            anchor = objects_with_unknown_edges.pop()
            self.anchor = anchor
            prefix_actions = self._move_to(anchor)
            while any(anchor in p for p in self._unknown_edges_in_room(rid)): # exist unknown edges from anchor
                scores = {d: self._score_rotation(d) for d in (0, 90, 180, 270)}
                best, best_score = max(scores.items(), key=lambda kv: kv[1]) if scores else (0, 0)
                if best_score == 0:
                    break
                self._observe(prefix_actions + self._rotate_by(best))
                prefix_actions = []
            objects_with_unknown_edges = self._anchors_with_unknown_edges(rid)
            self.anchor = None



class ObserverAnalystAgentProxy(AgentProxy):
    """Hybrid Agent: (a) observe-all-nodes via oracle/strategist, then (b) pick best observe(anchor,ori) to reduce relations in current room."""

    def __init__(self, room: Room, agent: Agent, grid_size: int | None = None,
                 rel_threshold: int = 0, eval_samples: int = 30, delegate: str = 'oracle', max_observes: int = 100,
                 metric: str = 'positions'):
        super().__init__(room, agent, grid_size=grid_size)
        g = (max(self.room.mask.shape) if getattr(self.room, 'mask', None) is not None else 10)
        self.grid_size = int(g if grid_size is None else grid_size)
        self.rel_threshold = int(rel_threshold)
        self.eval_samples = int(eval_samples)
        self.delegate = (delegate or 'oracle').lower()
        self.max_observes = int(max_observes)
        self.metric = (metric or 'positions').lower()
        # Global solver: all scene nodes + initial_pos
        self.solver: SpatialSolver = SpatialSolver([o.name for o in self.room.all_objects] + ['initial_pos'], grid_size=self.grid_size)
        self.solver.set_initial_position('initial_pos', (0, 0))
        # Room-local solver used only for per-room planning
        self.room_solver: SpatialSolver | None = None

    # ---- step (a): reuse existing in-room observe logic ----
    def _entry_observe_delegate(self, is_initial: bool, prefix_actions: List = None) -> None:
        if self.delegate == 'strategist':
            StrategistAgentProxy._on_entry_observe(self, is_initial=is_initial, prefix_actions=prefix_actions)
        else:
            OracleAgentProxy._on_entry_observe(self, is_initial=is_initial, prefix_actions=prefix_actions)

    # ---- step (b): room-local solver + greedy best observe(anchor, rot) ----
    def _build_room_solver(self, anchor_name: str = 'initial_pos') -> None:
        rid = self._current_room()
        names = sorted(self.nodes_by_room.get(rid, set()))
        self.room_solver = SpatialSolver(names, grid_size=self.grid_size)
        self.room_solver.set_initial_position(anchor_name, (0, 0))

    def _ingest_recent_observations(self, start_idx: int) -> None:
        assert self.room_solver is not None
        for t in self.turns[start_idx:]:
            obs = t.actions[-1] if t.actions else None
            triples = obs.data.get('relation_triples', []) if hasattr(obs, 'data') else []
            if triples:
                # keep only triples within solver variables
                keep = {n for n in self.room_solver.solver.variables.keys()}
                filt = [tr for tr in triples if tr.subject in keep and tr.anchor in keep]
                if filt:
                    self.room_solver.add_observation(filt)
    def _metrics(self) -> int:
        assert self.room_solver is not None
        _, total_positions, _, total_rels = self.room_solver.compute_metrics(max_samples_per_var=self.eval_samples, bin_system=CardinalBinsAllo())
        return total_rels if self.metric != 'positions' else total_positions

    # 1) simulate returns best anchor + absolute orientation (not rotation)
    def _simulate_observe_gain(self, anchor: str, desired_ori: np.ndarray, baseline_metric: int) -> float:
        assert self.room_solver is not None
        if anchor not in self.room_solver.solver.variables: return 0.0
        sim = self.room_solver.copy()
        tmp = self.agent.copy()
        obj = self.room.get_object_by_name(anchor)
        tmp.pos, tmp.room_id, tmp.ori = obj.pos.copy(), obj.room_id, desired_ori.copy()
        res = ObserveAction().execute(self.room, tmp)
        triples = res.data.get('relation_triples', []) if hasattr(res, 'data') else []
        if not triples: return -1.0
        keep = set(sim.solver.variables.keys())
        filt = [tr for tr in triples if tr.subject in keep and tr.anchor in keep]
        if filt: sim.add_observation(filt)
        _, new_positions, _, new_rels = sim.compute_metrics(max_samples_per_var=self.eval_samples, bin_system=CardinalBinsAllo())
        new_metric = new_rels if self.metric != 'positions' else new_positions
        return float(baseline_metric - new_metric)

    # 2) pick best (anchor, absolute orientation)
    def _best_observe(self, baseline_metric: int) -> tuple[str | None, np.ndarray | None, float]:
        observed = set(self.mgr.observed_items or set())
        # anchor must be a node in current room and already observed
        candidate_anchors = observed & set(self._room_node_names())
        assert self.room_solver is not None
        domain_sizes, _, rel_sets, _ = self.room_solver.compute_metrics(max_samples_per_var=self.eval_samples, bin_system=CardinalBinsAllo())
        scores = []
        if self.metric == 'positions':
            for obj in candidate_anchors:
                scores.append((obj, int(domain_sizes.get(obj, 0))))
        else:
            for obj in candidate_anchors:
                best = 0
                for other in candidate_anchors:
                    if other == obj: continue
                    key = (obj, other) if (obj, other) in rel_sets else (other, obj)
                    best = max(best, len(rel_sets.get(key, set())))
                scores.append((obj, best))
        top_anchors = [o for o, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:10]]

        best_a, best_ori, best_gain = None, None, -1.0
        for a in top_anchors:
            # Try the 4 absolute cardinals at the anchor
            for ori in (np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0])):
                gain = self._simulate_observe_gain(a, ori, baseline_metric)
                if gain > best_gain:
                    best_a, best_ori, best_gain = a, ori, gain
        return best_a, best_ori, best_gain

    # 3) observe_at now rotates to a target absolute orientation AFTER move_to
    def _observe_at(self, anchor: str, desired_ori: np.ndarray) -> None:
        self._observe_at_anchor(anchor, desired_ori)
        self._ingest_last_into_solver(self.room_solver)

    # 4) loop uses (anchor, orientation) instead of (anchor, rotation)
    def _explore_room(self, is_initial: bool, prefix_actions: List = None) -> None:
        start_idx = len(self.turns)
        self._entry_observe_delegate(is_initial=is_initial, prefix_actions=prefix_actions)
        self._build_room_solver(anchor_name='initial_pos' if is_initial else (self.current_gate or 'initial_pos'))
        self._ingest_recent_observations(start_idx)
        while True:
            total = self._metrics()
            a, ori, gain = self._best_observe(total)
            if (gain is None) or (gain <= 1e-9) or (a is None) or (ori is None):
                break
            self._observe_at(a, ori)

    # No custom run; used as delegate in AnalystAgentProxy

    def _prune_dfs(self, rid: int):
        # Follow oracle DFS pruning when using oracle-style entry observation
        return OracleAgentProxy._prune_dfs(self, rid) if self.delegate == 'oracle' else False



class CandidatePlannerAgentProxy(ObserverAnalystAgentProxy):
    """Plan via candidate domains: move to the most ambiguous node, then observe facing the direction that covers most candidate pairs. Early stop when all candidates are singletons or all edges are covered."""

    def _all_singleton_room(self) -> bool:
        assert self.room_solver is not None
        domain_sizes, _, _, _ = self.room_solver.compute_metrics(max_samples_per_var=self.eval_samples, bin_system=CardinalBinsAllo())
        names = list(set(self._room_node_names()) & set(domain_sizes.keys()))
        if not names:
            return True
        return all(int(domain_sizes.get(n, 0)) == 1 for n in names)

    def _room_edges_done(self) -> bool:
        return not self._unknown_edges_in_room(self._current_room())

    def _edges_from_anchor_remaining(self, anchor: str) -> bool:
        return any(anchor in e for e in self._unknown_edges_in_room(self._current_room()))

    # uses base _room_node_names

    def _nodes_with_unknown_edges(self) -> Set[str]:
        unknown = self._unknown_edges_in_room(self._current_room())
        nodes = set(self._room_node_names())
        return {
            a for a in nodes
            if any((a in e) and (next(iter(e - {a})) in nodes) for e in unknown)
        }

    def _choose_anchor_with_most_candidates(self, moved: Set[str]) -> str | None:
        assert self.room_solver is not None
        names = sorted(self._nodes_with_unknown_edges() - set(moved))
        if not names:
            return None
        domain_sizes, _, _, _ = self.room_solver.compute_metrics(max_samples_per_var=self.eval_samples, bin_system=CardinalBinsAllo())
        return max(names, key=lambda n: domain_sizes.get(n, 0), default=None)

    def _sample(self, dom: Set[tuple]) -> List[tuple]:
        k = int(self.eval_samples)
        if len(dom) <= k:
            return list(dom)
        return random.sample(list(dom), k)

    def _best_direction_for_anchor(self, anchor: str, exclude_others: Set[str], exclude_dirs: Set[tuple] | None = None) -> tuple[np.ndarray | None, str | None]:
        assert self.room_solver is not None
        pos = self.room_solver.get_possible_positions()
        if anchor not in pos or not pos[anchor]:
            return None, None
        # consider only others with unknown edges to anchor
        rid = self._current_room()
        unknown = self._unknown_edges_in_room(rid)
        allowed_others = {o for o in self._room_node_names() if o != anchor and frozenset({anchor, o}) in unknown}
        dirs = [np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0])]
        allowed_dirs = [d for d in dirs if not (exclude_dirs and tuple(d.tolist()) in exclude_dirs)]
        if not allowed_dirs:
            return None, None
        totals: Dict[tuple, int] = {tuple(d.tolist()): 0 for d in allowed_dirs}
        per_dir_target: Dict[tuple, tuple[str | None, int]] = {tuple(d.tolist()): (None, -1) for d in allowed_dirs}

        anchor_pts = self._sample(pos[anchor])
        for other in allowed_others:
            if other in exclude_others:
                continue
            if other not in pos or not pos[other]:
                continue
            other_pts = self._sample(pos[other])
            dir_count: Dict[tuple, int] = {tuple(d.tolist()): 0 for d in allowed_dirs}
            for pa in anchor_pts:
                for pb in other_pts:
                    key = tuple(_closest_octant(np.array(pb) - np.array(pa)).tolist())
                    if key in dir_count:
                        dir_count[key] += 1
            for key, cnt in dir_count.items():
                totals[key] += cnt
                _, best = per_dir_target[key]
                if cnt > best:
                    per_dir_target[key] = (other, cnt)

        if max(totals.values(), default=0) <= 0:
            return None, None
        best_key, _ = max(totals.items(), key=lambda kv: kv[1])
        target_obj, _ = per_dir_target[best_key]
        return np.array(best_key), target_obj

    def _observe_facing(self, anchor: str, desired_ori: np.ndarray) -> None:
        self._observe_at_anchor(anchor, desired_ori)
        self._ingest_last_into_solver(self.room_solver)

    def _explore_room(self, is_initial: bool, prefix_actions: List = None) -> None:
        start_idx = len(self.turns)
        self._entry_observe_delegate(is_initial=is_initial, prefix_actions=prefix_actions)
        self._build_room_solver(anchor_name='initial_pos' if is_initial else (self.current_gate or 'initial_pos'))
        self._ingest_recent_observations(start_idx)
        if self._all_singleton_room() or self._room_edges_done():
            return
        budget = int(self.max_observes)
        moved: Set[str] = set()
        while budget > 0 and not self._all_singleton_room() and not self._room_edges_done():
            anchor = self._choose_anchor_with_most_candidates(moved)
            if anchor is None:
                break
            # Step 1: choose best ori BEFORE moving, then move+rotate+observe
            used_dirs: Set[tuple] = set()
            ori0, _ = self._best_direction_for_anchor(anchor, exclude_others=moved, exclude_dirs=used_dirs)
            if ori0 is None:
                moved.add(anchor)
                continue
            before = len(self.turns)
            self._observe_facing(anchor, ori0)
            used_dirs.add(tuple(ori0.tolist()))
            if len(self.turns) > before:
                budget -= 1
            if self._all_singleton_room() or self._room_edges_done():
                moved.add(anchor)
                break
            # Step 2: local loop at anchor
            seen: Set[str] = set(self._visible_others_from_last(anchor))
            while budget > 0 and self._edges_from_anchor_remaining(anchor) and not (self._all_singleton_room() or self._room_edges_done()):
                exclude = set(seen) | set(moved)
                ori, _ = self._best_direction_for_anchor(anchor, exclude, exclude_dirs=used_dirs)
                if ori is None:
                    break
                self._observe_facing(anchor, ori)
                used_dirs.add(tuple(ori.tolist()))
                budget -= 1
                if self._all_singleton_room() or self._room_edges_done():
                    break
                seen |= self._visible_others_from_last(anchor)
            moved.add(anchor)
        

    # uses base _move_to_visible

    def _visible_others_from_last(self, anchor: str) -> Set[str]:
        if not self.turns or not self.turns[-1].actions:
            return set()
        last = self.turns[-1].actions[-1]
        vis = set(last.data.get('visible_objects', []) if hasattr(last, 'data') else [])
        return {n for n in vis if n != anchor}

class AnalystAgentProxy(AgentProxy):
    """Analyst Agent: run observe-based planning delegates and terminate."""

    def __init__(self, room: Room, agent: Agent, grid_size: int | None = None,
                 max_queries: int = 16, rel_threshold: int = 0, eval_samples: int = 30,
                 delegate: str = 'oracle', observer_delegate: str = 'oracle', metric: str = 'positions'):
        super().__init__(room, agent, grid_size=grid_size)
        g = (max(self.room.mask.shape) if getattr(self.room, 'mask', None) is not None else 10)
        self.solver = SpatialSolver([o.name for o in self.room.all_objects] + ['initial_pos'], grid_size=(g if grid_size is None else grid_size))
        self.solver.set_initial_position('initial_pos', (0, 0))
        self.max_queries = int(max_queries)
        self.rel_threshold = int(rel_threshold)
        self.eval_samples = int(eval_samples)
        self.delegate = (delegate or 'strategist').lower()
        self.observer_delegate = (observer_delegate or 'strategist').lower()
        self.metric = (metric or 'positions').lower()

    def _ingest_observations(self) -> None:
        for i, t in enumerate(self.turns):
            obs = t.actions[-1]
            triples = obs.data.get('relation_triples', []) if hasattr(obs, 'data') else []
            if triples:
                self.solver.add_observation(triples)

    def _current_metrics(self) -> tuple[Dict[str, int], int, Dict[tuple, Set[str]], int]:
        """Current discrete metrics using cardinal bins."""
        return self.solver.compute_metrics(max_samples_per_var=self.eval_samples, bin_system=CardinalBinsAllo())

    def run(self) -> List[Turn]:
        """Run analyst delegate and end with Term()."""
        # Overview observe via selected delegate
        mapping = {
            'oracle': OracleAgentProxy,
            'strategist': StrategistAgentProxy,
            'inquisitor': InquisitorAgentProxy,
            'greedy_inquisitor': GreedyInquisitorAgentProxy,
            'greedy': GreedyInquisitorAgentProxy,
            'observer_analyst': ObserverAnalystAgentProxy,
            'candidate_planner': CandidatePlannerAgentProxy,
        }
        DelegateCls = mapping.get(self.delegate, OracleAgentProxy)
        if DelegateCls is ObserverAnalystAgentProxy:
            delegate = DelegateCls(self.room, self.agent, delegate=self.observer_delegate, metric=self.metric)
        elif DelegateCls is CandidatePlannerAgentProxy:
            delegate = DelegateCls(self.room, self.agent, delegate=self.observer_delegate, metric=self.metric)
        else:
            delegate = DelegateCls(self.room, self.agent)
        d_turns = delegate.run()
        self.mgr, self.room, self.agent = delegate.mgr, delegate.mgr.exploration_room, delegate.mgr.agent
        self.turns = list(d_turns[:-1]) if d_turns else [] # drop final Term()
        # adopt exploration knowledge/state from delegate
        self._adopt_state_from(delegate)

        # Ingest all observed relation triples into solver
        self._ingest_observations()

        # Terminate
        self._add_turn([self.mgr.execute_success_action(TermAction())])
        return self.turns


def get_agent_proxy(name: str, room: Room, agent: Agent, grid_size: int | None = None) -> AgentProxy:
    """Return one of the three supported proxies: 'scout', 'strategist', or 'oracle'.

    - scout -> OracleAgentProxy (greedy rotations to quickly reveal nodes)
    - strategist -> StrategistAgentProxy (sweep rotations)
    - oracle -> OracleAgentProxy
    """
    assert name in ['scout', 'strategist', 'oracle'], f"Invalid agent proxy name: {name}"
    if name == 'scout':
        return StrategistAgentProxy(room, agent, grid_size=grid_size)
    elif name == 'strategist':
        return AnalystAgentProxy(room, agent, delegate='candidate_planner', observer_delegate='strategist', grid_size=grid_size)
    elif name == 'oracle':
        return AnalystAgentProxy(room, agent, delegate='observer_analyst', observer_delegate='oracle', grid_size=grid_size)


def get_exploration_history_stats(data_dir: str, agent_type: str) -> Dict[str, Any]:
    """Iterate over runs in data_dir, run agent proxy, and return stats."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    runs = sorted([d for d in os.listdir(data_dir) if d.startswith("run") and os.path.isdir(os.path.join(data_dir, d))])
    
    total_cost = 0.0
    step_counts = []
    info_gain_lists = []
    action_counts_sum = {}
    n = 0

    for r in tqdm(runs, desc=f"Processing {agent_type}"):
        meta = os.path.join(data_dir, r, 'meta_data.json')
        if not os.path.isfile(meta): continue
            
        with open(meta, 'r') as f:
            data = json.load(f)
            
        room, agent = initialize_room_from_json(data)
        proxy = get_agent_proxy(agent_type, room, agent)
        turns = proxy.run()
        
        steps = len(turns)
        if steps > 0 and turns[-1].actions and turns[-1].actions[0].action_type == 'term':
            steps -= 1
        step_counts.append(steps)
        
        summary = proxy.mgr.get_exp_summary()
        total_cost += summary.get('action_cost', 0.0)
        info_gain_lists.append(summary.get('info_gain_list', []) or [])
        
        for k, v in (summary.get('action_counts', {}) or {}).items():
            action_counts_sum[k] = action_counts_sum.get(k, 0.0) + float(v)
        n += 1

    return {
        "num_envs": n,
        "avg_steps": float(np.mean(step_counts)) if n > 0 else 0.0,
        "std_steps": float(np.std(step_counts)) if n > 0 else 0.0,
        "avg_action_cost": (total_cost / n) if n > 0 else 0.0,
        "avg_info_gain_per_step": ExplorationManager._avg_lists_carry_forward(info_gain_lists) if n > 0 else [],
        "avg_action_counts": {k: v / n for k, v in sorted(action_counts_sum.items())} if n > 0 else {},
    }


if __name__ == "__main__":
    # Simple test for get_exploration_history_stats
    # Assume we run this from project root or relative to the file location
    # Adjust path to point to a valid room_data directory if needed
    argument = ArgumentParser()
    argument.add_argument("--data_dir", type=str, default="vagen/env/spatial/room_data")
    argument.add_argument("--agent_type", type=str, default="scout")
    args = argument.parse_args()
    root = args.data_dir
    if os.path.exists(root):
        print(f"Testing get_exploration_history_stats with '{args.agent_type}' on {root}...")
        try:
            stats = get_exploration_history_stats(root, args.agent_type)
            print(json.dumps(stats, indent=2))
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Test skipped: {root} not found. Please provide a valid data directory.")