"""
Cognitive Map Manager

Minimal, modular evaluator for cognitive maps.

Responsibilities:
- Extract JSON from LLM response
- Transform JSON sections (global/local/rooms/gates) into BaseRoom-compatible data
- Evaluate global, local, room maps (dir/facing/pos) using consistent coordinates
- Evaluate gates connectivity
- Log all results per turn for summary aggregation
"""

import json
import re
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
import copy
from ..actions.base import BaseAction
from ..core.room import Room, BaseRoom
from ..core.object import Object, Agent, Gate
# Utils
from ..utils.cogmap.transforms import (
    transform_baseroom,
)
from ..utils.cogmap.metrics import compute_map_metrics
from ..utils.cogmap.consistency import (
    local_vs_global_consistency,
    stability,
)
from ..utils.cogmap.types import BaseCogMetrics, MapCogMetrics, ConsistencySummary, UnexploredMetrics
from ..utils.cogmap.analysis import (
    get_last_exploration_cogmap,
    avg_nested_dicts,
    avg_float_list_skip_none,
)
from ..utils.cogmap.unexplored import (
    evaluate_unexplored_predictions,
    parse_fog_probe_response,
)
from ..utils.room_utils import RoomPlotter


@dataclass
class BaseCogMapTurnLog:
    """Common fields for all cogmap types."""
    type: str
    extraction_success: bool = False
    original_response: str = ""
    pred_json: Dict[str, Any] = field(default_factory=dict)
    pred_room_state: Optional['BaseRoom'] = None
    metrics: BaseCogMetrics = field(default_factory=BaseCogMetrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "extraction_success": self.extraction_success,
            "original_response": self.original_response,
            "pred_json": self.pred_json,
            "pred_room_state": self.pred_room_state.to_dict() if self.pred_room_state else {},
            "metrics": (self.metrics.to_dict() if self.metrics.valid else {}),
        }

@dataclass
class GlobalCogMapTurnLog(BaseCogMapTurnLog):
    gt_room_state: Optional['BaseRoom'] = None
    gt_json: Dict[str, Any] = field(default_factory=dict)
    gt_room_state_full: Optional['BaseRoom'] = None
    gt_json_full: Dict[str, Any] = field(default_factory=dict)
    metrics_full: BaseCogMetrics = field(default_factory=BaseCogMetrics)
    metric_agent: BaseCogMetrics = field(default_factory=BaseCogMetrics)

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "gt_room_state": self.gt_room_state.to_dict() if self.gt_room_state else {},
            "gt_json": self.gt_json,
            "gt_room_state_full": self.gt_room_state_full.to_dict() if self.gt_room_state_full else {},
            "gt_json_full": self.gt_json_full,
            "metrics_full": (self.metrics_full.to_dict() if self.metrics_full.valid else {}),
            "metric_agent": (self.metric_agent.to_dict() if self.metric_agent.valid else {}),
        })
        return out

@dataclass
class LocalCogMapTurnLog(BaseCogMapTurnLog):
    gt_room_state: Optional['BaseRoom'] = None
    gt_json: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "gt_room_state": self.gt_room_state.to_dict() if self.gt_room_state else {},
            "gt_json": self.gt_json,
        })
        return out

@dataclass
class FogProbeCogMapTurnLog(BaseCogMapTurnLog):
    """Turn log for fog probe predictions."""
    all_candidate_points: List[Tuple[int, int]] = field(default_factory=list)
    pred_points: List[Tuple[int, int]] = field(default_factory=list)
    correct_points: List[Tuple[int, int]] = field(default_factory=list)
    symbolic_map: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "all_candidate_points": [[int(x), int(y)] for x, y in (self.all_candidate_points or [])],
            "pred_points": [[int(x), int(y)] for x, y in (self.pred_points or [])],
            "correct_points": [[int(x), int(y)] for x, y in (self.correct_points or [])],
            "symbolic_map": self.symbolic_map,
        })
        return out


@dataclass
class CognitiveMapTurnLog:
    """Aggregate per-type logs for one turn."""
    global_log: Optional[GlobalCogMapTurnLog] = None
    local_log: Optional[LocalCogMapTurnLog] = None
    local_newly_log: Optional[LocalCogMapTurnLog] = None
    fog_probe_log: Optional[FogProbeCogMapTurnLog] = None
    consistency: Optional[ConsistencySummary] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.global_log:
            out["global"] = self.global_log.to_dict()
        if self.local_log:
            out["local"] = self.local_log.to_dict()
        if self.local_newly_log:
            out["local_newly"] = self.local_newly_log.to_dict()
        if self.fog_probe_log:
            out["fog_probe"] = self.fog_probe_log.to_dict()
        if self.consistency:
            out["consistency"] = self.consistency.to_dict()
        return out


class CognitiveMapManager:
    """Evaluate cognitive map JSON against ground truth."""    
    def __init__(self, cogmap_type: str = "standard", pos_allow_scale: bool = False, scope: str = "all"):
        """Initialize cognitive map manager."""
        self.config = {
            "cogmap_type": cogmap_type,
            "pos_allow_scale": bool(pos_allow_scale),
            "scope": (scope if scope in ("global", "all") else "all"),
        }
        # room_id -> first-entry gate name
        self.entry_gate_by_room: dict[int, str] = {}
        # position normalization scale (computed once in global frame)
        self._pos_norm_L: float | None = None
        self._start_room_id: int | None = None
        
        # State tracking for newly observed items
        self._last_observed_items: Set[str] = set()

    def _compute_retention(self, pred_br: BaseRoom, exp_pred_json: Dict, name: str, flags: Optional[Dict] = None) -> Dict[str, Optional[float]]:
        """Compute retention metric for a single object (new pred vs old pred).
        Object names may use spaces or underscores in JSON keys.
        If flags given, only compute pos/facing based on change type; otherwise compute both.
        """
        exp_info = exp_pred_json.get(name) or exp_pred_json.get(name.replace(" ", "_"))
        if not (exp_info and isinstance(exp_info, dict)):
            return {"dir": None, "pos": None, "facing": None, "overall": None}
        exp_br = self._parse_section_to_baseroom({name: exp_info}, "exp")
        exp_only = self._filter_br_by_names(exp_br, {name}) if exp_br else None
        if not (exp_only and exp_only.objects):
            return {"dir": None, "pos": None, "facing": None, "overall": None}
        m = self._compare_baserooms(pred_br, exp_only)
        if not m.valid:
            return {"dir": None, "pos": None, "facing": None, "overall": None}
        if flags:
            return {"dir": None, "pos": float(m.pos) if flags.get("pos") else None,
                    "facing": float(m.facing) if flags.get("ori") else None, "overall": None}
        return {"dir": None, "pos": float(m.pos), "facing": float(m.facing), "overall": None}

    def evaluate_false_belief_cogmap(
        self, assistant_response: str, fb_turn_log: Dict[str, Any],
        exploration_room_dict: Optional[Dict[str, Any]] = None,
        exploration_agent_dict: Optional[Dict[str, Any]] = None,
        last_exploration_cogmap: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate false-belief (global) cognitive map.

        The returned dict is intended to be stored at:
        `turn_log["false_belief_log"]["cogmap_log"]`.
        
        Args:
            exploration_room_dict/exploration_agent_dict: If provided, use these
                to compute pos_norm_L (from exploration stage, before FB changes).
            last_exploration_cogmap: The cogmap_log from the last exploration turn.
                Used to compute retention metric (new prediction vs old prediction).
        """
        if not isinstance(fb_turn_log, dict):
            return {}
        room_dict = fb_turn_log.get("room_state")
        agent_dict = fb_turn_log.get("agent_state")
        if not isinstance(room_dict, dict) or not isinstance(agent_dict, dict):
            return {"original_response": str(assistant_response or ""), "changed_objects_per_object": {}, "unchanged_objects": {}}

        gt_room = Room.from_dict(room_dict)
        gt_agent = Agent.from_dict(agent_dict)
        
        # Compute pos_norm_L from exploration room (before FB changes)
        if exploration_room_dict and exploration_agent_dict:
            exp_room = Room.from_dict(exploration_room_dict)
            exp_agent = Agent.from_dict(exploration_agent_dict)
            self._ensure_pos_norm_L(exp_room, exp_agent)

        fb_log = fb_turn_log.get("false_belief_log") or {}
        newly_observed_changed = fb_log.get("newly_observed_changed_objects") or []
        if not isinstance(newly_observed_changed, list):
            newly_observed_changed = []
        newly_observed_unchanged = fb_log.get("newly_observed_unchanged_objects") or []
        if not isinstance(newly_observed_unchanged, list):
            newly_observed_unchanged = []

        ground_truth_changes = fb_log.get("ground_truth_changes") or []
        changes_map: dict[str, dict[str, bool]] = {}
        all_changed_names: set[str] = set()
        for c in (ground_truth_changes or []):
            if isinstance(c, dict) and c.get("name"):
                name = str(c["name"]).replace("_", " ")
                all_changed_names.add(name)
                flags = changes_map.setdefault(name, {"pos": False, "ori": False})
                flags["pos"] = bool(flags["pos"] or c.get("pos"))
                flags["ori"] = bool(flags["ori"] or c.get("ori"))

        all_object_names = {o.name for o in gt_room.all_objects}
        unchanged_object_names = sorted(all_object_names - all_changed_names)

        responses_by_type = {"global": str(assistant_response or "")}
        changed_objects_metrics: Dict[str, Dict[str, float]] = {}
        retention_metrics: Dict[str, Dict[str, Optional[float]]] = {}
        # Store raw positions for inertia computation at sample level
        inertia_data_per_object: Dict[str, Dict[str, List[float]]] = {}

        # Extract last exploration pred_json for retention/inertia metric
        exp_pred_json = ((last_exploration_cogmap or {}).get("global") or {}).get("pred_json") or {}

        # Evaluate unchanged objects
        unchanged_log = (
            self.evaluate_cogmaps(responses_by_type, gt_room, gt_agent, unchanged_object_names)
            if unchanged_object_names else None
        )
        # Collect squared distances for newly_observed_unchanged (for sigma at sample level)
        unchanged_dists_sq: List[float] = []
        newly_unchanged_set = {str(n).replace("_", " ") for n in newly_observed_unchanged if isinstance(n, str)}
        if unchanged_log and unchanged_log.global_log and newly_unchanged_set:
            pred_br = unchanged_log.global_log.pred_room_state
            gt_br = unchanged_log.global_log.gt_room_state
            if pred_br and gt_br:
                for po in pred_br.objects:
                    if po.name not in newly_unchanged_set:
                        continue
                    go = next((o for o in gt_br.objects if o.name == po.name), None)
                    if go is not None:
                        unchanged_dists_sq.append(float(np.sum((np.array(po.pos) - np.array(go.pos)) ** 2)))

        # Evaluate each newly observed changed object
        for obj_name in newly_observed_changed:
            if not isinstance(obj_name, str) or not obj_name:
                continue
            name = obj_name.replace("_", " ")
            flags = changes_map.get(name) or {}
            single = self.evaluate_cogmaps(responses_by_type, gt_room, gt_agent, [name])
            if not (single and single.global_log and single.global_log.pred_room_state and single.global_log.gt_room_state):
                continue
            pred_only = self._filter_br_by_names(single.global_log.pred_room_state, {name})
            gt_only = self._filter_br_by_names(single.global_log.gt_room_state, {name})
            m = self._compare_baserooms(pred_only, gt_only)
            changed_objects_metrics[name] = {
                "dir": None,
                "pos": (float(m.pos) if flags.get("pos") else None),
                "facing": (float(m.facing) if flags.get("ori") else None),
                "overall": None,
            }
            retention_metrics[name] = self._compute_retention(pred_only, exp_pred_json, name, flags)
            # Store raw positions for inertia (position-changed objects only)
            if flags.get("pos") and pred_only and pred_only.objects and gt_only and gt_only.objects:
                exp_info = exp_pred_json.get(name) or exp_pred_json.get(name.replace(" ", "_"))
                new_obj = pred_only.objects[0]
                gt_obj = gt_only.objects[0]
                if exp_info and isinstance(exp_info, dict) and isinstance(exp_info.get("position"), list):
                    inertia_data_per_object[name] = {
                        "old_pos": [float(exp_info["position"][0]), float(exp_info["position"][1])],
                        "new_pos": list(new_obj.pos),
                        "gt_pos": list(gt_obj.pos),
                    }

        # Compute unchanged_retention for newly observed unchanged objects
        unchanged_retention_metrics: Dict[str, Dict[str, Optional[float]]] = {}
        if unchanged_log and unchanged_log.global_log and unchanged_log.global_log.pred_room_state:
            fb_pred_br = unchanged_log.global_log.pred_room_state
            for obj_name in newly_observed_unchanged:
                name = str(obj_name).replace("_", " ")
                pred_only = self._filter_br_by_names(fb_pred_br, {name})
                if not (pred_only and pred_only.objects):
                    continue
                unchanged_retention_metrics[name] = self._compute_retention(pred_only, exp_pred_json, name)

        return {
            "original_response": str(assistant_response or ""),
            "changed_objects_per_object": changed_objects_metrics,
            "retention_per_object": retention_metrics,
            "inertia_data_per_object": inertia_data_per_object,
            "unchanged_dists_sq": unchanged_dists_sq,
            "unchanged_retention_per_object": unchanged_retention_metrics,
            "unchanged_objects": (unchanged_log.to_dict() if unchanged_log else {}),
            "newly_observed_changed_objects": [str(x).replace("_", " ") for x in newly_observed_changed if isinstance(x, str)],
            "all_changed_object_names": sorted(all_changed_names),
            "unchanged_object_names": unchanged_object_names,
        }

    def evaluate_cogmap_type(self, assistant_response: str, gt_room: Room, gt_agent: Agent, observed_items: Optional[List[str]], map_type: str) -> Optional[BaseCogMapTurnLog]:
        """Extract JSON and evaluate a single cogmap type (global|local|rooms). Only compute what's needed for the given type."""
        self._register_active_entry_gate(gt_room)
        t = (map_type or "global").lower()
        json_dict = self._extract_json_from_text(assistant_response)
        if json_dict is None or gt_room is None:
            m = MapCogMetrics.invalid()
            return BaseCogMapTurnLog(type=t, extraction_success=False, original_response=assistant_response, metrics=m)
        all_item_names = {o.name for o in gt_room.all_objects}
        observed_set: set[str] = set(all_item_names if observed_items is None else [str(x).replace('_', ' ') for x in observed_items])
        visible_names = self._visible_object_names(gt_room, gt_agent)

        if t == "global":
            pred_global_br = self._preprocess_predicted(json_dict, observed_set, visible_names, gt_room, gt_agent, map_type)
            gt_global_br = self._build_gt_global_baseroom(gt_room, gt_agent, observed_set)
            full_global = transform_baseroom(self._baseroom_from_gt(gt_room, gt_agent), gt_agent.init_pos, gt_agent.init_ori)
            agent_br = self._build_gt_global_agent_baseroom(gt_room, gt_agent)
            self._ensure_pos_norm_L(gt_room, gt_agent)
            return self._eval_global(pred_global_br, gt_global_br, full_global, agent_br, assistant_response, json_dict)
        
        if t == "local":
            pred_local_br = self._preprocess_predicted(json_dict, observed_set, visible_names, gt_room, gt_agent, map_type)
            gt_local_br = self._build_gt_local_baseroom(gt_room, gt_agent)
            # If nothing is visible, skip this local turn (mark invalid so aggregations ignore it).
            if not gt_local_br.objects:
                return LocalCogMapTurnLog(
                    type="local",
                    extraction_success=True,
                    original_response=assistant_response,
                    pred_json=json_dict,
                    pred_room_state=pred_local_br,
                    metrics=MapCogMetrics.invalid(),
                    gt_room_state=gt_local_br,
                    gt_json={},
                )
            self._ensure_pos_norm_L(gt_room, gt_agent)
            return self._eval_local(pred_local_br, gt_local_br, assistant_response, json_dict)

        raise ValueError(f"Invalid map type: {t}")

    def evaluate_fog_probe(
        self,
        assistant_response: str,
        all_candidate_coords: Optional[List[Tuple[int, int]]],
        correct_coords: List[Tuple[int, int]],
        gt_room: Optional[Room] = None,
        gt_agent: Optional[Agent] = None,
    ) -> FogProbeCogMapTurnLog:
        """Evaluate fog-probe predictions where the LLM selects labeled candidates (A-Z).

        The prompt displays candidate points labeled 'A', 'B', ... in the same
        order as `all_candidate_coords`. The model should return something like
        `{"unexplored": ["A", "C"]}`. This method parses those labels,
        maps them to coordinates, and evaluates using the same unexplored
        prediction metrics.
        """
        assert correct_coords and all_candidate_coords, "No correct or candidate coordinates provided"

        # Use shared parser from utils.cogmap.unexplored
        labels, pred_coords = parse_fog_probe_response(assistant_response, all_candidate_coords)

        # Evaluate predictions using the same unexplored evaluator
        metrics = evaluate_unexplored_predictions(pred_coords, correct_coords)
        
        # Generate symbolic map for visualization (always useful to have text representation available)
        symbolic_map = None
        if gt_room and gt_agent and all_candidate_coords:
            symbolic_map = RoomPlotter.get_symbolic_map(gt_room, gt_agent, False, all_candidate_coords)

        return FogProbeCogMapTurnLog(
            type="fog_probe",
            extraction_success=True,
            original_response=assistant_response,
            pred_json={"parsed_from_text": True, "predicted_labels": labels, "predicted_coords": [[int(x), int(y)] for x, y in pred_coords]},
            all_candidate_points=[(int(x), int(y)) for x, y in (all_candidate_coords or [])],
            pred_points=pred_coords,
            correct_points=[(int(x), int(y)) for x, y in correct_coords],
            metrics=metrics,
            symbolic_map=symbolic_map,
        )

    def _eval_global(self, pred_global_br: BaseRoom,  gt_global_br: BaseRoom, gt_room_state_full: BaseRoom, agent_br: BaseRoom, assistant_response: str, pred_json: Dict) -> GlobalCogMapTurnLog:
        gt_json = self.baseroom_to_json(gt_global_br, include_gates=True)
        metrics = self._compare_baserooms(pred_global_br, gt_global_br)
        gt_json_full = self.baseroom_to_json(gt_room_state_full, include_gates=True)
        metrics_full = self._compare_baserooms(pred_global_br, gt_room_state_full)
        metric_agent = self._compare_baserooms(pred_global_br, agent_br)
        return GlobalCogMapTurnLog(
            type="global",
            extraction_success=True,
            original_response=assistant_response,
            pred_json=pred_json,
            pred_room_state=pred_global_br,
            metrics=metrics,
            gt_room_state=gt_global_br,
            gt_json=gt_json,
            gt_room_state_full=gt_room_state_full,
            gt_json_full=gt_json_full,
            metrics_full=metrics_full,
            metric_agent=metric_agent,
        )

    def _eval_local(self, pred_local_br: BaseRoom, gt_local_br: BaseRoom, assistant_response: str, pred_json: Dict) -> LocalCogMapTurnLog:
        metrics = self._compare_baserooms(pred_local_br, gt_local_br)
        return LocalCogMapTurnLog(
            type="local",
            extraction_success=True,
            original_response=assistant_response,
            pred_json=pred_json,
            pred_room_state=pred_local_br,
            metrics=metrics,
            gt_room_state=gt_local_br,
            gt_json=self.baseroom_to_json(gt_local_br, include_gates=True),
        )
    
    def evaluate_cogmaps(
        self,
        responses_by_type: Dict[str, str],
        gt_room: Room,
        gt_agent: Agent,
        observed_items: Optional[List[str]],
        all_correct_coords: Optional[List[Tuple[int, int]]] = None,
        all_candidate_coords: Optional[List[Tuple[int, int]]] = None,
    ) -> CognitiveMapTurnLog:
        """Evaluate multiple types and record one aggregate log for the turn.
        
        Args:
            responses_by_type: Dict mapping map_type to LLM response
            gt_room: Ground truth room
            gt_agent: Ground truth agent
            observed_items: List of observed item names
            all_correct_coords: List of correct unexplored (x, y) coordinates
        """
        # Calculate newly observed items
        current_observed = set([str(x).replace('_', ' ') for x in (observed_items or [])])
        newly_observed = current_observed - self._last_observed_items
        self._last_observed_items = current_observed

        out = CognitiveMapTurnLog()
        for map_type_key, resp in (responses_by_type or {}).items():
            if not isinstance(resp, str):
                continue
            
            if map_type_key not in ("global", "local", "fog_probe"):
                continue

            if map_type_key == "fog_probe":
                single = self.evaluate_fog_probe(resp, all_candidate_coords, all_correct_coords or [], gt_room=gt_room, gt_agent=gt_agent)
                setattr(out, f"{single.type}_log", single)
            else:
                single = self.evaluate_cogmap_type(resp, gt_room, gt_agent, observed_items, map_type_key)
                setattr(out, f"{single.type}_log", single)

                # If local map, compute newly observed metrics as a separate log entry
                if map_type_key == "local" and single and single.pred_room_state and single.gt_room_state:
                    # Filter both pred and gt to only include newly observed objects
                    pred_newly = self._filter_br_by_names(single.pred_room_state, newly_observed)
                    gt_newly = self._filter_br_by_names(single.gt_room_state, newly_observed)
                    
                    # Only compute if there are actually newly observed objects in GT
                    metrics_newly = MapCogMetrics.invalid()
                    if gt_newly.objects:
                        metrics_newly = self._compare_baserooms(pred_newly, gt_newly)
                    
                    # Create a new log for newly observed
                    out.local_newly_log = LocalCogMapTurnLog(
                        type="local_newly",
                        extraction_success=single.extraction_success,
                        original_response=single.original_response,
                        pred_json=single.pred_json,
                        pred_room_state=pred_newly,
                        metrics=metrics_newly,
                        gt_room_state=gt_newly,
                        gt_json=self.baseroom_to_json(gt_newly, include_gates=True)
                    )
        # Consistency fields per turn
        summary = ConsistencySummary()
        if (
            out.local_log and out.global_log
            and out.local_log.extraction_success and out.global_log.extraction_success
            and out.local_log.metrics.valid and out.global_log.metrics.valid
        ):
            cm = local_vs_global_consistency(
                out.local_log.pred_room_state,
                out.global_log.pred_room_state,
                gt_agent,
                allow_scale=bool(self.config.get('pos_allow_scale', False)),
                pos_norm_L=self._pos_norm_L,
            )
            summary.local_vs_global = cm
        
        out.consistency = summary
        return out
            

    @staticmethod
    def aggregate_group_performance(env_data_list: List[Dict], exp_type: str = None) -> Dict[str, Any]:
        """Aggregate cognitive map metrics per scenario.

        exp_type in {
            'active': error + consistency + correctness,
            'passive': correctness (global only),
        }
        Prefer precomputed per-sample metrics when available.
        """
        assert isinstance(env_data_list, list) and len(env_data_list) > 0, "env_data_list must be a non-empty list"

        pre_list = [cogmap for s in env_data_list if (cogmap := (s.get('metrics') or {}).get('cogmap')) is not None]
        
        # Helper to count samples with valid data at specific path
        def count_valid(dicts, path):
            c = 0
            for d in dicts:
                curr = d
                valid = True
                for k in path:
                    if isinstance(curr, dict) and curr.get(k) is not None:
                        curr = curr[k]
                    else:
                        valid = False
                        break
                if valid:
                    c += 1
            return c

        # Calculate counts
        n_global = count_valid(pre_list, ['exploration', 'error', 'global_vs_gt_global_avg'])
        n_local = count_valid(pre_list, ['exploration', 'error', 'local_vs_gt_local_avg'])
        n_newly = count_valid(pre_list, ['exploration', 'error', 'newly_observed_vs_gt_local_avg'])
        n_fog = count_valid(pre_list, ['exploration', 'fog_probe', 'f1_avg'])

        if exp_type == 'active':
            exploration = avg_nested_dicts([m.get('exploration') or {} for m in pre_list])
            evaluation = avg_nested_dicts([m.get('evaluation') or {} for m in pre_list])

            per_turn_list = [(m.get('per_turn_metrics') or {}) for m in pre_list if isinstance(m, dict)]
            update_turn = avg_nested_dicts([{'cogmap_update_per_turn': d.get('cogmap_update_per_turn') or {}} for d in per_turn_list]).get('cogmap_update_per_turn', {})
            full_turn = avg_nested_dicts([{'cogmap_full_per_turn': d.get('cogmap_full_per_turn') or {}} for d in per_turn_list]).get('cogmap_full_per_turn', {})
            self_tracking_turn = avg_nested_dicts([{'self_tracking_per_turn': d.get('self_tracking_per_turn') or {}} for d in per_turn_list]).get('self_tracking_per_turn', {})
            fb_unchanged_turn = avg_nested_dicts([{'cogmap_fb_unchanged_per_turn': d.get('cogmap_fb_unchanged_per_turn') or {}} for d in per_turn_list]).get('cogmap_fb_unchanged_per_turn', {})
            
            fog_probe_f1_turn = avg_nested_dicts([{'fog_probe_f1_per_turn': d.get('fog_probe_f1_per_turn') or []} for d in per_turn_list]).get('fog_probe_f1_per_turn', [])
            fog_probe_p_turn = avg_nested_dicts([{'fog_probe_p_per_turn': d.get('fog_probe_p_per_turn') or []} for d in per_turn_list]).get('fog_probe_p_per_turn', [])
            fog_probe_r_turn = avg_nested_dicts([{'fog_probe_r_per_turn': d.get('fog_probe_r_per_turn') or []} for d in per_turn_list]).get('fog_probe_r_per_turn', [])
            
            position_update_turn = avg_nested_dicts([{'position_update_per_turn': d.get('position_update_per_turn') or []} for d in per_turn_list]).get('position_update_per_turn', [])
            facing_update_turn = avg_nested_dicts([{'facing_update_per_turn': d.get('facing_update_per_turn') or []} for d in per_turn_list]).get('facing_update_per_turn', [])
            position_stability_turn = avg_nested_dicts([{'position_stability_per_turn': (d.get('position_stability_per_turn') or d.get('stability_per_turn') or [])} for d in per_turn_list]).get('position_stability_per_turn', [])
            facing_stability_turn = avg_nested_dicts([{'facing_stability_per_turn': d.get('facing_stability_per_turn') or []} for d in per_turn_list]).get('facing_stability_per_turn', [])

            per_turn_metrics = {
                'cogmap_update_per_turn': update_turn,
                'cogmap_full_per_turn': full_turn,
                "self_tracking_per_turn": self_tracking_turn,
                "fog_probe_f1_per_turn": fog_probe_f1_turn,
                "fog_probe_p_per_turn": fog_probe_p_turn,
                "fog_probe_r_per_turn": fog_probe_r_turn,
                "position_update_per_turn": position_update_turn,
                "facing_update_per_turn": facing_update_turn,
                "position_stability_per_turn": position_stability_turn,
                "facing_stability_per_turn": facing_stability_turn,
                "cogmap_fb_unchanged_per_turn": fb_unchanged_turn,
            }
            
            # Aggregate cogmap_fb metrics if available
            cogmap_fb_list = [m.get('cogmap_fb') or {} for m in pre_list if isinstance(m, dict) and m.get('cogmap_fb')]
            cogmap_fb_aggregated = {}
            if cogmap_fb_list:
                fb_metrics_list = [fb.get('metrics') or {} for fb in cogmap_fb_list]
                fb_avg_summary = avg_nested_dicts(fb_metrics_list)
                # Compute summary inertia from all samples' inertia_lists
                all_inertia = []
                for fb in cogmap_fb_list:
                    il = (fb.get('metrics') or {}).get('inertia_list') or []
                    all_inertia.extend([float(v) for v in il if isinstance(v, (int, float))])
                summary_inertia = (sum(all_inertia) / len(all_inertia)) if all_inertia else None
                fb_avg_summary['inertia'] = summary_inertia
                fb_avg_summary.pop('inertia_list', None)
                cogmap_fb_aggregated = {'metrics': fb_avg_summary}
            
            # Separate fog_probe
            fog_probe = exploration.pop('fog_probe', {}) if exploration else {}
            
            # Add counts
            if exploration:
                exploration['n_samples'] = f"Global: {n_global}, Local: {n_local}, Newly: {n_newly}"
            if fog_probe:
                fog_probe['n_samples'] = n_fog

            result = {
                "exploration": exploration,
                "fog_probe": fog_probe,
                "evaluation": evaluation if evaluation else {"correctness": {}},
                "per_turn_metrics": per_turn_metrics,
            }
            
            # Add cogmap_fb if available
            if cogmap_fb_aggregated:
                result["cogmap_fb"] = cogmap_fb_aggregated
            
            return result
        if exp_type == "passive":
            exploration = avg_nested_dicts([m.get('exploration') or {} for m in pre_list])
            n_global_full = count_valid(pre_list, ['exploration', 'correctness', 'global_full'])
            res = {'exploration': {'correctness': {'global_full': (exploration.get('correctness') or {}).get('global_full', {})}}}
            res['exploration']['n_samples'] = n_global_full
            return res

        # Default: average nested
        res = avg_nested_dicts(pre_list)
        if isinstance(res, dict) and 'exploration' in res:
             fog = res['exploration'].pop('fog_probe', None)
             if fog:
                 res['fog_probe'] = fog
                 res['fog_probe']['n_samples'] = n_fog
             res['exploration']['n_samples'] = f"Global: {n_global}, Local: {n_local}, Newly: {n_newly}"
        return res

    @staticmethod
    def compute_last_exploration_unchanged_metrics(
        last_cogmap_log: Optional[Dict[str, Any]],
        fb_turn_logs: List[Dict],
        gt_room_dict: Optional[Dict[str, Any]] = None,
        gt_agent_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        if not last_cogmap_log or not fb_turn_logs or not gt_room_dict or not gt_agent_dict:
            return {}
        # Use exploration gt_room/gt_agent to calculate pos_norm_L (same as exploration evaluation)
        gt_room = Room.from_dict(gt_room_dict)
        gt_agent = Agent.from_dict(gt_agent_dict)
        tmp_mgr = CognitiveMapManager(scope="global")
        tmp_mgr._ensure_pos_norm_L(gt_room, gt_agent)
        g = (last_cogmap_log.get('global') or {}) if isinstance(last_cogmap_log, dict) else {}
        pred = BaseRoom.from_dict(g.get('pred_room_state') or {})
        gt_full = BaseRoom.from_dict((g.get('gt_room_state_full') or {}) if isinstance(g, dict) else {})
        gt = BaseRoom.from_dict((g.get('gt_room_state') or {}) if isinstance(g, dict) else {})
        if not pred.objects or not gt_full.objects:
            return {}
        changed = set()
        for fb_turn in (fb_turn_logs or []):
            fb_log = fb_turn.get('false_belief_log') or {}
            for c in (fb_log.get('ground_truth_changes') or []):
                if isinstance(c, dict) and c.get('name'):
                    changed.add(str(c['name']).replace('_', ' '))
        base_names = {o.name for o in gt.objects if o.name != "agent"} or {o.name for o in gt_full.objects if o.name != "agent"}
        unchanged = base_names - changed
        if not unchanged:
            return {}
        pred_only = BaseRoom(objects=[o for o in pred.objects if o.name in unchanged], name=pred.name)
        gt_only = BaseRoom(objects=[o for o in gt_full.objects if o.name in unchanged], name=gt_full.name)
        m = compute_map_metrics(pred_only, gt_only, allow_scale=False, pos_norm_L=tmp_mgr._pos_norm_L)
        return m.to_dict() if m.valid else {}

    @staticmethod
    def compute_fb_unchanged_retention(fb_turn_logs: List[Dict]) -> Dict[str, Optional[float]]:
        """Aggregate unchanged_retention from all FB turns (already first-observation filtered)."""
        if not fb_turn_logs:
            return {}
        pos_vals, facing_vals = [], []
        for fb_turn in fb_turn_logs:
            cm_log = ((fb_turn.get('false_belief_log') or {}).get('cogmap_log') or {})
            for m in (cm_log.get('unchanged_retention_per_object') or {}).values():
                if not isinstance(m, dict):
                    continue
                if isinstance(m.get('pos'), (int, float)):
                    pos_vals.append(float(m['pos']))
                if isinstance(m.get('facing'), (int, float)):
                    facing_vals.append(float(m['facing']))
        if not pos_vals and not facing_vals:
            return {}
        avg = lambda v: sum(v) / len(v) if v else None
        return {'dir': None, 'pos': avg(pos_vals), 'facing': avg(facing_vals), 'overall': None}

    @staticmethod
    def compute_false_belief_metrics(
        fb_turn_logs: List[Dict],
        last_exploration_unchanged: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """Compute aggregated false belief metrics from turn logs."""
        if not fb_turn_logs:
            return {}

        def _avg(values: List[float]) -> Optional[float]:
            v = [float(x) for x in (values or []) if isinstance(x, (int, float))]
            return (sum(v) / len(v)) if v else None

        # Changed: average ONLY the relevant metric per change type (pos or facing). No per-turn.
        pos_vals: List[float] = []
        facing_vals: List[float] = []
        # retention: compare against last exploration cogmap prediction
        ret_pos_vals: List[float] = []
        ret_facing_vals: List[float] = []

        for fb_turn in (fb_turn_logs or []):
            fb_log = fb_turn.get('false_belief_log') or {}
            cm_log = fb_log.get('cogmap_log') or {}
            per_obj = cm_log.get('changed_objects_per_object') or {}
            ret_per_obj = cm_log.get('retention_per_object') or {}

            # name -> {pos, ori}
            changes_map: dict[str, dict[str, bool]] = {}
            for c in (fb_log.get('ground_truth_changes') or []):
                if not isinstance(c, dict) or not c.get('name'):
                    continue
                name = str(c['name']).replace('_', ' ')
                flags = changes_map.setdefault(name, {'pos': False, 'ori': False})
                flags['pos'] = bool(flags['pos'] or c.get('pos'))
                flags['ori'] = bool(flags['ori'] or c.get('ori'))

            if not isinstance(per_obj, dict):
                continue
            for obj_name, m in per_obj.items():
                if not isinstance(m, dict):
                    continue
                name = str(obj_name).replace('_', ' ')
                flags = changes_map.get(name) or {}
                if flags.get('pos'):
                    v = m.get('pos')
                    if isinstance(v, (int, float)):
                        pos_vals.append(float(v))
                if flags.get('ori'):
                    v = m.get('facing')
                    if isinstance(v, (int, float)):
                        facing_vals.append(float(v))

            # Aggregate retention metrics
            for obj_name, m_ret in (ret_per_obj or {}).items():
                if not isinstance(m_ret, dict):
                    continue
                name = str(obj_name).replace('_', ' ')
                flags = changes_map.get(name) or {}
                if flags.get('pos'):
                    v = m_ret.get('pos')
                    if isinstance(v, (int, float)):
                        ret_pos_vals.append(float(v))
                if flags.get('ori'):
                    v = m_ret.get('facing')
                    if isinstance(v, (int, float)):
                        ret_facing_vals.append(float(v))

        # Compute sigma from all unchanged_dists_sq across all turns, then compute inertia
        all_unchanged_dists_sq: List[float] = []
        all_inertia_data: List[Dict] = []
        for fb_turn in (fb_turn_logs or []):
            cm_log = ((fb_turn.get('false_belief_log') or {}).get('cogmap_log') or {})
            all_unchanged_dists_sq.extend(cm_log.get('unchanged_dists_sq') or [])
            for name, data in (cm_log.get('inertia_data_per_object') or {}).items():
                if isinstance(data, dict):
                    all_inertia_data.append(data)
        sigma = float(np.sqrt(sum(all_unchanged_dists_sq) / len(all_unchanged_dists_sq))) if all_unchanged_dists_sq else 1.0
        # Compute inertia for each object using sample-level sigma
        inertia_list: List[float] = []
        for data in all_inertia_data:
            old_pos = np.array(data.get('old_pos', [0, 0]))
            new_pos = np.array(data.get('new_pos', [0, 0]))
            gt_pos = np.array(data.get('gt_pos', [0, 0]))
            v = old_pos - gt_pos
            e = new_pos - gt_pos
            v_norm, e_norm = float(np.linalg.norm(v)), float(np.linalg.norm(e))
            if v_norm < 1e-9 or e_norm < 1e-9:
                continue
            cos_theta = float(np.dot(e, v)) / (e_norm * v_norm + 1e-9)
            dist_sq = float(np.sum((new_pos - old_pos) ** 2))
            w = float(np.exp(-dist_sq / (2 * sigma ** 2 + 1e-9))) if sigma > 1e-9 else 1.0
            inertia_list.append(cos_theta * w)
        inertia = _avg(inertia_list)
        
        changed_avg = {'dir': None, 'pos': _avg(pos_vals), 'facing': _avg(facing_vals), 'overall': None}
        retention_avg = {'dir': None, 'pos': _avg(ret_pos_vals), 'facing': _avg(ret_facing_vals), 'overall': None}

        # Unchanged: averaged across turns
        unchanged_metrics: List[MapCogMetrics] = []
        for fb_turn in (fb_turn_logs or []):
            cm_log = ((fb_turn.get('false_belief_log') or {}).get('cogmap_log') or {})
            g_log = ((cm_log.get('unchanged_objects') or {}).get('global') or {})
            m = MapCogMetrics.from_dict((g_log.get('metrics') or {}) if isinstance(g_log, dict) else {})
            if m.valid:
                unchanged_metrics.append(m)
        unchanged_avg = (MapCogMetrics.average(unchanged_metrics).to_dict() if unchanged_metrics else {})

        # Compute unchanged_retention from per-turn per-object data
        unchanged_retention_avg = CognitiveMapManager.compute_fb_unchanged_retention(fb_turn_logs)

        metrics = {
            'changed': changed_avg,
            'retention': retention_avg,
            'unchanged': unchanged_avg,
            'inertia': inertia,
            'inertia_list': inertia_list,  # stored for summary-level aggregation
        }
        if unchanged_retention_avg:
            metrics['unchanged_retention'] = unchanged_retention_avg
            # Compute unchanged_retention - retention (pos and facing separately)
            diff = {}
            for k in ('pos', 'facing'):
                ur_v = unchanged_retention_avg.get(k)
                r_v = retention_avg.get(k)
                if isinstance(ur_v, (int, float)) and isinstance(r_v, (int, float)):
                    if float(ur_v) != 0:
                        # diff[k] = (float(ur_v) - float(r_v)) / float(ur_v)
                        diff[k] = 1 - (float(r_v) / float(ur_v)) ** 2
            if diff:
                metrics['unchanged_retention_minus_retention'] = diff
        if isinstance(last_exploration_unchanged, dict) and last_exploration_unchanged:
            metrics['unchanged_exploration'] = last_exploration_unchanged
        return {'metrics': metrics}

    @staticmethod
    def compute_false_belief_unchanged_per_turn(
        fb_turn_logs: List[Dict],
        initial_metrics: Optional[Dict[str, Optional[float]]] = None,
    ) -> Dict[str, List[Optional[float]]]:
        """Per-turn series for unchanged objects during false-belief phase."""
        out = {'dir': [], 'facing': [], 'pos': [], 'overall': []}
        if isinstance(initial_metrics, dict) and initial_metrics:
            out['dir'].append(float(initial_metrics.get('dir')) if isinstance(initial_metrics.get('dir'), (int, float)) else None)
            out['facing'].append(float(initial_metrics.get('facing')) if isinstance(initial_metrics.get('facing'), (int, float)) else None)
            out['pos'].append(float(initial_metrics.get('pos')) if isinstance(initial_metrics.get('pos'), (int, float)) else None)
            out['overall'].append(float(initial_metrics.get('overall')) if isinstance(initial_metrics.get('overall'), (int, float)) else None)
        for t in (fb_turn_logs or []):
            cm_log = (((t or {}).get('false_belief_log') or {}).get('cogmap_log') or {})
            g = ((cm_log.get('unchanged_objects') or {}).get('global') or {})
            m = MapCogMetrics.from_dict((g.get('metrics') or {}) if isinstance(g, dict) else {})
            out['dir'].append(float(m.dir) if m.valid else None)
            out['facing'].append(float(m.facing) if m.valid else None)
            out['pos'].append(float(m.pos) if m.valid else None)
            out['overall'].append(float(m.overall) if m.valid else None)
        return out

    @staticmethod
    def aggregate_per_sample(env_data: Dict[str, Any], exp_type: str | None = None) -> Dict[str, Any]:
        """Aggregate cognitive-map metrics within a single sample (over turns).
        Returns exploration error/correctness/consistency and per-turn global metrics.
        """
        fb_turn_logs = env_data.get('false_belief_turn_logs') or []

        # Helper: get exploration turns' cogmap logs
        turn_logs = env_data.get('env_turn_logs') or []
        cog_logs = []
        exp_logs = []
        for t in turn_logs:
            if t.get('is_exploration_phase', False) and t.get('cogmap_log'):
                cog_logs.append(t['cogmap_log'])
                exp_logs.append(t.get('exploration_log') or {})
        last = get_last_exploration_cogmap(env_data)
        # Find last exploration turn for gt_room/gt_agent (before FB modifications)
        last_turn = next((t for t in reversed(turn_logs) if t.get('is_exploration_phase') and t.get('cogmap_log')), None)
        fb_last_unchanged = (
            CognitiveMapManager.compute_last_exploration_unchanged_metrics(
                last, fb_turn_logs,
                (last_turn or {}).get('room_state'),
                (last_turn or {}).get('agent_state'),
            ) if fb_turn_logs else {}
        )
        fb_unchanged_turn = (
            CognitiveMapManager.compute_false_belief_unchanged_per_turn(fb_turn_logs, fb_last_unchanged)
            if fb_turn_logs else None
        )
        if not cog_logs:
            # Keep placeholder exploration metrics, but still include false-belief cogmap if present.
            res = {
                'exploration': {
                    'error': {},
                    'correctness': {},
                    'consistency': {},
                    'fog_probe': {'f1_avg': None, 'precision_avg': None, 'recall_avg': None},
                },
                'per_turn_metrics': {},
            }
            if fb_unchanged_turn:
                res['per_turn_metrics']['cogmap_fb_unchanged_per_turn'] = fb_unchanged_turn
            cogmap_fb_metrics = (
                CognitiveMapManager.compute_false_belief_metrics(fb_turn_logs, fb_last_unchanged)
                if fb_turn_logs else {}
            )
            if cogmap_fb_metrics:
                res['cogmap_fb'] = cogmap_fb_metrics
            return res
        # Use shared helper to find last exploration cogmap
        
        
        # Average metrics over turns
        def _avg_maps(dicts: List[Dict[str, Any]], path: List[str]) -> MapCogMetrics:
            mats: List[MapCogMetrics] = []
            for d in dicts:
                cur = d
                ok = True
                for key in path:
                    if isinstance(cur, dict) and key in cur:
                        cur = cur[key]
                    else:
                        ok = False
                        break
                if ok and isinstance(cur, dict):
                    m = MapCogMetrics.from_dict(cur)
                    if m.valid:
                        mats.append(m)
            return MapCogMetrics.average(mats) if mats else MapCogMetrics.invalid()

        def _d(m: MapCogMetrics) -> Dict[str, float]:
            return m.to_dict() if m.valid else {}

        error = {
            'local_vs_gt_local_avg': _d(_avg_maps(cog_logs, ['local', 'metrics'])),
            'global_vs_gt_global_avg': _d(_avg_maps(cog_logs, ['global', 'metrics'])),
            'agent_vs_gt_agent_avg': _d(_avg_maps(cog_logs, ['global', 'metric_agent'])),
            'newly_observed_vs_gt_local_avg': _d(_avg_maps(cog_logs, ['local_newly', 'metrics'])),
        }

        # Correctness: last global_full
        correctness = {
            'last_global_vs_gt_full': (lambda _m: (_m.to_dict() if _m.valid else {}))(MapCogMetrics.from_dict((((last or {}).get('global') or {}).get('metrics_full') or {}))),
        }

        # Consistency
        # local_vs_global average over turns
        def _avg_consistency_lvsg(dicts: List[Dict[str, Any]]) -> MapCogMetrics:
            mats: List[MapCogMetrics] = []
            for d in dicts:
                cm = (d.get('consistency') or {}).get('local_vs_global') or {}
                m = MapCogMetrics.from_dict(cm)
                if m.valid:
                    mats.append(m)
            return MapCogMetrics.average(mats) if mats else MapCogMetrics.invalid()

        # Compute stability metrics (now returns dictionary of lists)
        stab_res = stability(env_data, threshold=1)
        
        pos_stab = stab_res.get('position_stability')

        consistency = {
            'local_vs_global_avg': _d(_avg_consistency_lvsg(cog_logs)),
            'position_update_avg': avg_float_list_skip_none(stab_res['position_update']),
            'facing_update_avg': avg_float_list_skip_none(stab_res['facing_update']),
            'position_stability_avg': avg_float_list_skip_none(pos_stab),
            'facing_stability_avg': avg_float_list_skip_none(stab_res['facing_stability']),
        }

        # Per-turn global metrics (list)
        per_turn_update, per_turn_full, per_turn_self_tracking = CognitiveMapManager.compute_per_turn_global_metrics(cog_logs)
        
        # Fog Probe: keep per_turn F1 for plotting
        fog_probe_f1_per_turn = []
        fog_probe_p_per_turn = []
        fog_probe_r_per_turn = []
        
        fog_probe_f1_vals = []
        fog_probe_p_vals = []
        fog_probe_r_vals = []

        for cl in cog_logs:
            fp_log = cl.get('fog_probe') or {}
            metrics = UnexploredMetrics.from_dict(fp_log.get('metrics') or {})
            
            f1 = float(metrics.overall) if metrics.valid else None
            p = float(metrics.precision) if metrics.valid else None
            r = float(metrics.recall) if metrics.valid else None
            
            fog_probe_f1_per_turn.append(f1)
            fog_probe_p_per_turn.append(p)
            fog_probe_r_per_turn.append(r)
            
            if f1 is not None: fog_probe_f1_vals.append(f1)
            if p is not None: fog_probe_p_vals.append(p)
            if r is not None: fog_probe_r_vals.append(r)

        fog_probe_f1_avg = avg_float_list_skip_none(fog_probe_f1_vals)
        fog_probe_p_avg = avg_float_list_skip_none(fog_probe_p_vals)
        fog_probe_r_avg = avg_float_list_skip_none(fog_probe_r_vals)

        if exp_type == 'passive':
            return {
                'exploration': {
                    'correctness': {
                        'global_full': correctness['last_global_vs_gt_full']
                    }
                }
            }

        per_turn_metrics = {
            'cogmap_update_per_turn': per_turn_update,
            'cogmap_full_per_turn': per_turn_full,
            'self_tracking_per_turn': per_turn_self_tracking,
            'fog_probe_f1_per_turn': fog_probe_f1_per_turn,
            'fog_probe_p_per_turn': fog_probe_p_per_turn,
            'fog_probe_r_per_turn': fog_probe_r_per_turn,
            'position_update_per_turn': [None] + stab_res['position_update'],
            'facing_update_per_turn': [None] + stab_res['facing_update'],
            'position_stability_per_turn': [None] + (pos_stab or []),
            'facing_stability_per_turn': [None] + stab_res['facing_stability'],
        }

        # Process false belief cogmap data if available
        if fb_unchanged_turn:
            per_turn_metrics['cogmap_fb_unchanged_per_turn'] = fb_unchanged_turn
        cogmap_fb_metrics = CognitiveMapManager.compute_false_belief_metrics(fb_turn_logs, fb_last_unchanged) if fb_turn_logs else {}

        result = {
            'exploration': {
                'error': error,
                'correctness': correctness,
                'consistency': consistency,
                'fog_probe': {
                    'f1_avg': fog_probe_f1_avg,
                    'precision_avg': fog_probe_p_avg,
                    'recall_avg': fog_probe_r_avg,
                },
            },
            'per_turn_metrics': per_turn_metrics,
        }
        
        # Add cogmap_fb metrics if available
        if cogmap_fb_metrics:
            result['cogmap_fb'] = cogmap_fb_metrics
        
        return result

    @staticmethod
    def compute_per_turn_global_metrics(cog_logs: List[Dict[str, Any]]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
        """Return (update, full, self_tracking) per-turn global metric lists."""
        per_turn_update = {'dir': [], 'facing': [], 'pos': [], 'overall': []}
        per_turn_full = {'dir': [], 'facing': [], 'pos': [], 'overall': []}
        per_turn_self_tracking = {'dir': [], 'facing': [], 'pos': [], 'overall': []}
        for d in cog_logs:
            g = d.get('global') or {}
            mu = MapCogMetrics.from_dict(g.get('metrics') or {})
            mf = MapCogMetrics.from_dict(g.get('metrics_full') or {})
            ma = MapCogMetrics.from_dict(g.get('metric_agent') or {})
            per_turn_update['dir'].append(float(mu.dir) if mu.valid else None)
            per_turn_update['facing'].append(float(mu.facing) if mu.valid else None)
            per_turn_update['pos'].append(float(mu.pos) if mu.valid else None)
            per_turn_update['overall'].append(float(mu.overall) if mu.valid else None)
            per_turn_full['dir'].append(float(mf.dir) if mf.valid else None)
            per_turn_full['facing'].append(float(mf.facing) if mf.valid else None)
            per_turn_full['pos'].append(float(mf.pos) if mf.valid else None)
            per_turn_full['overall'].append(float(mf.overall) if mf.valid else None)
            per_turn_self_tracking['dir'].append(float(ma.dir) if ma.valid else None)
            per_turn_self_tracking['facing'].append(float(ma.facing) if ma.valid else None)
            per_turn_self_tracking['pos'].append(float(ma.pos) if ma.valid else None)
            per_turn_self_tracking['overall'].append(float(ma.overall) if ma.valid else None)
        return per_turn_update, per_turn_full, per_turn_self_tracking
    
    # register entry gates for active exploratoin
    def _register_active_entry_gate(self, gt_room) -> None:
        """
        Register entry gates for rooms based on room structure.
        For simplicity, assign the first gate connecting to room 1 as the entry gate for each room.
        Room 1 is considered the starting room and doesn't get an entry gate.
        """
        if not hasattr(gt_room, 'gates') or not gt_room.gates:
            return
            
        # Set room 1 as the starting room
        if self._start_room_id is None:
            self._start_room_id = 1

        # For each room (except room 1), find its connection to room 1 or already processed rooms
        processed_rooms = {self._start_room_id}
        
        # Keep processing until no more rooms can be processed
        changed = True
        while changed:
            changed = False
            for g in gt_room.gates:
                if len(g.room_id) == 2:
                    room_a, room_b = int(g.room_id[0]), int(g.room_id[1])
                    
                    # If one room is processed and the other isn't, register the gate for the unprocessed room
                    if room_a in processed_rooms and room_b not in processed_rooms:
                        if room_b not in self.entry_gate_by_room:
                            self.entry_gate_by_room[room_b] = g.name
                            processed_rooms.add(room_b)
                            changed = True
                    elif room_b in processed_rooms and room_a not in processed_rooms:
                        if room_a not in self.entry_gate_by_room:
                            self.entry_gate_by_room[room_a] = g.name
                            processed_rooms.add(room_a)
                            changed = True

    # =============================== Parsing helpers =============================== 
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON content from text."""
        # Try fenced blocks first
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        candidates = fenced if fenced else []

        # Fallback: scan for outermost balanced braces
        stack, start = [], None
        for i, ch in enumerate(text):
            if ch == '{':
                if not stack:
                    start = i
                stack.append(ch)
            elif ch == '}' and stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(text[start:i+1])
                    start = None

        # Try to load the largest candidate
        candidates.sort(key=len, reverse=True)
        for cand in candidates:
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                continue
        return None
    
    def _parse_section_to_baseroom(self, mapping: Dict[str, Any], room_name: str) -> Optional[BaseRoom]:
        """Parse a single section (object_name -> attrs) to BaseRoom.
        Keeps 'agent' as a regular object for evaluation symmetry.
        """
        direction_mapping = {
            "north": np.array([0, 1]),
            "south": np.array([0, -1]),
            "east": np.array([1, 0]),
            "west": np.array([-1, 0])
        }
        objects: List[Object] = []
        for obj_name, obj_info in mapping.items():
            if not isinstance(obj_info, dict):
                continue
            position = obj_info.get('position')
            if not isinstance(position, list) or len(position) != 2 or not all(isinstance(x, (int, float, str)) for x in position):
                continue
            pos = np.array([float(position[0]), float(position[1])])
            facing = obj_info.get('facing', None)
            if isinstance(facing, str):
                ori = direction_mapping.get(facing.lower(), direction_mapping['north'])
                has_orientation = True
            else:
                ori = np.array([0, 0])
                has_orientation = False
            objects.append(Object(name=str(obj_name).replace('_', ' '), pos=pos, ori=ori, has_orientation=has_orientation))

        return BaseRoom(objects=objects, name=room_name)


    # =============================== GT constructors =============================== 

    def _baseroom_from_gt(self, gt_room: Room, gt_agent: Agent) -> BaseRoom:
        objs: List[Object] = []
        # include all non-gate objects
        for o in gt_room.objects:
            objs.append(Object(name=o.name, pos=o.pos.copy(), ori=o.ori.copy(), has_orientation=o.has_orientation))
        # include gates
        for g in gt_room.gates:
            objs.append(Object(name=g.name, pos=g.pos.copy(), ori=g.ori.copy(), has_orientation=True))
        # include agent
        objs.append(Agent(name='agent', pos=gt_agent.pos.copy(), ori=gt_agent.ori.copy(), has_orientation=True))
        return BaseRoom(objects=objs, name='gt')

    def _build_gt_global_agent_baseroom(self, gt_room: Room, gt_agent: Agent) -> BaseRoom:
        raw = self._baseroom_from_gt(gt_room, gt_agent)
        br = transform_baseroom(raw, gt_agent.init_pos, gt_agent.init_ori)
        return self._filter_br_by_names(br, {"agent"})
    
    def _build_gt_global_baseroom(self, gt_room: Room, gt_agent: Agent, observed_set: set[str]) -> BaseRoom:
        raw = self._baseroom_from_gt(gt_room, gt_agent)
        br = transform_baseroom(raw, gt_agent.init_pos, gt_agent.init_ori)
        keep = set(observed_set) | {"agent"}
        return self._filter_br_by_names(br, keep)

    def _build_gt_local_baseroom(self, gt_room: Room, gt_agent: Agent) -> BaseRoom:
        visible = self._visible_object_names(gt_room, gt_agent)
        objs: List[Object] = []
        for name in visible:
            o = gt_room.get_object_by_name(name)
            objs.append(Object(name=o.name, pos=o.pos.copy(), ori=o.ori.copy(), has_orientation=getattr(o, 'has_orientation', True)))
        raw = BaseRoom(objects=objs, name='gt_local_raw')
        return transform_baseroom(raw, gt_agent.pos, gt_agent.ori)

    def baseroom_to_json(self, room: BaseRoom, include_gates: bool = True) -> Dict[str, Any]:
        """
        Convert a BaseRoom into a cognitive map–style JSON.

        Args:
            room (BaseRoom): the BaseRoom instance to convert
            include_gates (bool): whether to include gates in the output

        Returns:
            Dict[str, Any]: JSON-like dictionary following the cognitive map schema
        """
        ori_mapping = {(0, 1): "north", (0, -1): "south", (1, 0): "east", (-1, 0): "west"}
        out: Dict[str, Any]={}
        # Objects (includes agent if present)
        for obj in room.objects:
            facing = ori_mapping.get(tuple(obj.ori), "")
            out[obj.name] = {
                "position": [int(obj.pos[0]), int(obj.pos[1])],
                "facing": facing
            }

        # Gates
        if include_gates and room.gates:
            for g in room.gates:
                gate_facing = ori_mapping.get(tuple(g.ori), "")
                out[g.name] = {
                    "position": [int(g.pos[0]), int(g.pos[1])],
                    "facing": gate_facing
                }

        return out

    # =============================== Room comparisons =============================== 

    def _compare_baserooms(self, pred_room: BaseRoom, gt_room: BaseRoom) -> MapCogMetrics:
        m = compute_map_metrics(
            pred_room,
            gt_room,
            allow_scale=bool(self.config.get('pos_allow_scale', False)),
            pos_norm_L=self._pos_norm_L,
        )
        return m

    # =============================== Filters and preprocessing =============================== 
    def _filter_br_by_names(self, br: Optional[BaseRoom], keep: set[str]) -> BaseRoom:
        if br is None:
            return BaseRoom(objects=[], name='empty')
        objs = [o for o in br.objects if o.name in keep]
        return BaseRoom(objects=objs, name=br.name)

    def _visible_object_names(self, gt_room: Room, gt_agent: Agent) -> set[str]:
        names = set()
        for o in gt_room.all_objects:
            if BaseAction._is_visible(gt_agent, o):
                names.add(o.name)
        return names

    def _preprocess_predicted(self, json_data: Dict[str, Any], observed: set[str], visible: set[str], gt_room: Room, gt_agent: Agent, map_type: str) -> Dict[str, Any]:
        jd = copy.deepcopy(json_data) if isinstance(json_data, dict) else {}
        gate_names = {g.name for g in gt_room.gates}
        
        def _norm_face_local(f, anchor_ori):
            """For local/room sections - convert relative directions to absolute based on anchor orientation"""
            if not isinstance(f, str):
                return f
            s = f.strip().lower()
            # anchor_ori is like [0,1] for north, [1,0] for east, etc.
            if tuple(anchor_ori) == (0, 1):  # north
                mapping = {"+x": "east", "-x": "west", "+y": "north", "-y": "south"}
            elif tuple(anchor_ori) == (1, 0):  # east
                mapping = {"+x": "south", "-x": "north", "+y": "east", "-y": "west"}
            elif tuple(anchor_ori) == (0, -1):  # south
                mapping = {"+x": "west", "-x": "east", "+y": "south", "-y": "north"}
            elif tuple(anchor_ori) == (-1, 0):  # west
                mapping = {"+x": "north", "-x": "south", "+y": "west", "-y": "east"}
            else:
                # fallback to identity
                return s
            return mapping.get(s, s)

        def _norm_face_global(f):
            """Best-effort: normalize common variants (incl. ego terms) to cardinal directions."""
            if not isinstance(f, str):
                return f
            s = f.strip().lower()
            mapping = {
                # canonical
                "north": "north", "n": "north",
                "south": "south", "s": "south",
                "east": "east", "e": "east",
                "west": "west", "w": "west",
                # local axis variants (treat as global frame where north=+y, east=+x)
                "+y": "north", "-y": "south",
                "+x": "east", "-x": "west",
                # ego variants (robustness; assume global frame uses initial-facing-as-north)
                "forward": "north", "front": "north", "ahead": "north",
                "back": "south", "backward": "south", "behind": "south",
                "right": "east",
                "left": "west",
            }
            return mapping.get(s, s)

        def _norm_map(obj_map: Dict[str, Any], keep: set = None, anchor_ori = None, face_fn=None) -> Dict[str, Any]:
            out = {}
            for name, info in (obj_map or {}).items():
                if not isinstance(info, dict):
                    continue
                # Apply keep filter if provided
                if keep is not None:
                    should_keep, preferred_key = _should_keep_key(name, keep)
                    if not should_keep:
                        continue
                    name = preferred_key
                # normalize facing
                if "facing" in info:
                    if face_fn is not None:
                        info["facing"] = face_fn(info["facing"])
                    elif anchor_ori is not None:
                        info["facing"] = _norm_face_local(info["facing"], anchor_ori)
                out[name] = info
            return out

        def _should_keep_key(key: str, keep_set: set) -> tuple[bool, str]:
            """Check if key should be kept and return the preferred key name from keep_set"""
            if key in keep_set:
                return True, key
            # Check if key with underscores matches any keep element without underscores
            if '_' in key:
                key_no_underscore = key.replace('_', '')
                for keep_item in keep_set:
                    if keep_item.replace('_', '') == key_no_underscore:
                        return True, keep_item

            return False, key

        def _flatten_nested_json(jd: Dict[str, Any]) -> Dict[str, Any]:
            """Convert nested JSON format (objects/gates arrays) to flat format."""
            if not (isinstance(jd, dict) and ("objects" in jd or "gates" in jd)):
                return jd
                
            flat = {}

            # keep other top-level dict entries (e.g., "agent")
            for k, v in jd.items():
                if k in ("objects", "gates"):
                    continue
                if isinstance(v, dict):
                    flat[str(k).replace('_', ' ')] = v

            def _norm_name(x):
                s = x.get("label") or x.get("name") or x.get("id") or x.get("type") or ""
                s = str(s).strip()
                return s.replace("_", " ") if s else ""

            def _emit(name, info):
                if not name:
                    return
                # require a position-like field
                pos = info.get("position") or info.get("pos") or info.get("xy")
                if pos is None:
                    return
                out = {"position": pos}
                if "facing" in info:
                    out["facing"] = info["facing"]
                # keep other fields, but don't clobber position/facing
                for k, v in info.items():
                    if k not in ("position", "pos", "xy", "facing", "id", "label", "name", "type"):
                        out[k] = v
                flat[name] = out

            def _add(sec):
                if isinstance(sec, list):
                    for it in sec:
                        if isinstance(it, dict):
                            _emit(_norm_name(it), it)
                elif isinstance(sec, dict):
                    for name, info in sec.items():
                        if isinstance(info, dict):
                            _emit(str(name).replace('_', ' '), info)

            _add(jd.get("objects"))
            _add(jd.get("gates"))
            return flat or jd

        # --- Global: keep observed + gates + agent; also handle list-based sections ---
        if map_type == "global":
            # Flatten {"objects":[...], "gates":[...]} into {name: {position, facing, ...}}
            jd = _flatten_nested_json(jd)
            keep = set(observed) | gate_names | {"agent"}
            jd = _norm_map(jd, keep, face_fn=_norm_face_global)
            return self._parse_section_to_baseroom(jd, "pred_global") or BaseRoom(objects=[], name="pred_global")

        # --- Local: drop origin + keep only visible objects ---
        if map_type == "local":
            # Handle nested format if present
            jd = _flatten_nested_json(jd)
            if "objects" in jd:
                jd = jd["objects"]
            jd = _norm_map(jd, visible, gt_agent.ori)
            return self._parse_section_to_baseroom(jd, "pred_local") or BaseRoom(objects=[], name="pred_local")

        raise ValueError(f"Invalid map_type: {map_type}")


    def _ensure_pos_norm_L(self, gt_room: Room, gt_agent: Agent) -> None:
        if self._pos_norm_L is not None:
            return
        raw = self._baseroom_from_gt(gt_room, gt_agent)
        br = transform_baseroom(raw, gt_agent.init_pos, gt_agent.init_ori)
        keep = {o.name for o in gt_room.objects}
        br = self._filter_br_by_names(br, keep)
        if not br.objects:
            self._pos_norm_L = 1.0
            return
        P = np.array([o.pos for o in br.objects], dtype=float)
        L = float(np.sqrt((P ** 2).sum(axis=1).mean()))
        self._pos_norm_L = (L if L > 0 else 1.0)


def test_evaluate_cogmaps():
    """Test function to demonstrate calling CognitiveMapManager.evaluate_cogmaps method."""
    import json
    import numpy as np

    # Path to the JSON file
    json_file_path = "results-test/GLM-4.5V/5fde50e6fe43edcb/vision/active/think/exploration_turn_logs.json"

    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    turn_log = data[2]

    print(f"Selected turn number: {turn_log.get('turn_number', 'Unknown')}")
    print(f"Total turns available: {len(data)}")

    observed_items = turn_log.get('observed_items', [])
    room_state_data = turn_log.get('room_state', {})
    agent_state_data = turn_log.get('agent_state', {})

    # Prepare responses by type
    # Since cogmap_response is None, let's use the original_response from cogmap_log
    responses_by_type = {}
    cogmap_log = turn_log.get('cogmap_log', {})

    # Extract original responses from each cogmap type'global', 'local', 
    
    print(f"\n📋 Constructing gt_room and gt_agent from turn data...")

    # Import required classes
    from ..core.room import Room
    from ..core.object import Agent

    # Construct gt_room directly from room_state_data using Room.from_dict
    gt_room = Room.from_dict(room_state_data)

    # Construct gt_agent from agent_state_data using Agent.from_dict
    gt_agent = Agent.from_dict(agent_state_data)

    print(f"✅ Constructed gt_room with {len(gt_room.objects)} objects and {len(gt_room.gates)} gates")
    print(f"✅ Constructed gt_agent at position {gt_agent.pos} facing {gt_agent.ori}")

    # Create CognitiveMapManager instance
    manager = CognitiveMapManager(cogmap_type="standard", pos_allow_scale=False, scope="all")

    print(f"\n🚀 Calling manager.evaluate_cogmaps()...")

    # Extract correct coordinates directly
    all_correct_coords_raw = turn_log['exploration_log'].get('all_correct_coords', [])
    all_correct_coords = [(int(pt[0]), int(pt[1])) for pt in all_correct_coords_raw] if all_correct_coords_raw else []

    # Call the actual evaluate_cogmaps method
    result = manager.evaluate_cogmaps(
        responses_by_type, 
        gt_room, 
        gt_agent, 
        observed_items, 
        all_correct_coords=all_correct_coords
    )

    print(f"✅ Successfully called evaluate_cogmaps!")
    print(f"📊 Result type: {type(result)}")

    # Display the results
    if result:
        print(f"\n📊 New evaluation results:")
        result_dict = result.to_dict()
        for map_type, log_data in result_dict.items():
            if isinstance(log_data, dict) and 'metrics' in log_data:
                metrics = log_data['metrics']
                print(f"   {map_type}: {metrics}")


if __name__ == "__main__":
    # Run the test function
    test_evaluate_cogmaps()