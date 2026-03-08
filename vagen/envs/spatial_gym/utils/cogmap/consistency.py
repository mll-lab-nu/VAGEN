from typing import Dict, List, Any
import numpy as np
import copy

from ...core.room import BaseRoom
from ...core.object import Agent
from .metrics import compute_map_metrics
from .types import MapCogMetrics
from .transforms import transform_baseroom


def compare_on_common_subset(a: BaseRoom | None, b: BaseRoom | None, allow_scale: bool, pos_norm_L: float | None) -> MapCogMetrics:
    if a is None or b is None:
        return MapCogMetrics.invalid()
    names_a = {o.name for o in a.objects}
    if not names_a:
        return MapCogMetrics(dir=1.0, facing=1.0, overall=1.0, pos=1.0)  # No objects in A, trivially perfect
    names_b = {o.name for o in b.objects}
    names = names_a & names_b
    if not names:
        # No overlap -> treat as wrong
        return MapCogMetrics(dir=0.0, facing=0.0, overall=0.0, pos=0.0, valid=True)
    a_sub = BaseRoom(objects=[o for o in a.objects if o.name in names], name=a.name)
    b_sub = BaseRoom(objects=[o for o in b.objects if o.name in names], name=b.name)
    return compute_map_metrics(a_sub, b_sub, allow_scale=allow_scale, pos_norm_L=pos_norm_L)


def local_vs_global_consistency(pred_local: BaseRoom | None, pred_global: BaseRoom | None, agent: Agent, allow_scale: bool, pos_norm_L: float | None) -> MapCogMetrics:
    if pred_local is None or pred_global is None:
        return MapCogMetrics.invalid()
    
    # Find predicted agent in global map
    global_agent = next((o for o in pred_global.objects if o.name == 'agent'), None)
    if global_agent is None:
        return MapCogMetrics.invalid()
    
    # Transform global map to use predicted agent as origin (make copy to avoid modifying original)
    
    global_copy = copy.deepcopy(pred_global)
    global_agent_centered = transform_baseroom(global_copy, global_agent.pos, global_agent.ori)
    
    # Compare directly (local should already be agent-centered)
    return compare_on_common_subset(pred_local, global_agent_centered, allow_scale=allow_scale, pos_norm_L=pos_norm_L)

def _is_valid_for_facing(name: str, gt_curr_dict: Dict, gt_prev_dict: Dict) -> bool:
    """Check if object is valid for facing evaluation (not agent, not gate, has orientation)."""
    if name == 'agent':
        return False
        
    # Check exclusion based on GT if available
    if name in gt_curr_dict:
        gt_obj = gt_curr_dict[name]
        is_gate = "door" in name.lower() or "gate" in name.lower()
        if not gt_obj.has_orientation or is_gate:
            return False
    elif name in gt_prev_dict:
        gt_obj = gt_prev_dict[name]
        is_gate = "door" in name.lower() or "gate" in name.lower()
        if not gt_obj.has_orientation or is_gate:
            return False
    else:
        # Fallback exclusion based on name
        if "door" in name.lower() or "gate" in name.lower():
            return False
            
    return True

def stability(
    env_data_or_logs: Dict | List[Dict],
    threshold: int = 1,
) -> Dict[str, List[float | None]]:
    """Per-adjacent-turn stability metrics (merged update/stability).

    For each adjacent exploration turn (t-1 -> t), for every object observed before t:
    - position: Only objects whose possible-position count changes by <= `threshold`; then check non-worse
    - facing: Check non-worse

    Returns:
        Dict with keys: 'position_update', 'facing_update', 'position_stability', 'facing_stability'
        Each value is a List[float] (or None where invalid/not applicable)
    """
    # Normalize input to a list of exploration turns
    if isinstance(env_data_or_logs, dict):
        logs = env_data_or_logs.get('env_turn_logs', []) or []
    else:
        logs = env_data_or_logs or []

    expl = [t for t in logs if t.get('is_exploration_phase')]
    
    out = {
        'position_update': [],
        'facing_update': [],
        'position_stability': [],
        'facing_stability': []
    }

    if len(expl) <= 1:
        return out

    def _pos_non_worse(name: str,
                       pred_prev_dict: Dict[str, Any], pred_curr_dict: Dict[str, Any],
                       gt_prev_dict: Dict[str, Any], gt_curr_dict: Dict[str, Any]) -> float | None:
        prev_dist = np.linalg.norm(np.array(pred_prev_dict[name].pos) - np.array(gt_prev_dict[name].pos))
        curr_dist = np.linalg.norm(np.array(pred_curr_dict[name].pos) - np.array(gt_curr_dict[name].pos))
        return 1.0 if curr_dist <= prev_dist else 0.0

    def _facing_non_worse(name: str,
                          pred_prev_dict: Dict[str, Any], pred_curr_dict: Dict[str, Any],
                          gt_prev_dict: Dict[str, Any], gt_curr_dict: Dict[str, Any]) -> float | None:
        if not _is_valid_for_facing(name, gt_curr_dict, gt_prev_dict):
            return None
        prev_correct = np.array_equal(pred_prev_dict[name].ori, gt_prev_dict[name].ori)
        curr_correct = np.array_equal(pred_curr_dict[name].ori, gt_curr_dict[name].ori)
        if prev_correct and not curr_correct:
            return 0.0
        if curr_correct and not prev_correct:
            return 1.0
        unchanged = np.array_equal(pred_curr_dict[name].ori, pred_prev_dict[name].ori)
        return 1.0 if unchanged else 0.0

    for i in range(1, len(expl)):
        prev_log = expl[i - 1]
        curr_log = expl[i]
        prev_exp = prev_log.get('exploration_log') or {}
        curr_exp = curr_log.get('exploration_log') or {}
        
        observed_before = set(prev_exp.get('observed_items') or [])

        # Need previous and current predicted and GT global rooms
        g_prev = ((prev_log.get('cogmap_log') or {}).get('global') or {})
        pred_prev = BaseRoom.from_dict((g_prev.get('pred_room_state')) or {})
        gt_prev = BaseRoom.from_dict((g_prev.get('gt_room_state_full') or g_prev.get('gt_room_state')) or {})

        g_curr = ((curr_log.get('cogmap_log') or {}).get('global') or {})
        pred_curr = BaseRoom.from_dict((g_curr.get('pred_room_state')) or {})
        gt_curr = BaseRoom.from_dict((g_curr.get('gt_room_state_full') or g_curr.get('gt_room_state')) or {})

        pred_prev_dict = {o.name: o for o in pred_prev.objects}
        pred_curr_dict = {o.name: o for o in pred_curr.objects}
        gt_prev_dict = {o.name: o for o in gt_prev.objects}
        gt_curr_dict = {o.name: o for o in gt_curr.objects}
        common_names = set(pred_curr_dict) & set(pred_prev_dict) & set(gt_curr_dict) & set(gt_prev_dict)

        stable_names = (observed_before & common_names) - {'agent'}

        prev_possible = prev_exp.get('possible_positions') or {}
        curr_possible = curr_exp.get('possible_positions') or {}

        def _n_possible(d: Dict[str, Any], name: str) -> int | None:
            v = d.get(name)
            return len(v) if isinstance(v, list) else None

        pos_stab_names: List[str] = []
        for n in stable_names:
            a = _n_possible(prev_possible, n)
            b = _n_possible(curr_possible, n)
            if a is not None and b is not None and abs(a - b) <= threshold:
                pos_stab_names.append(n)
        facing_stab_names = stable_names

        pos_stab_scores = [_pos_non_worse(n, pred_prev_dict, pred_curr_dict, gt_prev_dict, gt_curr_dict) for n in pos_stab_names]
        fac_stab_scores = [_facing_non_worse(n, pred_prev_dict, pred_curr_dict, gt_prev_dict, gt_curr_dict) for n in facing_stab_names]
        pos_stab_scores = [s for s in pos_stab_scores if s is not None]
        fac_stab_scores = [s for s in fac_stab_scores if s is not None]

        pos_stab_avg = float(np.mean(pos_stab_scores)) if pos_stab_scores else None
        fac_stab_avg = float(np.mean(fac_stab_scores)) if fac_stab_scores else None
        out['position_stability'].append(pos_stab_avg)
        out['facing_stability'].append(fac_stab_avg)
        out['position_update'].append(pos_stab_avg)
        out['facing_update'].append(fac_stab_avg)

    return out


__all__ = [
    "compare_on_common_subset",
    "local_vs_global_consistency",
    "stability",
]

if __name__ == "__main__":
    pass
