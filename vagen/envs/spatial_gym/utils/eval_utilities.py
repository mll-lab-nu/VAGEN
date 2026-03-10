"""Evaluation helpers for spatial tasks."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
from types import SimpleNamespace
import math
import re
import json
import numpy as np

from ..actions.base import BaseAction
from ..core.relationship import PairwiseRelationshipDiscrete, OrientationRel

if TYPE_CHECKING:
    from .cogmap.metrics import compute_pos_sim


# ========== Constants ==========
_COORD_TOL = 1e-3
_BACKWARD_LOC_DEFAULT_THRESHOLD = 0.8
_E2A_DEFAULT_SIM_THRESHOLD = 0.9
_COS_45 = float(np.cos(np.deg2rad(45)))
_ORI_TO_DEG = {
    (0, 1): 0,
    (1, 1): 45,
    (1, 0): 90,
    (1, -1): 135,
    (0, -1): 180,
    (-1, -1): 225,
    (-1, 0): 270,
    (-1, 1): 315,
}
_DEG_TO_ORI = {deg: ori for ori, deg in _ORI_TO_DEG.items()}

# Label aliases for direction and distance
_LABEL_ALIASES: Dict[str, Sequence[str]] = {
    # Egocentric
    "front-left": ("front-left", "left-front", "front left", "left front", "frontleft", "leftfront", "fl"),
    "front-right": ("front-right", "right-front", "front right", "right front", "frontright", "rightfront", "fr"),
    "front-slight-left": ("front-slight-left", "front slightly left", "slightly front left", "frontslightleft", "slight front left"),
    "front-slight-right": ("front-slight-right", "front slightly right", "slightly front right", "frontslightright", "slight front right"),
    "mid distance": ("mid distance", "mid-distance", "mid", "medium", "medium distance"),
    "same distance": ("same distance", "same", "same-distance"),
    "slightly far": ("slightly far", "slightly-far"),
    "very far": ("very far", "very-far"),
    "extremely far": ("extremely far", "extremely-far", "extreme", "extreme far"),
    
    # Cardinal
    "north": ("north", "n"),
    "south": ("south", "s"),
    "east": ("east", "e"),
    "west": ("west", "w"),
    "north-east": ("north-east", "north east", "northeast", "ne"),
    "north-west": ("north-west", "north west", "northwest", "nw"),
    "south-east": ("south-east", "south east", "southeast", "se"),
    "south-west": ("south-west", "south west", "southwest", "sw"),
}


# ========== Text Normalization ==========
def _normalize_joined(text: str) -> str:
    """Remove whitespace and hyphens, convert to lowercase."""
    return re.sub(r"[\s-]+", "", text.lower())

def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace and convert to lowercase."""
    return re.sub(r"\s+", " ", text.strip()).lower()

def _require_text(value: Any) -> Optional[str]:
    """Return stripped text if non-empty, else None."""
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    return None

def _casefold(text: Any) -> str:
    """Casefold string."""
    return str(text).casefold()


# ========== Label Matching System ==========
_LABEL_LOOKUP: Dict[str, str] = {}
_LABEL_COMPONENTS: Dict[str, set[str]] = {}

def _build_label_system() -> None:
    """Build label alias lookup and component tables."""
    for canonical, variants in _LABEL_ALIASES.items():
        canonical_norm = _normalize_joined(canonical)
        _LABEL_LOOKUP[canonical_norm] = canonical_norm
        
        parts = {canonical_norm}
        _LABEL_COMPONENTS[canonical_norm] = parts
        
        for variant in variants:
            _LABEL_LOOKUP[_normalize_joined(variant)] = canonical_norm

_build_label_system()

def _canonicalize_label(label: Any) -> str:
    """Convert label to canonical form."""
    norm = _normalize_joined(str(label)) if label is not None else ""
    return _LABEL_LOOKUP.get(norm, norm)

def _label_components(label: Any) -> set[str]:
    """Get component parts of a label."""
    canonical = _canonicalize_label(label)
    return _LABEL_COMPONENTS.get(canonical, {canonical})

def _labels_match(pred_label: Any, target_label: Any) -> bool:
    """Check if two labels match (exact or partial component match)."""
    pred_canonical = _canonicalize_label(pred_label)
    target_canonical = _canonicalize_label(target_label)
    
    if not pred_canonical or not target_canonical:
        return False
    if pred_canonical == target_canonical:
        return True
    
    target_parts = _label_components(target_label)
    pred_parts = _label_components(pred_label)
    return (pred_canonical in target_parts) or (target_canonical in pred_parts)


# ========== Parsing Utilities ==========
def extract_sequence(
    raw: Any,
    value_type: type = str,
    clean_pattern: str | None = None,
) -> Optional[List[Any]]:
    """Extract sequence from various formats (list, tuple, or string)."""
    if raw is None:
        return None
    
    # Direct list/tuple
    if isinstance(raw, (list, tuple)):
        try:
            return [value_type(item) for item in raw]
        except (TypeError, ValueError):
            return None
    
    # Parse from string
    text = str(raw)
    content = text
    
    # Extract from parentheses or brackets
    if '(' in text and ')' in text:
        match = re.search(r"\((.*?)\)", text)
        if match:
            content = match.group(1)
    elif '[' in text and ']' in text:
        match = re.search(r"\[(.*?)\]", text)
        if match:
            content = match.group(1)
    
    # Parse comma-separated values
    items: List[Any] = []
    for token in content.split(','):
        value = token.strip().strip("'\"")
        if not value:
            continue
        if clean_pattern:
            value = re.sub(clean_pattern, '', value, flags=re.IGNORECASE)
        try:
            if value_type is int:
                items.append(int(value))
            elif value_type is float:
                items.append(float(value))
            else:
                items.append(value_type(value) if value_type is not str else value)
        except (TypeError, ValueError):
            return None
    
    return items or None

def extract_elements(
    raw: Any,
    expected_type: type = str,
    clean_pattern: str | None = None,
) -> Optional[List[Any]]:
    """Public API for sequence extraction (alias for extract_sequence)."""
    return extract_sequence(raw, expected_type, clean_pattern)

def _parse_direction_distance(raw: str) -> Optional[Tuple[str, str]]:
    """Parse direction and distance from text."""
    # Try as comma-separated list
    items = extract_sequence(raw, str)
    if items and len(items) >= 2:
        return items[0], items[1]
    
    # Try semicolon or newline separated
    segments = [seg.strip() for seg in re.split(r"[;\n]", str(raw)) if seg.strip()]
    if len(segments) >= 2:
        return segments[0], segments[1]
    
    # Try space/comma separated words
    words = [word.strip() for word in re.split(r"[ ,]", str(raw)) if word.strip()]
    if len(words) >= 2:
        return (' '.join(words[:-1]), words[-1]) if len(words) > 2 else (words[0], words[1])
    
    return None

def _parse_coordinate_list(raw: Any) -> Optional[List[Tuple[float, float]]]:
    """Parse list of (x, y) coordinates from text."""
    matches = re.findall(r"-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?", str(raw))
    if not matches:
        return None
    
    coords: List[Tuple[float, float]] = []
    for pair in matches:
        x_str, y_str = pair.split(',')
        try:
            coords.append((float(x_str.strip()), float(y_str.strip())))
        except ValueError:
            return None
    
    return coords or None

def _parse_indexed_lines(raw: Any) -> List[Tuple[int, str]]:
    """Parse numbered list format: '1. item'."""
    if not isinstance(raw, str):
        return []
    
    pairs: List[Tuple[int, str]] = []
    for line in raw.splitlines():
        text = line.strip()
        if not text or '. ' not in text:
            continue
        idx_part, value = text.split('. ', 1)
        try:
            idx = int(idx_part.strip()) - 1
        except ValueError:
            continue
        pairs.append((idx, value.strip()))
    
    return pairs

def _parse_coord_orientation(text: str) -> Optional[Tuple[Tuple[float, float], str]]:
    """Parse coordinate and orientation from text like '(x, y) facing north'."""
    if not isinstance(text, str):
        return None
    
    coord_match = re.search(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", text)
    if not coord_match:
        return None
    
    coord = (float(coord_match.group(1)), float(coord_match.group(2)))
    
    ori_match = re.search(r'facing\s+([a-zA-Z-]+)', text, re.IGNORECASE)
    if ori_match:
        orientation = ori_match.group(1).strip().lower()
    else:
        fallback = re.findall(r'(north|east|south|west)', text, re.IGNORECASE)
        if not fallback:
            return None
        orientation = fallback[-1].strip().lower()
    
    return coord, orientation

def _parse_action_sequence(text: str) -> List[Tuple[str, Any]]:
    """Parse navigation actions: 'JumpTo(obj), Rotate(90)'."""
    actions: List[Tuple[str, Any]] = []
    
    for token in [p.strip() for p in str(text).split(',') if p.strip()]:
        jump = re.match(r'JumpTo\s*\(\s*(.+?)\s*\)', token, re.IGNORECASE)
        if jump:
            actions.append(('jumpto', jump.group(1).strip()))
            continue
        
        rotate = re.match(r'Rotate\s*\(\s*(-?\d+)\s*\)', token, re.IGNORECASE)
        if rotate:
            actions.append(('rotate', int(rotate.group(1))))
            continue
        
        return []  # Invalid format
    
    return actions


# ========== Geometry Utilities ==========
def _coerce_point(pt: Sequence[Any]) -> Tuple[float, float]:
    """Convert sequence to (x, y) float tuple."""
    if not isinstance(pt, (list, tuple, np.ndarray)) or len(pt) < 2:
        raise ValueError("invalid point")
    return float(pt[0]), float(pt[1])

def _coord_norm_from_gt(coords: Sequence[Sequence[float]]) -> float:
    """Calculate RMS norm from ground truth coordinates."""
    arr = np.array(coords, dtype=float)
    if arr.size == 0:
        return 1.0
    norms = (arr * arr).sum(axis=1)
    mean_norm = float(norms.mean()) if norms.size else 0.0
    return float(math.sqrt(mean_norm)) if mean_norm > 0 else 1.0

def _coords_to_room(coords: Sequence[Sequence[float]]) -> SimpleNamespace:
    """Convert coordinates to room-like object for metric computation."""
    objects = [
        SimpleNamespace(name=str(idx), pos=np.array(pt, dtype=float))
        for idx, pt in enumerate(coords)
    ]
    return SimpleNamespace(objects=objects)

def _normalize_orientation(ori: Sequence[int]) -> Tuple[int, int]:
    """Normalize orientation to one of 8 headings."""
    vec = np.array([float(ori[0]), float(ori[1])], dtype=float)
    if np.allclose(vec, 0.0):
        return (0, 1)
    vec = vec / float(np.linalg.norm(vec))
    cand = (int(np.sign(vec[0])), int(np.sign(vec[1])))
    if cand in _ORI_TO_DEG:
        return cand

    basis = np.array(list(_ORI_TO_DEG.keys()), dtype=float)
    idx = int(np.argmax(basis @ vec))
    return tuple(int(x) for x in basis[idx])

def _rotate_orientation(ori: Tuple[int, int], degrees: int) -> Tuple[int, int]:
    """Rotate orientation by given degrees (must be multiple of 45)."""
    if int(degrees) % 45 != 0:
        raise ValueError(f"Invalid rotation: {degrees}")
    
    cur_deg = _ORI_TO_DEG.get(_normalize_orientation(ori), 0)
    new_deg = (cur_deg + int(degrees)) % 360
    return _DEG_TO_ORI.get(new_deg, (0, 1))

def _is_visible_from(
    agent_pos: Tuple[float, float],
    agent_ori: Tuple[int, int],
    target_pos: Tuple[float, float],
    fov: int = 90,
    agent_room_id: Any = None,
    target_room_id: Any = None,
) -> bool:
    """Check if target is visible from agent position and orientation."""
    def _norm_room_id(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        return list(v) if isinstance(v, (list, tuple)) else v

    from_obj = SimpleNamespace(
        pos=np.asarray(agent_pos, dtype=float),
        ori=np.asarray(agent_ori, dtype=float),
        room_id=_norm_room_id(agent_room_id),
    )
    to_obj = SimpleNamespace(
        pos=np.asarray(target_pos, dtype=float),
        room_id=_norm_room_id(target_room_id),
    )
    return BaseAction._is_visible(from_obj, to_obj, field_of_view=fov)


# ========== Navigation Simulation ==========
def _simulate_navigation(
    actions: List[Tuple[str, Any]],
    init_pos: Tuple[float, float],
    init_ori: Tuple[int, int],
    object_positions: Dict[str, Tuple[float, float]],
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[int, int]], Optional[str]]:
    """Simulate navigation actions and return final position, orientation, and any error."""
    pos = _coerce_point(init_pos)
    ori = _normalize_orientation(init_ori)
    
    for action_type, param in actions:
        if action_type == 'rotate':
            try:
                ori = _rotate_orientation(ori, int(param))
            except ValueError:
                return None, None, 'invalid_rotation'
        
        elif action_type == 'jumpto':
            name = str(param).lower()
            target = object_positions.get(name)
            if target is None:
                return None, None, 'unknown_object'
            if not _is_visible_from(pos, ori, target):
                return None, None, 'target_not_visible'
            pos = target
        
        else:
            return None, None, 'invalid_action'
    
    return pos, ori, None


# ========== Task-Specific Evaluators ==========
def _eval_token_sequence(pred: str, answer: Sequence[str]) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate exact token sequence match."""
    tokens = extract_sequence(pred, str)
    if not tokens or len(tokens) != len(answer):
        return False, {}
    
    expected = [_normalize_whitespace(item) for item in answer]
    actual = [_normalize_whitespace(token) for token in tokens]
    return actual == expected, {}

def _eval_direction_text(pred: str, answer: Union[str, Sequence[str]]) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate direction and distance pair."""
    parsed = _parse_direction_distance(pred)
    
    # Normalize answer
    normalized_answer = None
    if isinstance(answer, (list, tuple)) and len(answer) >= 2:
        normalized_answer = (_canonicalize_label(str(answer[0])), _canonicalize_label(str(answer[1])))
    else:
        parsed_ans = _parse_direction_distance(str(answer))
        if parsed_ans:
            normalized_answer = (_canonicalize_label(parsed_ans[0]), _canonicalize_label(parsed_ans[1]))
    
    if not parsed or not normalized_answer:
        return False, {}
    
    # Canonicalize parsed prediction
    normalized_pred = (_canonicalize_label(parsed[0]), _canonicalize_label(parsed[1]))
    
    # Check normal order
    match_normal_0 = _labels_match(normalized_pred[0], normalized_answer[0])
    match_normal_1 = _labels_match(normalized_pred[1], normalized_answer[1])
    score_normal = (1.0 if match_normal_0 else 0.0) + (1.0 if match_normal_1 else 0.0)
    score_normal /= 2.0
    
    # Check swapped order
    match_swapped_0 = _labels_match(normalized_pred[1], normalized_answer[0])
    match_swapped_1 = _labels_match(normalized_pred[0], normalized_answer[1])
    score_swapped = (1.0 if match_swapped_0 else 0.0) + (1.0 if match_swapped_1 else 0.0)
    score_swapped /= 2.0
    
    return max(score_normal, score_swapped), {}


def _eval_exact_text(pred: str, answer: Union[str, Sequence[str]]) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate exact text match with label matching."""
    return _labels_match(pred, answer), {}

def _eval_forward_nav(pred: str, answer: str) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate forward navigation/localization task."""
    return _eval_direction_text(pred, answer)


def resolve_gate_orientation(
    gate_room_ids: Union[List[int], Tuple[int, ...], int, None],
    gate_ori_by_room: Dict[int, Tuple[int, int]],
    gate_base_ori: Tuple[int, int],
    agent_room_ids: Union[List[int], Tuple[int, ...], int, None]
) -> Tuple[int, int]:
    """Resolve effective orientation of a gate based on agent's room."""
    # Normalize inputs to sets
    def _to_set(x):
        if x is None: return set()
        if isinstance(x, (list, tuple)): return set(x)
        return {x}

    a_rids = _to_set(agent_room_ids)
    g_rids = _to_set(gate_room_ids)
    
    # Find shared room
    intersection = list(a_rids & g_rids)
    rid = intersection[0] if len(intersection) == 1 else None
    
    if rid is not None and int(rid) in gate_ori_by_room:
         return gate_ori_by_room[int(rid)]
    return gate_base_ori


def check_fov_consistency(agent_pos: Tuple[float, float], agent_ori: Tuple[int, int], answer: Dict[str, Any]) -> bool:
    """Check if the agent's view matches the ground truth observation in answer."""
    final_visible = answer.get('final_observation', [])
    assert final_visible, "final_observation is empty"

    obs_visibles = {}
    for item in final_visible:
        name = str(item.get('name', '')).strip().lower()
        direction = _canonicalize_label(item.get('direction', ''))
        distance = _canonicalize_label(item.get('distance', ''))
        orientation = _canonicalize_label(item.get('orientation')) if item.get('orientation') else None
        if name and direction and distance:
            obs_visibles[name] = (direction, distance, orientation)
            
    object_positions = {str(k).lower(): _coerce_point(v) for k, v in answer.get('object_positions', {}).items()}
    object_orientations = {str(k).lower(): _normalize_orientation(v) for k, v in answer.get('object_orientations', {}).items()}
    
    # Extract extra info for gates if available
    gate_info = answer.get('gate_info', {})
    agent_room_ids = answer['room_id']

    for obj_name, (dir_gt, dist_gt, ori_gt) in obs_visibles.items():
        target_pos = object_positions.get(obj_name)
        
        # Check if object exists and is visible
        if target_pos is None or not _is_visible_from(
            agent_pos,
            agent_ori,
            target_pos,
            agent_room_id=agent_room_ids,
            target_room_id=answer.get('object_rooms', {}).get(obj_name),
        ):
            return False
        
        # Check relations (Direction, Distance)
        rel = PairwiseRelationshipDiscrete.relationship(target_pos, agent_pos, anchor_ori=agent_ori)
        if _canonicalize_label(rel.direction.bin_label) != dir_gt or \
           _canonicalize_label(rel.dist.bin_label) != dist_gt:
            return False
            
        # Check relative orientation
        if ori_gt:
            # Skip if target object orientation is not known
            if obj_name in object_orientations:
                target_ori = object_orientations[obj_name]
                is_gate = False
                
                # Handle gate logic if info available
                if obj_name in gate_info:
                    is_gate = True
                    g_info = gate_info[obj_name]
                    # Parse gate info (handling json serialization types)
                    g_rids = g_info.get('room_ids')
                    g_ori_map = {int(k): tuple(v) for k, v in g_info.get('ori_by_room', {}).items()}
                    
                    target_ori = resolve_gate_orientation(
                        gate_room_ids=g_rids,
                        gate_ori_by_room=g_ori_map,
                        gate_base_ori=target_ori,
                        agent_room_ids=agent_room_ids
                    )

                # Calculate relative orientation using OrientationRel
                rel_ori = OrientationRel.get_relative_orientation(target_ori, agent_ori)
                rel_label = OrientationRel.to_string(rel_ori, perspective='ego', if_gate=is_gate)
                
                if _canonicalize_label(rel_label) != ori_gt:
                    return False
    return True


def _eval_backward_nav(
    pred: str, 
    answer: Union[str, Dict], 
    require_exact_pose: bool = False,
    weight_by_steps: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate backward navigation task.
    
    Args:
        require_exact_pose: If True, checks if final pose exactly matches ground truth.
                          If False, checks if final view matches ground truth description (FOV simulation).
    """
    if not isinstance(answer, dict):
        return False, {}
    
    pred_actions = _parse_action_sequence(pred)
    if not pred_actions:
        return False, {'error': 'invalid_format'}
    
    # Extract ground truth data
    init_pos = _coerce_point(answer['init_pos'])
    init_ori = _normalize_orientation(answer['init_ori'])
    object_positions = {str(k).lower(): _coerce_point(v) for k, v in answer['object_positions'].items()}
    
    # Simulate navigation
    final_pos, final_ori, error = _simulate_navigation(pred_actions, init_pos, init_ori, object_positions)
    if error:
        return False, {'error': error}
    
    # Common matching info
    gt_pos = tuple(answer['final_pos'])
    gt_ori = tuple(answer['final_ori'])
    
    best_info = {
        'pos_match': final_pos == gt_pos,
        'ori_match': final_ori == gt_ori,
        'final_pos': final_pos,
        'final_ori': final_ori,
        'visible_match': False
    }

    if require_exact_pose:
        # Vision task: exact pose match
        is_correct = best_info['pos_match'] and best_info['ori_match']
        return (1.0 if is_correct else 0.0), best_info
    else:
        # Text task: FOV consistency match
        if final_pos and final_ori and check_fov_consistency(final_pos, final_ori, answer):
            best_info['visible_match'] = True
            score = 1.0
            if weight_by_steps:
                minimal_plan = answer.get('minimal_plan', [])
                minimal_steps = len(minimal_plan) if isinstance(minimal_plan, list) else minimal_plan
                score = min(minimal_steps / len(pred_actions), 1.0)
            return score, best_info
            
        # Fallback check (if FOV check failed or not possible, but pos matches?)
        if check_fov_consistency(gt_pos, gt_ori, answer):
             # Ground truth pose is valid (sanity check)
             pass
        
        return 0.0, best_info


def _eval_backward_nav_rev(pred: str, answer: Union[str, Dict]) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate backward navigation reverse task.

    This task requires navigating back to the starting position from a termination location.
    The action sequence must end with JumpTo(initial_pos).
    """
    if not isinstance(answer, dict):
        return False, {}

    pred_actions = _parse_action_sequence(pred)
    if not pred_actions:
        return False, {'error': 'invalid_format'}

    # Check if last action is JumpTo(initial_pos)
    last_action_type, last_action_param = pred_actions[-1]
    if last_action_type != 'jumpto' or str(last_action_param).lower() != 'initial_pos':
        return False, {'error': 'missing_final_jumpto_initial_pos'}

    # Remove the last action
    actions_without_final = pred_actions[:-1]

    # Extract ground truth data
    start_pos = _coerce_point(answer['start_pos'])  # Starting from termination location
    start_ori = _normalize_orientation(answer['start_ori'])
    target_pos = _coerce_point(answer['target_pos'])  # Target is the initial position
    object_positions = {str(k).lower(): _coerce_point(v) for k, v in answer['object_positions'].items()}

    # Add initial_pos to object_positions for visibility check
    object_positions['initial_pos'] = target_pos

    # Simulate navigation without the final JumpTo(initial_pos)
    final_pos, final_ori, error = _simulate_navigation(actions_without_final, start_pos, start_ori, object_positions)
    if error:
        return False, {'error': error}

    # Check if initial_pos (target_pos) is visible from the final position
    if final_pos is None or final_ori is None:
        return False, {'error': 'simulation_failed'}

    is_visible = _is_visible_from(final_pos, final_ori, target_pos)

    best_info = {
        'initial_pos_visible': is_visible,
        'final_pos': final_pos,
        'final_ori': final_ori,
        'target_pos': target_pos,
    }

    minimal_plan = answer.get('minimal_plan', [])
    minimal_steps = len(minimal_plan) if isinstance(minimal_plan, list) else minimal_plan
    eval_score = 1.0 if is_visible else 0.0
    return eval_score * (minimal_steps / len(pred_actions)), best_info


def _coord_similarity(pred_coord: tuple[float, float], gt_coord: tuple[float, float]) -> float:
    pred = np.array(pred_coord, dtype=float)
    gt = np.array(gt_coord, dtype=float)

    rmse = np.linalg.norm(pred - gt)
    L = np.linalg.norm(gt)

    if L == 0:
        return 1.0 if rmse == 0 else 0.0
    similarity_score = np.exp(-rmse / L)

    return similarity_score

def _score_similarity_mra(similarity: float) -> float:
    similarity = max(0, min(1, similarity))
    
    quality_thresholds = [
        0.50, 0.55, 0.60, 0.65, 0.70, 
        0.75, 0.80, 0.85, 0.90, 0.95
    ]
    
    passed_levels = 0
    for threshold in quality_thresholds:
        if similarity >= threshold:
            passed_levels += 1
            
    final_score = passed_levels / len(quality_thresholds)
    
    return final_score


def _eval_backward_pov(pred: str, answer: Union[str, Dict]) -> Tuple[float, Dict[str, Any]]:
    """Evaluate backward POV task (Text).
    
    Checks if the view from predicted object matches the description.
    """
    # If answer is just a string (old code compat), fallback
    if isinstance(answer, str):
        return _eval_exact_text(pred, answer)
    
    if not isinstance(answer, dict):
        return False, {}
    
    # 1. Identify the object by name
    pred_name = str(pred).strip()
    
    # Extract object states from answer dict
    object_positions = {str(k).lower(): _coerce_point(v) for k, v in answer.get('object_positions', {}).items()}
    # We need orientations
    object_orientations = {str(k).lower(): _normalize_orientation(v) for k, v in answer.get('object_orientations', {}).items()}
    
    pred_key = pred_name.lower()
    if pred_key not in object_positions:
        # Predicted object not found in room
        return 0.0, {'name_match': False, 'error': 'object_not_found'}
    
    pos = object_positions[pred_key]
    ori = object_orientations.get(pred_key, (0, 1)) # Default if missing
    
    # 2. Check observation consistency (Superset check)
    valid_view = check_fov_consistency(pos, ori, answer)
    
    # Debug info (name matching)
    gt_name = str(answer.get('answer', '')).strip()
    is_name_match = _labels_match(pred_name, gt_name)
    
    score = 1.0 if (is_name_match or valid_view) else 0.0
    return score, {'name_match': is_name_match, 'view_match': valid_view}


def _eval_backward_pov_vision(pred: str, answer: Union[str, Dict]) -> Tuple[float, Dict[str, Any]]:
    """Evaluate backward POV task (Vision).
    
    Only checks if the predicted answer string matches the ground truth answer string.
    Ignores FOV check.
    """
    gt_name = ""
    if isinstance(answer, dict):
        gt_name = str(answer.get('answer', ''))
    else:
        gt_name = str(answer)
        
    return _eval_exact_text(pred, gt_name)


def _eval_backward_loc(pred: str, answer: Any, use_mra: bool = False) -> Tuple[float, Dict[str, Any]]:
    """Evaluate backward localization task."""
    if not isinstance(answer, dict):
        return False, {}
    
    # Try parsing just coordinate first
    coord_pred = None
    coords = _parse_coordinate_list(pred)
    if coords and len(coords) > 0:
        coord_pred = coords[0]
    else:
        # Fallback to full parser
        parsed = _parse_coord_orientation(pred)
        if parsed:
            coord_pred, _ = parsed
            
    if coord_pred is None:
        return False, {}
    
    try:
        coord_ans = _coerce_point(answer.get('coord', ()))
    except ValueError:
        return False, {}
    
    similarity = _coord_similarity(coord_pred, coord_ans)
    
    if use_mra:
        score = _score_similarity_mra(similarity)
    else:
        score = similarity
        
    return score, {
        'similarity': similarity,
        'raw_score': score
    }


# ========== E2A Evaluation ==========
def e2a_eval_fn(pred: Any, answer: Any) -> Tuple[float, Dict[str, Any]]:
    """Evaluate E2A (Egocentric to Allocentric) coordinate prediction.
    
    Answer can be:
    - List/tuple of coordinates
    - Dict with:
        - 'coords': ground truth coordinates (required)
        - 'absolute_coords': absolute positions of selected objects for normalization (optional)
        - 'threshold': similarity threshold (optional, default 0.9)
    """
    # Import here to avoid circular dependency
    from .cogmap.metrics import compute_pos_sim, compute_dir_sim
    
    if not isinstance(pred, str):
        return 0.0, {'error': 'prediction_not_string'}
    
    coords = _parse_coordinate_list(pred)
    if not coords:
        return 0.0, {'error': 'invalid_prediction_format'}
    
    # Extract ground truth and threshold
    threshold = _E2A_DEFAULT_SIM_THRESHOLD
    gt_coords_raw: Any = answer
    norm_coords_raw: Any = None  # Coordinates to use for normalization
    
    if isinstance(answer, dict):
        threshold = float(answer.get('threshold', threshold))
        gt_coords_raw = answer.get('coords', [])
        # Use absolute coordinates for normalization if provided
        norm_coords_raw = answer.get('absolute_coords', None)
    
    if not isinstance(gt_coords_raw, (list, tuple)) or not gt_coords_raw:
        return 0.0, {'error': 'missing_ground_truth'}
    
    try:
        gt_coords = [_coerce_point(pt) for pt in gt_coords_raw]
    except ValueError:
        return 0.0, {'error': 'invalid_ground_truth'}
    
    if len(coords) != len(gt_coords):
        return 0.0, {'error': 'mismatched_coordinate_count'}
    
    # Use absolute coordinates for normalization if provided, otherwise use gt_coords
    if norm_coords_raw:
        try:
            norm_coords = [_coerce_point(pt) for pt in norm_coords_raw]
        except ValueError:
            norm_coords = gt_coords  # Fallback to gt_coords
    else:
        norm_coords = gt_coords
    
    pred_room = _coords_to_room(coords)
    gt_room = _coords_to_room(gt_coords)
    pos_norm_L = _coord_norm_from_gt(norm_coords)
    
    pos_sim = compute_pos_sim(pred_room, gt_room, allow_scale=False, pos_norm_L=pos_norm_L)
    dir_sim = compute_dir_sim(pred_room, gt_room)
    similarity = (pos_sim + dir_sim) / 2
    return similarity, {'similarity': similarity, 'threshold': threshold}


# ========== Public Helper Functions ==========
def exp_evaluate_fn(pred: str, relationships_to_query: List[Dict]) -> bool:
    """Evaluate exploration task (query object pairs or terminate)."""
    text = _require_text(pred)
    if text is None:
        return False
    
    if not relationships_to_query:
        return "terminate" in text.lower()
    
    pred_pair = extract_elements(text, expected_type=str)
    if not pred_pair or len(pred_pair) != 2:
        return False
    
    a, b = (_casefold(pred_pair[0]), _casefold(pred_pair[1]))
    for rel in relationships_to_query:
        x = _casefold(rel['object1'])
        y = _casefold(rel['object2'])
        if (a == x and b == y) or (a == y and b == x):
            return True
    
    return False

def _match_text_sequence(pred: str, expected: Sequence[Any]) -> bool:
    """Check if predicted sequence matches expected sequence."""
    if not pred or not isinstance(pred, str):
        return False
    
    items = extract_elements(pred, str)
    if not items or len(items) != len(expected):
        return False
    
    expected_lower = [_casefold(item) for item in expected]
    return all(_casefold(pred_item) == exp for pred_item, exp in zip(items, expected_lower))

def tuple_eval_fn(pred: str, ground_truth: tuple) -> bool:
    """Evaluate tuple/sequence prediction."""
    return _match_text_sequence(pred, ground_truth)

def list_dir_eval_fn(pred: str, gt_list: List[tuple]) -> bool:
    """Evaluate numbered list of directions."""
    pairs = _parse_indexed_lines(pred)
    if not pairs:
        return False
    
    correct = 0
    for idx, pred_dir in pairs:
        if 0 <= idx < len(gt_list) and _match_text_sequence(pred_dir, gt_list[idx]):
            correct += 1
    
    return correct == len(gt_list)

def obj_seq_eval_fn(pred: str, gt_object_sequence: List[str]) -> bool:
    """Evaluate object sequence prediction."""
    return _match_text_sequence(pred, gt_object_sequence)

def deg_seq_eval_fn(pred: str, gt_degree_sequence: List[int]) -> bool:
    """Evaluate degree sequence prediction."""
    if not pred or not isinstance(pred, str):
        return False
    
    pred_degrees = extract_elements(pred, int, clean_pattern=r'[°degrees\s]')
    return bool(pred_degrees and pred_degrees == list(gt_degree_sequence))

def obj_presence_eval_fn(pred: Any, answer: List[str]) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate object presence detection with precision/recall metrics."""
    if not isinstance(pred, str):
        return False, {}
    
    pred_objects = extract_elements(pred, str)
    if not pred_objects:
        return False, {}
    
    gt_objects = {_casefold(obj) for obj in answer}
    pred_set = {_casefold(obj) for obj in pred_objects}
    
    correct_count = len(gt_objects & pred_set)
    total_gt = len(gt_objects)
    precision = correct_count / len(pred_set) if pred_set else 0.0
    recall = correct_count / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, {'precision': precision, 'recall': recall, 'f1': f1}


# ========== Common Evaluation Utilities ==========
def create_and_plot_room(seed: int = 0, plot: bool = True):
    """Generate a room, plot it, and return room, agent, and rng."""
    from ..utils.room_utils import RoomPlotter, RoomGenerator
    
    np_random = np.random.default_rng(seed)
    room, agent = RoomGenerator.generate_room(
        room_size=(30, 30),
        n_objects=10,
        np_random=np_random,
        room_name='room',
        level=2,
        main=6,
    )
    if plot:
        RoomPlotter.plot(room, agent, mode='img', save_path=f'room_{seed}.png')
    return room, agent, np_random


def manual_test_loop(task_name: str, task, eval_func: Callable[[str, Any, Any, Any], Tuple[float, Any]]):
    """Interactive loop for manual testing of evaluation tasks."""
    print(f"Question: {task.generate_question()}")
    print(f"Ground Truth Answer: {task.answer}")
    
    # Run the automatic correct answer check first as requested
    score, info = eval_func(task_name, task.answer, task.answer, task.choices)
    print(f"Correct Answer Evaluation: {score}, details: {info}")
    
    print("\n--- Manual Testing Mode ---")
    print(f"Enter your answer for '{task_name}' (or 'q' to quit).")
    print("You can try robust variations like different casing, spacing, etc.")
    
    while True:
        try:
            user_input = input("Answer > ").strip()
            if user_input.lower() == 'q':
                break
                
            # If user uses the "answer=xxx; evaluate(answer)" style or just inputs the answer
            # We treat the input as the answer candidates
            # We can also support literally `answer=...` syntax if needed, but direct input is simpler.
            # If the user typed "answer=foo", we strip "answer="
            if user_input.lower().startswith("answer="):
                user_input = user_input[7:].strip()
            
            score, info = eval_func(task_name, user_input, task.answer, task.choices)
            print(f"Evaluation: Score={score}, Details={info}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during evaluation: {e}")
    

def multi_choice_eval_fn(pred: str, answer: Union[List[str], str]) -> bool:
    """Evaluate multiple choice with multiple acceptable answers."""
    text = _require_text(pred)
    if text is None:
        return False
    
    answers = [answer] if isinstance(answer, str) else answer
    return text.lower() in {_casefold(ans) for ans in answers}


# ========== Task Evaluator Registry ==========
TaskEvaluator = Callable[[Any, Any, Optional[List[str]]], Tuple[bool, Dict[str, Any]]]

def _wrap_eval(
    func: Callable,
    *,
    pred_cast: Callable[[Any], Any] | None = str,
    answer_cast: Callable[[Any], Any] | None = None,
    pass_choices: bool = False,
) -> TaskEvaluator:
    """Wrap evaluation function to handle type casting."""
    def wrapped(pred: Any, answer: Any, choices: Optional[List[str]]) -> Tuple[bool, Dict[str, Any]]:
        pred_arg = pred_cast(pred) if pred_cast else pred
        ans_arg = answer_cast(answer) if answer_cast else answer
        
        if pass_choices:
            return func(pred_arg, ans_arg, choices)
        return func(pred_arg, ans_arg)
    
    return wrapped

TASK_EVALUATORS: Dict[str, TaskEvaluator] = {
    'RotEvaluationTask': _wrap_eval(_eval_token_sequence),
    'DirectionEvaluationTask': _wrap_eval(_eval_direction_text),
    'PovEvaluationTask': _wrap_eval(_eval_direction_text),
    'BackwardPovTextEvaluationTask': _wrap_eval(_eval_backward_pov, pred_cast=str, answer_cast=None),
    'BackwardPovVisionEvaluationTask': _wrap_eval(_eval_backward_pov_vision, pred_cast=str, answer_cast=None),
    'DirectionPov': _wrap_eval(_eval_direction_text),
    'AlloMappingEvaluationTask': lambda pred, answer, _choices: e2a_eval_fn(pred, answer),
    'Action2ViewEvaluationTask': _wrap_eval(_eval_forward_nav),
    'View2ActionTextEvaluationTask': _wrap_eval(lambda p, a: _eval_backward_nav(p, a, require_exact_pose=False), pred_cast=str, answer_cast=None),
    'View2ActionVisionEvaluationTask': _wrap_eval(lambda p, a: _eval_backward_nav(p, a, require_exact_pose=True), pred_cast=str, answer_cast=None),
    'View2ActionRevEvaluationTask': _wrap_eval(_eval_backward_nav_rev, pred_cast=str, answer_cast=None),
    'Location2ViewEvaluationTask': _wrap_eval(_eval_forward_nav),
    'View2LocationTextEvaluationTask': _wrap_eval(_eval_backward_loc, pred_cast=str, answer_cast=None),
    'View2LocationVisionEvaluationTask': _wrap_eval(_eval_backward_loc, pred_cast=str, answer_cast=None),
    'FalseBeliefDirectionPov': _wrap_eval(_eval_direction_text),
}

def evaluate_task_answer(
    task_type: str,
    pred: Any,
    answer: Any,
    choices: Optional[List[str]],
) -> Tuple[float, Dict[str, Any]]:
    """Main entry point for task evaluation."""
    evaluator = TASK_EVALUATORS.get(task_type)
    assert evaluator is not None, f"Unknown task evaluator: {task_type}"
    return evaluator(pred, answer, choices)


def parse_cogmap_response_content(content: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Parse content containing a cogmap JSON and an answer text."""
    if not content:
        return None, ""

    cogmap = None
    answer_text = content.strip()

    # Try parsing with tags
    cogmap_match = re.search(r"<cogmap>\s*(.*?)\s*</cogmap>", content, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL | re.IGNORECASE)

    if cogmap_match:
        json_str = cogmap_match.group(1).strip()
        # Extract potential JSON object (robust to wrapping text/markdown)
        match = re.search(r"(\{.*\})", json_str, re.DOTALL)
        if match:
            json_str = match.group(1)
        try:
            cogmap = json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # If explicit answer tag missing, remove cogmap block and use rest
        if not answer_match:
            answer_text = content.replace(cogmap_match.group(0), "", 1).strip()

    if answer_match:
        answer_text = answer_match.group(1).strip()
    
    # Fallback to old logic (markdown code block) if no tags found
    if not cogmap and not answer_match:
        code_block_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
        match = code_block_pattern.search(content)
        if match:
            try:
                cogmap = json.loads(match.group(1))
                answer_text = content.replace(match.group(0), "", 1).strip()
            except json.JSONDecodeError:
                pass

    return cogmap, answer_text


def evaluate_task_answer_with_cogmap(
    task_type: str,
    pred: Any,
    answer: Any,
    choices: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, Any], Optional[Dict[str, Any]]]:
    """Evaluate task answer after parsing out the cognitive map.
    
    Returns:
        (score, info_dict, cogmap_dict)
    """
    if not isinstance(pred, str):
        # Fallback if pred is not string
        score, info = evaluate_task_answer(task_type, pred, answer, choices)
        return score, info, None
        
    cogmap, answer_text = parse_cogmap_response_content(pred)
    
    # Evaluate the cleaned answer text
    score, info = evaluate_task_answer(task_type, answer_text, answer, choices)
    
    return score, info, cogmap


# ========== Exports ==========
__all__ = [
    'TaskEvaluator',
    'TASK_EVALUATORS',
    'evaluate_task_answer',
    'exp_evaluate_fn',
    'tuple_eval_fn',
    'list_dir_eval_fn',
    'obj_seq_eval_fn',
    'deg_seq_eval_fn',
    'obj_presence_eval_fn',
    'multi_choice_eval_fn',
    'e2a_eval_fn',
    'resolve_gate_orientation',
    'extract_elements',
    'parse_cogmap_response_content',
    'evaluate_task_answer_with_cogmap',
]
