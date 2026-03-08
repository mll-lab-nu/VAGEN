"""
Unexplored Areas Evaluation Module

Evaluates predictions of unexplored grid coordinates against ground truth unexplored regions.
Uses connectivity analysis to determine if predictions correctly identify distinct unexplored areas.
"""

from typing import List, Tuple, Set, Dict, Any, Optional
import numpy as np
import random
from collections import deque
import math

from .types import UnexploredMetrics


def compute_unexplored_regions(
    unexplored_positions: Set[Tuple[int, int]],
) -> List[Set[Tuple[int, int]]]:
    """Compute connected unexplored regions in the grid using BFS.
    
    Args:
        unexplored_positions: Set of (x, y) coordinates that are unexplored
        
    Returns:
        List of sets, each set containing (x, y) coordinates of a connected unexplored region
    """
    # Find connected components using BFS
    regions: List[Set[Tuple[int, int]]] = []
    visited: Set[Tuple[int, int]] = set()
    
    def bfs(start: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """BFS to find all connected unexplored cells from start."""
        region = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            x, y = queue.popleft()
            region.add((x, y))
            
            # Check 4-connected neighbors (can change to 8-connected if needed)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in unexplored_positions and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return region
    
    for cell in unexplored_positions:
        if cell not in visited:
            region = bfs(cell)
            if region:
                regions.append(region)
    
    return regions

def evaluate_unexplored_predictions(
    predicted_coords: List[Tuple[int, int]],
    correct_coords: List[Tuple[int, int]],
) -> UnexploredMetrics:
    """Evaluate predicted unexplored coordinates against ground truth.
    
    Overall score: F1 of precision/recall.
    
    Args:
        predicted_coords: List of (x, y) coordinates predicted as unexplored
        correct_coords: List of ground truth unexplored (x, y) coordinates
        
    Returns:
        UnexploredMetrics with precision, recall, and overall scores
    """
    assert correct_coords, "No correct coordinates provided"
    
    if not predicted_coords:
        # No predictions when there are unexplored regions
        return UnexploredMetrics(precision=0.0, recall=0.0, overall=0.0, valid=True)
    
    correct_set = set(correct_coords)
    predicted_set = set(predicted_coords)
    
    # Calculate correct and wrong predictions
    correct_predictions = len(predicted_set & correct_set)
    total_correct = len(correct_set)
    
    # Calculate precision and recall
    precision = correct_predictions / len(predicted_set) if predicted_set else 0.0
    recall = correct_predictions / total_correct if total_correct else 0.0
    
    # Overall score: F1
    overall = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return UnexploredMetrics(
        precision=precision,
        recall=recall,
        overall=overall,
        valid=True,
    )


def parse_unexplored_response(text: str) -> List[Tuple[int, int]]:
    """Parse predicted coordinates from an LLM response.

    The input is expected to follow the unexplored prompt output format, i.e. a JSON object:
    {"unexplored": "(5, 3); (2, 1); (10, 2)"}.
    This parser is intentionally more permissive to handle common formatting variations.

    Supported forms include:
    - JSON: {"unexplored": "(1,2); (3,4)"} or {"unexplored": [[1,2],[3,4]]}
    - Plain text: "(1,2), (3,4)" / "[1, 2]; [3, 4]" / mixed separators

    Returns: list[(x, y)] (possibly empty).
    """
    import json
    import re

    def _extract_json_candidates(s: str) -> List[str]:
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
        candidates = list(fenced) if fenced else []
        if candidates:
            return candidates
        # Fallback: scan for outermost balanced braces
        stack, start = [], None
        for i, ch in enumerate(s):
            if ch == '{':
                if not stack:
                    start = i
                stack.append(ch)
            elif ch == '}' and stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(s[start:i + 1])
                    start = None
        return candidates

    def _parse_pairs_from_string(s: str) -> List[Tuple[int, int]]:
        # (1,2) or [1,2] or {1,2} with optional spaces; allow negative ints
        pair_pat = r"[\(\[\{]\s*(-?\d+)\s*,\s*(-?\d+)\s*[\)\]\}]"
        out: List[Tuple[int, int]] = []
        for x_str, y_str in re.findall(pair_pat, s):
            try:
                out.append((int(x_str), int(y_str)))
            except Exception:
                continue
        # Also allow bare "1,2" pairs if nothing else matched
        if not out:
            bare_pat = r"(-?\d+)\s*,\s*(-?\d+)"
            for x_str, y_str in re.findall(bare_pat, s):
                try:
                    out.append((int(x_str), int(y_str)))
                except Exception:
                    continue
        # De-dup while keeping order
        seen = set()
        uniq: List[Tuple[int, int]] = []
        for p in out:
            if p in seen:
                continue
            seen.add(p)
            uniq.append(p)
        return uniq

    def _parse_value(v) -> List[Tuple[int, int]]:
        if v is None:
            return []
        if isinstance(v, str):
            return _parse_pairs_from_string(v)
        if isinstance(v, (list, tuple)):
            coords: List[Tuple[int, int]] = []
            for it in v:
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    try:
                        coords.append((int(it[0]), int(it[1])))
                    except Exception:
                        continue
                elif isinstance(it, str):
                    coords.extend(_parse_pairs_from_string(it))
            # De-dup while keeping order
            seen = set()
            uniq: List[Tuple[int, int]] = []
            for p in coords:
                if p in seen:
                    continue
                seen.add(p)
                uniq.append(p)
            return uniq
        if isinstance(v, dict):
            # Sometimes nested, e.g. {"coords": "..."}; best-effort: search within JSON text
            return _parse_pairs_from_string(json.dumps(v, ensure_ascii=False))
        return []

    if not isinstance(text, str):
        return []
    raw = text.strip()

    # Try JSON first
    json_dict = None
    for cand in _extract_json_candidates(raw):
        try:
            json_dict = json.loads(cand)
            break
        except json.JSONDecodeError:
            continue
    if isinstance(json_dict, dict):
        return _parse_value(json_dict.get("unexplored"))

    # Fallback: plain text parsing
    return _parse_value(raw)


def parse_fog_probe_response(text: str, all_candidate_coords: Optional[List[Tuple[int, int]]]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Parse fog-probe responses that select labeled candidates (A, B, ...).

    Returns a tuple (labels, coords) where `labels` is an ordered list of
    uppercase single-character labels ("A", "B", ...) and `coords` is the
    mapped coordinates from `all_candidate_coords` corresponding to those
    labels. If a label is out of range or cannot be mapped, it is ignored.
    """
    import json
    import re

    def _extract_json_candidates(s: str) -> List[str]:
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
        candidates = list(fenced) if fenced else []
        if candidates:
            return candidates
        # Fallback: scan for outermost balanced braces
        stack, start = [], None
        for i, ch in enumerate(s):
            if ch == '{':
                if not stack:
                    start = i
                stack.append(ch)
            elif ch == '}' and stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(s[start:i + 1])
                    start = None
        return candidates

    def _parse_labels_from_string(s: str) -> List[str]:
        labels = re.findall(r"\b([A-Z])\b", s)
        if labels:
            seen = set(); out = []
            for L in labels:
                if L in seen: continue
                seen.add(L); out.append(L)
            return out
        # fallback: look for quoted letters in lists e.g. "A","B"
        q = re.findall(r"['\"]([A-Za-z])[ '\"]?[,;\]]", s)
        if q:
            out = []
            seen = set()
            for L in q:
                U = L.upper()
                if U in seen: continue
                seen.add(U); out.append(U)
            return out
        return []

    labels: List[str] = []
    if isinstance(text, str):
        raw = text.strip()
        json_dict = None
        for cand in _extract_json_candidates(raw):
            try:
                json_dict = json.loads(cand)
                break
            except json.JSONDecodeError:
                continue
        if isinstance(json_dict, dict):
            v = json_dict.get('unexplored') or json_dict.get('points') or json_dict.get('candidates')
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, str) and len(it.strip()) == 1:
                        labels.append(it.strip().upper())
                    elif isinstance(it, (list, tuple)) and it:
                        try:
                            labs = [str(x).strip().upper() for x in it if isinstance(x, str) and len(str(x).strip()) == 1]
                            labels.extend(labs)
                        except Exception:
                            continue
            elif isinstance(v, str):
                labels = _parse_labels_from_string(v)
            else:
                labels = _parse_labels_from_string(json.dumps(json_dict))
        else:
            labels = _parse_labels_from_string(raw)

    # Map labels to coordinates using A->0, B->1, ...
    pred_coords: List[Tuple[int, int]] = []
    if isinstance(all_candidate_coords, list) and labels:
        for L in labels:
            idx = ord(L.upper()) - ord('A')
            if 0 <= idx < len(all_candidate_coords):
                c = all_candidate_coords[idx]
                pred_coords.append((int(c[0]), int(c[1])))

    return labels, pred_coords

def distances_to_explored(
    points: List[Tuple[int, int]],
    explored_points: Set[Tuple[int, int]],
) -> List[float]:
    """Distance to explored region (closest observed point) for each point.

    Uses Euclidean distance on grid coordinates.
    """
    if not points:
        return []
    if not explored_points:
        return [float("inf")] * len(points)
    explored = list(explored_points)
    out: List[float] = []
    for x, y in points:
        best_d2: Optional[int] = None
        for ex, ey in explored:
            d2 = int(x - ex) ** 2 + int(y - ey) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
        out.append(float(math.sqrt(best_d2)) if best_d2 is not None else float("inf"))
    return out


def aggregate_unexplored_metrics(
    cog_logs: List[Dict[str, Any]],
    exp_logs: List[Dict[str, Any]],
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Aggregate unexplored metrics across turns.

    Returns:
        - per-turn F1 list (None for invalid turns)
        - per-turn Precision list
        - per-turn Recall list
        - micro-F1 over points across valid turns
        - micro-Precision over points across valid turns
        - micro-Recall over points across valid turns
        - correlation between distance-to-explored and hit (1=selected correct point)
    """
    unexp_f1_per_turn: List[Optional[float]] = []
    unexp_p_per_turn: List[Optional[float]] = []
    unexp_r_per_turn: List[Optional[float]] = []
    tp = fp = fn = 0
    dist_vals: List[float] = []
    hit_vals: List[float] = []

    for d, exp in zip(cog_logs, exp_logs):
        # un = (d.get('unexplored') or {})
        un = (d.get('fog_probe') or {})
        um = UnexploredMetrics.from_dict((un.get('metrics') or {}))
        unexp_f1_per_turn.append(float(um.overall) if um.valid else None)
        unexp_p_per_turn.append(float(um.precision) if um.valid else None)
        unexp_r_per_turn.append(float(um.recall) if um.valid else None)

        if not um.valid:
            continue
        if not (un.get('all_candidate_points') and un.get('correct_points')):
            continue
        try:
            pred = {tuple(map(int, p)) for p in (un.get('pred_points') or []) if isinstance(p, (list, tuple)) and len(p) == 2}
            corr = {tuple(map(int, p)) for p in (un.get('correct_points') or []) if isinstance(p, (list, tuple)) and len(p) == 2}
        except Exception:
            continue

        tp += len(pred & corr)
        fp += len(pred - corr)
        fn += len(corr - pred)

        cand = exp.get('all_candidate_coords') or []
        dists = exp.get('all_candidate_dists') or []
        if isinstance(cand, list) and isinstance(dists, list) and len(cand) == len(dists) and cand:
            dist_by_pt: Dict[Tuple[int, int], float] = {}
            for pt, di in zip(cand, dists):
                if isinstance(pt, (list, tuple)) and len(pt) == 2 and isinstance(di, (int, float)):
                    dist_by_pt[(int(pt[0]), int(pt[1]))] = float(di)
            for pt in corr:
                di = dist_by_pt.get(pt)
                if di is None:
                    continue
                dist_vals.append(float(di))
                hit_vals.append(1.0 if pt in pred else 0.0)

    unexp_f1_avg = None
    unexp_p_avg = None
    unexp_r_avg = None
    if (tp + fp + fn) > 0:
        unexp_p = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        unexp_r = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        unexp_f1_avg = (2.0 * unexp_p * unexp_r / (unexp_p + unexp_r)) if (unexp_p + unexp_r) > 0 else 0.0
        unexp_p_avg = float(unexp_p)
        unexp_r_avg = float(unexp_r)

    unexp_distance_hit_corr = None
    if len(dist_vals) >= 2 and len(hit_vals) == len(dist_vals):
        try:
            D = np.asarray(dist_vals, dtype=float)
            H = np.asarray(hit_vals, dtype=float)
            if float(np.std(D)) > 0.0 and float(np.std(H)) > 0.0:
                unexp_distance_hit_corr = float(np.corrcoef(D, H)[0, 1])
        except Exception:
            unexp_distance_hit_corr = None

    return unexp_f1_per_turn, unexp_p_per_turn, unexp_r_per_turn, unexp_f1_avg, unexp_p_avg, unexp_r_avg, unexp_distance_hit_corr


__all__ = [
    'compute_unexplored_regions',
    'evaluate_unexplored_predictions',
    'parse_unexplored_response',
    'parse_fog_probe_response',
    'distances_to_explored',
    'aggregate_unexplored_metrics',
    'UnexploredMetrics',
]