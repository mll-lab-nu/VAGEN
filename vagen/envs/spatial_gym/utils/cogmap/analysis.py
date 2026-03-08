from typing import List, Dict, Any, Callable, Optional
from .types import MapCogMetrics


def _avg(values: List[float]) -> float:
    v = [x for x in values if isinstance(x, (int, float))]
    return sum(v) / len(v) if v else 0.0

def avg_float_list_skip_none(values: List[Optional[float]]) -> Optional[float]:
    v = [float(x) for x in (values or []) if isinstance(x, (int, float))]
    return sum(v) / len(v) if v else None

def _avg_metrics(keys: List[str], metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {k: 0.0 for k in keys}
    return {k: _avg([float(m.get(k, 0.0)) for m in metrics_list if isinstance(m, dict)]) for k in keys}


def _avg_list_of_lists(list_of_lists: List[List[float]]) -> List[Optional[float]]:
    """Average lists element-wise, returning None for positions with no valid data."""
    if not list_of_lists:
        return []
    max_len = max(len(lst) for lst in list_of_lists)
    out = []
    for i in range(max_len):
        vals = [lst[i] for lst in list_of_lists if i < len(lst) and isinstance(lst[i], (int, float))]
        out.append(_avg(vals) if vals else None)
    return out


def aggregate_per_sample_then_group(samples: List[Any], per_sample_fn: Callable[[Any], Dict[str, float]]) -> Dict[str, float]:
    metrics_per_sample = []
    for s in samples:
        m = per_sample_fn(s)
        if isinstance(m, dict):
            metrics_per_sample.append(m)
    # keys come from first dict or default map keys
    keys = list(metrics_per_sample[0].keys()) if metrics_per_sample else ["dir", "facing", "pos", "overall"]
    return _avg_metrics(keys, metrics_per_sample)


def aggregate_lists_per_turn(samples: List[Any], per_sample_list_fn: Callable[[Any], List[float]]) -> List[float]:
    lists = []
    for s in samples:
        lst = per_sample_list_fn(s)
        if isinstance(lst, list):
            lists.append(lst)
    return _avg_list_of_lists(lists)


def calculate_cogmap_per_turn(env_data_list: List[Dict[str, Any]], mode: str = "update") -> Dict[str, List[float]]:
    """Average global cognitive map metrics for each turn across samples.

    Returns dict with keys 'dir', 'facing', 'pos', 'overall', each a list over turns.
    """
    from collections import defaultdict

    turn_to_metrics: Dict[int, List[MapCogMetrics]] = defaultdict(list)
    for env_data in env_data_list:
        for turn_idx, turn_log in enumerate(env_data.get('env_turn_logs', [])):
            cogmap_agg = turn_log.get('cogmap_log') or {}
            level_data = cogmap_agg.get('global') or {}
            if not isinstance(level_data, dict):
                continue
            metrics_data = level_data.get('metrics_full' if mode == 'full' else 'metrics') or {}
            if not isinstance(metrics_data, dict) or not metrics_data:
                continue
            m = MapCogMetrics.from_dict(metrics_data)
            if m.valid:
                turn_to_metrics[turn_idx].append(m)

    max_turn = max(turn_to_metrics.keys()) if turn_to_metrics else -1
    per_turn_avg = [MapCogMetrics.average(turn_to_metrics[i]) if i in turn_to_metrics else MapCogMetrics.invalid() for i in range(max_turn + 1)]

    return {
        'dir': [float(m.dir) if m.valid else None for m in per_turn_avg],
        'facing': [float(m.facing) if m.valid else None for m in per_turn_avg],
        'pos': [float(m.pos) if m.valid else None for m in per_turn_avg],
        'overall': [float(m.overall) if m.valid else None for m in per_turn_avg],
    }


def _avg_map_over_turns(env_data: Dict[str, Any], section: str, field: str) -> MapCogMetrics:
    mats: List[MapCogMetrics] = []
    for t in env_data.get('env_turn_logs', []):
        d = (t.get('cogmap_log') or {}).get(section, {}).get(field, {})
        m = MapCogMetrics.from_dict(d)
        if m.valid:
            mats.append(m)
    return MapCogMetrics.average(mats) if mats else MapCogMetrics.invalid()


# ---- Generic reusable helpers ----
def avg_nested_dicts(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Average a list of similarly-shaped nested dicts.
    - Numbers averaged arithmetically
    - Dicts averaged recursively
    - Lists of numbers averaged element-wise (length padded with 0.0)
    - Otherwise, take first non-None
    """
    if not dicts:
        return {}
    out: Dict[str, Any] = {}
    keys = set().union(*[d.keys() for d in dicts])
    for k in keys:
        vals = [d.get(k) for d in dicts]
        nums = [v for v in vals if isinstance(v, (int, float))]
        if nums:
            out[k] = float(sum(nums) / len(nums))
            continue
        subs = [v for v in vals if isinstance(v, dict)]
        if subs:
            out[k] = avg_nested_dicts(subs)
            continue
        lists = [v for v in vals if isinstance(v, list)]
        if lists:
            out[k] = _avg_list_of_lists(lists)
            continue
        out[k] = next((v for v in vals if v is not None), None)
    return out


def avg_lists(list_of_lists: List[List[float]]) -> List[float]:
    """Average lists of floats element-wise (None/invalid ignored, pads with 0.0)."""
    return _avg_list_of_lists(list_of_lists)


def compute_error_aggregates(env_data_list: List[Dict[str, Any]],) -> Dict[str, Any]:
    return {
        'local_vs_gt_local_avg': aggregate_per_sample_then_group(env_data_list, compute_error_per_sample_local),
        'global_vs_gt_global_avg': aggregate_per_sample_then_group(env_data_list, compute_error_per_sample_global),
    }


def compute_error_per_sample_local(env_data: Dict[str, Any]) -> Dict[str, float]:
    m = _avg_map_over_turns(env_data, section='local', field='metrics')
    return m.to_dict()


def compute_error_per_sample_global(env_data: Dict[str, Any]) -> Dict[str, float]:
    m = _avg_map_over_turns(env_data, section='global', field='metrics')
    return m.to_dict()


def get_last_exploration_cogmap(env_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the last exploration turn's cogmap_log if present."""
    for t in reversed(env_data.get('env_turn_logs', [])):
        if t.get('cogmap_log'):
            return t.get('cogmap_log')
    return None


__all__ = [
    "aggregate_per_sample_then_group",
    "aggregate_lists_per_turn",
    "calculate_cogmap_per_turn",
    "compute_error_aggregates",
    "avg_nested_dicts",
    "avg_lists",
    "compute_error_per_sample_local",
    "compute_error_per_sample_global",
    "avg_float_list_skip_none",
]
