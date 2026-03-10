from typing import Dict, Any, List, Optional
import numpy as np
from scipy.stats import pearsonr


def _to_float_or_none(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)) and not np.isnan(v):
        return float(v)
    return None


def compute_correlation_metrics(env_data_list: List[Dict[str, Any]], exp_type: str = 'active') -> Dict[str, Any]:
    """
    Compute correlations between cognitive map metrics and evaluation metrics, information gain metrics.
    """
    assert isinstance(env_data_list, list) and len(env_data_list) > 0, "env_data_list must be a non-empty list"
    if exp_type == 'passive':
        return {}

    # Collect valid samples
    samples = []
    task_names = set()
    
    for s in env_data_list:
        metrics = s.get('metrics') or {}
        
        # Get cogmap score (last_global_vs_gt_full)
        try:
            cog_score = metrics['cogmap']['exploration']['correctness']['last_global_vs_gt_full']['overall']
            if not isinstance(cog_score, (int, float)) or np.isnan(cog_score):
                continue
        except (KeyError, TypeError):
            continue

        # Get evaluation metrics
        eval_m = metrics.get('evaluation') or {}
        per_task = (eval_m.get('per_task') or {})
        task_names.update(per_task.keys())
        
        # Get info gain
        infogain = (metrics.get('exploration') or {}).get('final_information_gain')

        samples.append({
            'cog_score': float(cog_score),
            'eval_m': eval_m,
            'infogain': float(infogain) if isinstance(infogain, (int, float)) and not np.isnan(infogain) else None
        })

    if not samples:
        return {'n_samples': 0}

    # Arrays for correlation
    cog_scores = [s['cog_score'] for s in samples]
    infogains = [s['infogain'] for s in samples]
    
    # Calculate correlations
    cogmap_acc_correlations = {}
    
    # 1. Overall accuracy
    avg_accs = [_to_float_or_none((s['eval_m'].get('overall') or {}).get('avg_accuracy')) for s in samples]
    cogmap_acc_correlations['avg_accuracy'] = calculate_pearson_correlation(cog_scores, avg_accs)
    
    # 2. Per-task accuracy
    for task in task_names:
        task_accs = []
        for s in samples:
            acc = (s['eval_m'].get('per_task') or {}).get(task, {}).get('avg_accuracy')
            task_accs.append(_to_float_or_none(acc))

        cogmap_acc_correlations[task] = calculate_pearson_correlation(cog_scores, task_accs)

    return {
        'cogmap_acc_correlations': cogmap_acc_correlations,
        'cogmap_infogain_correlation': calculate_pearson_correlation(cog_scores, infogains),
        'last_global_vs_gt_fulls': cog_scores,
        'last_infogains': infogains,
        'avg_acc_metrics': avg_accs,
        'n_samples': len(samples)
    }


def calculate_pearson_correlation(x: List[float], y: List[float]) -> Dict[str, Any]:
    try:
        # Pairwise filter (ignore None/NaN), allow length mismatch by zipping
        valid_pairs = []
        for xi, yi in zip(x or [], y or []):
            if xi is None or yi is None:
                continue
            if isinstance(xi, (int, float)) and isinstance(yi, (int, float)) and not np.isnan(xi) and not np.isnan(yi):
                valid_pairs.append((float(xi), float(yi)))
        if len(valid_pairs) < 2:
            return {
                'pearson_r': None,
                'p_value': None,
                'significant': False,
                'n_samples': len(valid_pairs)
            }
        x_valid, y_valid = zip(*valid_pairs)
        corr_coef, p_value = pearsonr(x_valid, y_valid)
        return {
            'pearson_r': float(corr_coef),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'n_samples': len(valid_pairs)
        }
    except Exception as e:
        return {
            'pearson_r': None,
            'p_value': None,
            'significant': False,
            'n_samples': 0,
            'error': str(e)
        }

