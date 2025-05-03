# vagen/inference/utils/metrics.py

import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Union

def calculate_aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Calculate aggregate metrics from results.
    
    Args:
        results: List of result dictionaries from recording_to_log
        
    Returns:
        Dictionary of aggregate metrics
    """
    all_metrics = defaultdict(list)
    config_metrics = defaultdict(lambda: defaultdict(list))
    
    # Helper function to convert NumPy types to Python native types
    def convert_to_native(value):
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            return value
    
    for result in results:
        config_id = result.get("config_id", "unknown")
        for k, v in result["metrics"].items():
            native_v = convert_to_native(v)
            all_metrics[k].append(native_v)
            config_metrics[config_id][k].append(native_v)
    
    metrics = {}
    
    # Overall metrics
    for key, values in all_metrics.items():
        if not values or not isinstance(values[0], (int, float)):
            continue
            
        metrics[f"mean_{key}"] = float(np.mean(values))
        if len(values) > 1:
            metrics[f"std_{key}"] = float(np.std(values))
        metrics[f"min_{key}"] = float(np.min(values))
        metrics[f"max_{key}"] = float(np.max(values))
        
        # For boolean-like metrics
        if key in ["done", "success", "action_is_valid", "action_is_effective"] or \
           (all(isinstance(v, (int, float)) for v in values) and 
            all(0 <= v <= 1 for v in values)):
            metrics[f"percent_{key}"] = float(np.mean(values) * 100)
    
    # Metrics by config_id
    for config_id, config_data in config_metrics.items():
        for key, values in config_data.items():
            if not values or not isinstance(values[0], (int, float)):
                continue
                
            metrics[f"{config_id}/{key}"] = float(np.mean(values))
    
    return metrics

def log_rst_to_metrics_dict(rst: List[Dict], mode: str = 'eval') -> Dict[str, float]:
    """
    Convert results to metrics dictionary format compatible with PPO trainer.
    
    Args:
        rst: Results list from recording_to_log
        mode: Metrics prefix (eval/train/val)
        
    Returns:
        Metrics dictionary with proper prefixes
    """
    metrics_dict = {}
    metrics_by_config_id = defaultdict(lambda: defaultdict(list))
    
    # Helper function to convert NumPy types to Python native types
    def convert_to_native(value):
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            return value
    
    # Group metrics by config_id
    for item in rst:
        config_id = item["config_id"]
        for k, v in item["metrics"].items():
            native_v = convert_to_native(v)
            metrics_by_config_id[config_id][k].append(native_v)
    
    # Compute averages for each metric by config_id
    for config_id, metrics in metrics_by_config_id.items():
        for k, values in metrics.items():
            if values:
                metrics_dict[f'{mode}/{k}/{config_id}'] = float(np.mean(values))
    
    # Aggregate metrics across all configs
    all_metrics = defaultdict(list)
    for config_id, metrics in metrics_by_config_id.items():
        for k, values in metrics.items():
            all_metrics[k].extend(values)
    
    for k, values in all_metrics.items():
        if values:
            metrics_dict[f'{mode}/{k}/all'] = float(np.mean(values))
    
    return metrics_dict