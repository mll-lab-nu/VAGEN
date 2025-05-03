import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Union


def create_metric_series(
    results: List[Dict], 
    metric_name: str
) -> List[Dict[str, Any]]:
    """
    Create a series of (example_idx, value) pairs for a given metric.
    
    Args:
        results: List of result dictionaries
        metric_name: Name of the metric to extract
        
    Returns:
        List of dictionaries with example_idx and value
    """
    series = []
    
    for idx, result in enumerate(results):
        if metric_name in result['metrics']:
            value = result['metrics'][metric_name]
            # Convert numpy types to Python native types
            if hasattr(value, 'item'):
                value = value.item()
            
            # Only include numeric values
            if isinstance(value, (int, float)):
                series.append({
                    'example_idx': idx,
                    'value': float(value)
                })
    
    return series

def create_summary_metrics(results: List[Dict]) -> Dict[str, Any]:
    """
    Create summary metrics from results.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary of summary metrics
    """
    summary = {}
    
    # Basic counts
    summary['total_examples'] = len(results)
    summary['num_successful'] = sum(1 for r in results if r['metrics'].get('success', 0) > 0)
    summary['num_done'] = sum(1 for r in results if r['metrics'].get('done', 0) > 0)
    
    # Success and completion rates
    if len(results) > 0:
        summary['success_rate'] = summary['num_successful'] / len(results)
        summary['completion_rate'] = summary['num_done'] / len(results)
    
    # Collect all numeric metrics
    metric_values = defaultdict(list)
    
    for result in results:
        for metric_name, value in result['metrics'].items():
            # Convert numpy types to Python native types
            if hasattr(value, 'item'):
                value = value.item()
                
            if isinstance(value, (int, float)):
                metric_values[metric_name].append(float(value))
    
    # Calculate statistics for each metric
    for metric_name, values in metric_values.items():
        if len(values) > 0:
            summary[f'{metric_name}_mean'] = float(np.mean(values))
            if len(values) > 1:
                summary[f'{metric_name}_std'] = float(np.std(values))
            summary[f'{metric_name}_min'] = float(np.min(values))
            summary[f'{metric_name}_max'] = float(np.max(values))
    
    return summary

def organize_metrics_for_wandb(results: List[Dict]) -> Dict[str, Any]:
    """
    Organize metrics into sections for wandb logging.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with eval and summary sections
    """
    wandb_metrics = {}
    
    # Get all unique metric names
    all_metric_names = set()
    for result in results:
        all_metric_names.update(result['metrics'].keys())
    
    # Process each metric
    eval_section = {}
    for metric_name in all_metric_names:
        # Extract series data
        series = create_metric_series(results, metric_name)
        if series:
            for point in series:
                eval_key = f"eval/{metric_name}/example_{point['example_idx']}"
                eval_section[eval_key] = point['value']
    
    # Create summary section
    summary = create_summary_metrics(results)
    summary_section = {f"summary/{k}": v for k, v in summary.items()}
    
    # Combine sections
    wandb_metrics.update(eval_section)
    wandb_metrics.update(summary_section)
    
    return wandb_metrics