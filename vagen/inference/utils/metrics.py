"""
Utility functions for metrics calculation and result processing.
"""

import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from .logging import maybe_log_val_generations_to_wandb

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def log_rst_to_metrics_dict(rst: List[Dict], mode: str = 'eval') -> Dict[str, float]:
    """
    Helper method to convert results to metrics dictionary format
    compatible with PPO trainer.

    Args:
        rst: Results list from recording_to_log
        mode: Metrics prefix (eval/train/val)

    Returns:
        Aggregated metrics dictionary
    """
    metrics_dict = {}
    metrics_by_config_id = defaultdict(lambda: defaultdict(list))

    # Group metrics by config_id
    for item in rst:
        config_id = item["config_id"]
        for k, v in item["metrics"].items():
            metrics_by_config_id[config_id][k].append(v)

    # Compute averages for each metric by config_id
    for config_id, metrics in metrics_by_config_id.items():
        for k, values in metrics.items():
            metrics_dict[f'{mode}/{k}/{config_id}'] = sum(values) / len(values)

    # Also include aggregated metrics across all configs
    all_metrics = defaultdict(list)
    for config_id, metrics in metrics_by_config_id.items():
        for k, values in metrics.items():
            all_metrics[k].extend(values)

    for k, values in all_metrics.items():
        metrics_dict[f'{mode}/{k}/all'] = sum(values) / len(values)

    return metrics_dict

def calculate_aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Calculate aggregate metrics from results.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary of aggregate metrics
    """
    all_metrics = defaultdict(list)

    for result in results:
        for k, v in result["metrics"].items():
            all_metrics[k].append(v)

    metrics = {}
    for key, values in all_metrics.items():
        # Skip non-numeric values
        if not values or not isinstance(values[0], (int, float)):
            continue

        # Compute statistics
        metrics[f"mean_{key}"] = np.mean(values)
        if len(values) > 1:
            metrics[f"std_{key}"] = np.std(values)
        metrics[f"min_{key}"] = np.min(values)
        metrics[f"max_{key}"] = np.max(values)

        # For boolean metrics, also report percentage
        if key in ["done", "success", "completed"] or (max(values) <= 1.0 and min(values) >= 0.0 and all(v == int(v) for v in values)):
            metrics[f"percent_{key}"] = sum(1 for v in values if v) / len(values) * 100

    return metrics

def run_validation(
    inference_service,
    env_configs: List[Dict],
    max_steps: int,
    global_steps: int,
    output_dir: str,
    mode: str = "eval",
    use_wandb: bool = True
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Run validation, logging original env metrics to wandb.

    Args:
        inference_service: Inference rollout service
        env_configs: List of environment configurations
        max_steps: Maximum steps per environment
        global_steps: Current global training step
        output_dir: Output directory for results
        mode: Metrics prefix (eval/train/val)
        use_wandb: Whether to log to wandb

    Returns:
        Dictionary of validation metrics and list of results
    """

    print(f"[DEBUG] validation at global step {global_steps} begins")

    # Reset environments
    inference_service.reset(env_configs)

    # Run inference
    inference_service.run(max_steps=max_steps)

    # Get validation results
    results = inference_service.recording_to_log()

    # Basic metrics for terminal output
    total_envs = len(results)
    completed_envs = sum(1 for r in results if r['metrics'].get('done', False))
    success_envs = sum(1 for r in results if r['metrics'].get('success', False))
    mean_score = sum(r['metrics'].get('score', 0) for r in results) / total_envs if total_envs > 0 else 0
    
    # For wandb, we want the original metrics
    primary_metrics = {
        f"{mode}/success": success_envs / total_envs if total_envs > 0 else 0,
        f"{mode}/step": sum(r['metrics'].get('step', 0) for r in results) / total_envs if total_envs > 0 else 0,
        f"{mode}/score": mean_score,
        f"{mode}/done": completed_envs / total_envs if total_envs > 0 else 0,
        f"{mode}/action_is_valid": sum(r['metrics'].get('action_is_valid', 0) for r in results) / total_envs if total_envs > 0 else 0,
        f"{mode}/action_is_effective": sum(r['metrics'].get('action_is_effective', 0) for r in results) / total_envs if total_envs > 0 else 0
    }

    # Save results
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump({
            "results": results
        }, f, cls=NumpyEncoder, indent=2)

    # Log to wandb if enabled
    if use_wandb:
        try:
            import wandb
            
            # Log the primary metrics directly
            wandb.log(primary_metrics, step=global_steps)
            
            # Only try the original function for logging examples
            # This should be handled separately, so we don't attempt our own implementation here
            try:
                val_generations_to_log = inference_service.config.get("val_generations_to_log_to_wandb", 5)
                maybe_log_val_generations_to_wandb(results, val_generations_to_log, global_steps)
            except Exception as e:
                print(f"Warning: maybe_log_val_generations_to_wandb failed: {e}")
                print("Skipping example table logging, but metrics were logged successfully")
            
        except Exception as e:
            print(f"Error logging to wandb: {e}")

    # Print summary
    print("\n===== Inference Results =====")
    print(f"Total environments: {total_envs}")
    print(f"Completed environments: {completed_envs} ({completed_envs/total_envs*100 if total_envs>0 else 0:.1f}%)")
    print(f"Successful environments: {success_envs} ({success_envs/total_envs*100 if total_envs>0 else 0:.1f}%)")
    print(f"Mean score: {mean_score:.4f}")
    print(f"Results saved to {results_file}")

    print(f"[DEBUG] validation at global step {global_steps} ends")
    
    return primary_metrics, results