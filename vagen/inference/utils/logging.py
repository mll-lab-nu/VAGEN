import wandb
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class ValidationTableManager:
    """Manages the validation table for wandb logging."""
    
    def __init__(self):
        self.validation_table = None
    
    def log_generations_to_wandb(
        self, 
        log_rst: List[Dict[str, Any]], 
        generations_to_log: int, 
        global_steps: int = 0  # Default to 0 for inference
    ) -> None:
        """Log a table of validation samples."""
        if generations_to_log == 0:
            return
            
        if wandb.run is None:
            logger.warning('`val_generations_to_log_to_wandb` is set, but wandb is not initialized')
            return
        
        # Extract data from results
        inputs = []
        outputs = []
        scores = []
        images = []
        
        for item in log_rst:
            inputs.append(item['config_id'])
            outputs.append(item['output_str'])
            scores.append(item['metrics']['score'])
            images.append(item.get('image_data', None))
        
        # Check if we have images
        has_images = any(img_list for img_list in images if img_list)
        
        # Find maximum number of images in any sample
        if has_images:
            max_images_per_sample = max(
                len(img_list) if img_list else 0
                for img_list in images
            )
        else:
            max_images_per_sample = 0
        
        # Create samples
        if has_images:
            samples = list(zip(inputs, outputs, scores, images))
        else:
            samples = list(zip(inputs, outputs, scores))
        
        # Sort and shuffle for consistency
        samples.sort(key=lambda x: x[0])  # Sort by input text
        rng = np.random.RandomState(42)  # Use a fixed seed for reproducibility
        rng.shuffle(samples)
        
        # Take first N samples
        samples = samples[:generations_to_log]
        
        # Create columns for the table
        if has_images:
            columns = ["input", "output", "score"] + [f"image_{i+1}" for i in range(max_images_per_sample)]
        else:
            columns = ["input", "output", "score"]
        
        # Create table
        table = wandb.Table(columns=columns)
        
        # Add each sample as a separate row
        for sample in samples:
            if has_images:
                input_text, output_text, score, sample_images = sample
                
                # Convert images to wandb.Image
                wandb_images = []
                if sample_images:
                    for img in sample_images:
                        if img is not None:
                            if not isinstance(img, wandb.Image):
                                img = wandb.Image(img)
                            wandb_images.append(img)
                
                # Pad with None if fewer images than max
                while len(wandb_images) < max_images_per_sample:
                    wandb_images.append(None)
                
                # Add row
                table.add_data(input_text, output_text, score, *wandb_images)
            else:
                input_text, output_text, score = sample
                table.add_data(input_text, output_text, score)
        
        # Log the table
        wandb.log({"val/generations": table})


# Global instance for maintaining table state across calls
validation_table_manager = ValidationTableManager()


def maybe_log_val_generations_to_wandb(
    results: List[Dict[str, Any]], 
    generations_to_log: int, 
    global_step: int = 0  # Default to 0 for inference
) -> None:
    """
    Log validation generation examples to wandb as a table.
    Uses the global ValidationTableManager to maintain state.
    """
    validation_table_manager.log_generations_to_wandb(
        log_rst=results,
        generations_to_log=generations_to_log,
        global_steps=global_step
    )


def aggregate_metrics_for_summary(
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create a summary dictionary with mean, std, min, max for all numeric metrics.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary of summary metrics
    """
    summary = {}
    
    # Basic counts
    summary['summary/total_examples'] = len(results)
    summary['summary/num_successful'] = sum(1 for r in results if r['metrics'].get('success', 0) > 0)
    summary['summary/num_done'] = sum(1 for r in results if r['metrics'].get('done', 0) > 0)
    
    # Calculate rates
    if len(results) > 0:
        summary['summary/success_rate'] = summary['summary/num_successful'] / len(results)
        summary['summary/completion_rate'] = summary['summary/num_done'] / len(results)
    
    # Collect all metrics across examples
    metrics_values = defaultdict(list)
    for result in results:
        for metric_name, value in result['metrics'].items():
            if isinstance(value, (int, float)):
                metrics_values[metric_name].append(float(value))
    
    # Calculate statistics for each metric
    for metric_name, values in metrics_values.items():
        if values:
            summary[f'summary/{metric_name}_mean'] = float(np.mean(values))
            if len(values) > 1:
                summary[f'summary/{metric_name}_std'] = float(np.std(values))
            summary[f'summary/{metric_name}_min'] = float(np.min(values))
            summary[f'summary/{metric_name}_max'] = float(np.max(values))
    
    return summary


def log_metrics_by_example(
    results: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Organize metrics by example for line plots.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary mapping metric names to lists of (example_idx, value) pairs
    """
    # Get all unique metric names
    all_metric_names = set()
    for result in results:
        all_metric_names.update(result['metrics'].keys())
    
    # Collect values for each metric across all examples
    metrics_data = {}
    for metric_name in all_metric_names:
        # Skip non-numeric metrics
        sample_value = next((r['metrics'].get(metric_name) for r in results 
                           if metric_name in r['metrics']), None)
        if not isinstance(sample_value, (int, float)):
            continue
        
        # Collect (example_idx, value) pairs
        data_points = []
        for idx, result in enumerate(results):
            if metric_name in result['metrics']:
                data_points.append({
                    'example_idx': idx,
                    'value': float(result['metrics'][metric_name])
                })
        
        if data_points:
            metrics_data[f'eval/{metric_name}'] = data_points
    
    return metrics_data