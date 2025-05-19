from typing import List, Dict, Any
from collections import defaultdict
import logging
import wandb
import numpy as np

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
        if generations_to_log <= 0:
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
        
        # Log the table with a dedicated 'table' section
        wandb.log({"table": table}, step=global_steps)


def log_rst_to_metrics_dict(rst, mode='val'):
    """
    Convert raw results to metrics dictionary organized by config_id.
    
    Args:
        rst: List of result dictionaries from rollout
        mode: Mode prefix for metrics (default: 'val')
    
    Returns:
        Dictionary with metrics
    """
    metric_dict = {}
    metrics_by_config_id = defaultdict(lambda: defaultdict(list))
    
    for item in rst:
        config_id = item["config_id"]
        for k, v in item["metrics"].items():
            # Skip complex data types that can't be averaged
            if isinstance(v, (list, dict, tuple)) or v is None:
                continue
            
            # Ensure we only collect numeric values
            if isinstance(v, (int, float)):
                metrics_by_config_id[config_id][k].append(v)
    
    # For each config_id, calculate average metrics
    for config_id, metrics in metrics_by_config_id.items():
        for k, values in metrics.items():
            if values:  # Check if we have any values to calculate mean
                # Calculate mean safely (avoid numpy for potentially mixed types)
                try:
                    metric_dict[f'{mode}/{k}/{config_id}'] = sum(values) / len(values)
                except (TypeError, ValueError):
                    # If we can't calculate mean, skip this metric
                    logger.warning(f"Could not calculate mean for metric {k} with values {values}")
                    continue
    
    return metric_dict

# Global instance for maintaining table state across calls
validation_table_manager = ValidationTableManager()

def maybe_log_val_generations_to_wandb(log_rst: List[Dict[str, Any]], generations_to_log: int = 10, global_steps: int = 0):
    """Log a table of validation samples with multiple images per sample to wandb"""
    validation_table_manager.log_generations_to_wandb(log_rst, generations_to_log, global_steps)

def log_results_to_wandb(results: List[Dict], inference_config: Dict, global_steps: int = 0) -> None:
    """
    Log results to wandb with same format as trainer.
    
    Args:
        results: List of result dictionaries from rollout
        inference_config: Inference configuration dictionary
        global_steps: Global step counter (default: 0)
    """
    # Log metrics by config_id
    metric_dict = log_rst_to_metrics_dict(results, mode='val')
    wandb.log(metric_dict, step=global_steps)
    
    # Log generations table
    val_generations_to_log = inference_config.get('val_generations_to_log_to_wandb', 10)
    maybe_log_val_generations_to_wandb(results, val_generations_to_log, global_steps)