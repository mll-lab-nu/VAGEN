# vagen/inference/utils/logging.py

import wandb
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import PIL

logger = logging.getLogger(__name__)

class ValidationTableManager:
    """Manages the cumulative validation table for wandb logging."""
    
    def __init__(self):
        self.validation_table = None
    
    def log_generations_to_wandb(
        self, 
        log_rst: List[Dict[str, Any]], 
        generations_to_log: int, 
        global_steps: int
    ) -> None:
        """
        Log a table of validation samples with multiple images per sample to wandb.
        Matches the training code's _maybe_log_val_generations_to_wandb function.
        """
        if generations_to_log == 0:
            return
            
        if wandb.run is None:
            logger.warning('`val_generations_to_log_to_wandb` is set, but wandb is not initialized')
            return
        
        # Extract data from results exactly like training code
        inputs = []
        outputs = []
        scores = []
        images = []
        
        for item in log_rst:
            inputs.append(item['config_id'])
            outputs.append(item['output_str'])
            scores.append(item['metrics']['score'])
            images.append(item['image_data'])
        
        # Handle the case where images might not be provided
        if images is None or all(img is None for img in images):
            samples = list(zip(inputs, outputs, scores))
            has_images = False
            max_images_per_sample = 0
        else:
            samples = list(zip(inputs, outputs, scores, images))
            has_images = True
            # Find maximum number of images in any sample
            max_images_per_sample = max(
                len(img_list) if isinstance(img_list, (list, tuple)) else 1 
                for img_list in images
            )
        
        # Sort and shuffle exactly like training
        samples.sort(key=lambda x: x[0])  # Sort by input text
        
        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState()  # No seed specified in training code
        rng.shuffle(samples)
        
        # Take first N samples after shuffling
        samples = samples[:generations_to_log]
        
        # Create column names for all samples
        if has_images:
            columns = ["step"]
            for i in range(len(samples)):
                columns.extend([f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"])
                columns.extend([f"image_{i+1}_{j+1}" for j in range(max_images_per_sample)])
        else:
            columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"] 
                                     for i in range(len(samples))], [])
        
        if self.validation_table is None:
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)
        
        # Create a new table with same columns and existing data
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)
        
        # Add new row with all data
        row_data = []
        row_data.append(global_steps)
        
        for sample in samples:
            if has_images:
                input_text, output_text, score, sample_images = sample
                row_data.extend([input_text, output_text, score])
                
                # Handle if sample_images is a single image or list of images
                if not isinstance(sample_images, (list, tuple)):
                    sample_images = [sample_images]
                    
                # Convert each image to wandb.Image
                wandb_images = []
                for img in sample_images:
                    if not isinstance(img, wandb.Image):
                        img = wandb.Image(img)
                    wandb_images.append(img)
                    
                # Pad with None if there are fewer images than max_images_per_sample
                wandb_images.extend([None] * (max_images_per_sample - len(wandb_images)))
                row_data.extend(wandb_images)
            else:
                input_text, output_text, score = sample
                row_data.extend([input_text, output_text, score])
        
        new_table.add_data(*row_data)
        
        # Update reference and log
        wandb.log({"val/generations": new_table})
        self.validation_table = new_table


# Global instance for maintaining table state across calls
validation_table_manager = ValidationTableManager()


def maybe_log_val_generations_to_wandb(
    results: List[Dict[str, Any]], 
    generations_to_log: int, 
    global_step: int
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


def log_metrics_by_config_id(
    results: List[Dict[str, Any]], 
    mode: str = 'inference'
) -> Dict[str, float]:
    """
    Convert results to metrics dictionary with proper prefixes.
    Matches training code's log_rst_to_metrics_dict function.
    """
    from collections import defaultdict
    
    metric_dict = {}
    
    # Group metrics by config_id
    metrics_by_config_id = defaultdict(dict)  # dict of dict of list
    
    for item in results:
        config_id = item["config_id"]
        for k, v in item["metrics"].items():
            if k not in metrics_by_config_id[config_id]:
                metrics_by_config_id[config_id][k] = []
            metrics_by_config_id[config_id][k].append(v)
    
    # Aggregate metrics
    for config_id, metrics in metrics_by_config_id.items():
        for k, v in metrics.items():
            metric_key = f'{mode}/{k}/{config_id}'
            metric_dict[metric_key] = np.mean(v)
    
    return metric_dict