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
        """Log a table of validation samples with each example in its own row."""
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
            images.append(item['image_data'])
        
        # Check if we have images
        has_images = any(img_list for img_list in images)
        
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
        rng = np.random.RandomState()
        rng.shuffle(samples)
        
        # Take first N samples
        samples = samples[:generations_to_log]
        
        # Create columns for the table
        if has_images:
            columns = ["step", "input", "output", "score"] + [f"image_{i+1}" for i in range(max_images_per_sample)]
        else:
            columns = ["step", "input", "output", "score"]
        
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
                table.add_data(global_steps, input_text, output_text, score, *wandb_images)
            else:
                input_text, output_text, score = sample
                table.add_data(global_steps, input_text, output_text, score)
        
        # Log the table
        wandb.log({"val/generations": table})


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
) -> Dict[str, Any]:
    """
    Convert results to metrics dictionary without any aggregation.
    Just logs raw metrics as they are.
    """
    metric_dict = {}
    
    # Simply iterate through results and create metric entries
    for i, item in enumerate(results):
        config_id = item["config_id"]
        env_id = item.get("env_id", f"env_{i}")
        
        # Log each metric as is, with a unique key including env_id
        for k, v in item["metrics"].items():
            metric_key = f'{mode}/{config_id}/{env_id}/{k}'
            metric_dict[metric_key] = v
    
    return metric_dict