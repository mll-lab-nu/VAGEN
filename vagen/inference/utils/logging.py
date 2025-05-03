# vagen/inference/utils/logging.py

import logging
import wandb
from typing import Dict, List, Any
from datetime import datetime
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def setup_wandb_for_model(
    model_name: str,
    model_config: Dict,
    inference_config: DictConfig
) -> None:
    """
    Sets up wandb run for a specific model.
    Each model gets its own wandb run for independent tracking.
    
    Args:
        model_name: Name identifier for the model
        model_config: Model configuration
        inference_config: Inference configuration
    """
    if not inference_config.use_wandb:
        return
    
    # Create run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{timestamp}"
    
    # Initialize wandb
    wandb.init(
        project=inference_config.wandb_project,
        name=run_name,
        config={
            "model_name": model_name,
            "model_config": model_config,
            "inference_config": dict(inference_config)
        },
        reinit=True  # Allow multiple runs in same process
    )
    
    logger.info(f"Initialized wandb run: {run_name}")

def log_metrics_to_wandb(
    results: List[Dict],
    model_name: str
) -> None:
    """
    Logs raw environment metrics to wandb.
    No aggregation or modification - just raw metrics from environments.
    
    Args:
        results: List of result dictionaries from environments
        model_name: Model identifier for logging
    """
    if not wandb.run:
        return
    
    # Log each environment's metrics
    for result in results:
        env_id = result.get("env_id", "unknown")
        config_id = result.get("config_id", "unknown")
        metrics = result.get("metrics", {})
        
        # Create wandb log dict with raw metrics
        log_dict = {}
        for metric_name, value in metrics.items():
            # Use hierarchical naming for wandb
            key = f"{model_name}/{config_id}/{metric_name}"
            log_dict[key] = value
        
        # Log to wandb
        wandb.log(log_dict)
    
    logger.debug(f"Logged {len(results)} environment results to wandb")

def log_batch_progress(
    model_name: str,
    batch_idx: int,
    total_batches: int
) -> None:
    """
    Logs batch progress to wandb and console.
    
    Args:
        model_name: Model identifier
        batch_idx: Current batch index (0-based)
        total_batches: Total number of batches
    """
    progress = (batch_idx + 1) / total_batches * 100
    message = f"Model {model_name}: Batch {batch_idx + 1}/{total_batches} ({progress:.1f}%)"
    
    logger.info(message)
    
    if wandb.run:
        wandb.log({
            f"{model_name}/batch_progress": batch_idx + 1,
            f"{model_name}/progress_percent": progress
        })