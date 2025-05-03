# vagen/inference/run_inference.py

import os
import sys
import argparse
import logging
import yaml
import json
import wandb
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from vagen.mllm_agent.model_interface.factory_model import ModelFactory
from vagen.mllm_agent.inference_rollout.inference_rollout_service import InferenceRolloutService
from vagen.inference.utils.metrics import calculate_aggregate_metrics
from vagen.inference.utils.logging import maybe_log_val_generations_to_wandb, log_metrics_by_config_id

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with models")
    
    parser.add_argument("--inference_config_path", type=str, required=True,
                       help="Path to inference configuration YAML")
    parser.add_argument("--model_config_path", type=str, required=True,
                       help="Path to model configuration YAML")
    parser.add_argument("--val_files_path", type=str, required=True,
                       help="Path to validation dataset parquet file")
    
    return parser.parse_args()

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_environment_configs_from_parquet(val_files_path: str) -> List[Dict]:
    """Load environment configurations from parquet file."""
    df = pd.read_parquet(val_files_path)
    env_configs = []
    
    for idx, row in df.iterrows():
        extra_info = row.get('extra_info', {})
        config = {
            "env_name": extra_info.get("env_name"),
            "env_config": extra_info.get("env_config", {}),
            "seed": extra_info.get("seed", 42)
        }
        env_configs.append(config)
    
    return env_configs

def setup_wandb(model_name: str, model_config: Dict, inference_config: Dict) -> None:
    """Initialize wandb run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_inference_{timestamp}"
    
    wandb.init(
        project=inference_config.get('wandb_project', 'vagen-inference'),
        name=run_name,
        config={
            "model_name": model_name,
            "model_config": model_config,
            "inference_config": inference_config
        }
    )

def run_inference_batch(
    service: InferenceRolloutService,
    env_configs: List[Dict],
    batch_size: int,
    max_steps: int
) -> List[Dict]:
    """Run inference on a batch of environments."""
    all_results = []
    
    # Process environments in batches
    for i in range(0, len(env_configs), batch_size):
        batch_configs = env_configs[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(env_configs) + batch_size - 1)//batch_size}")
        
        # Reset environments for this batch
        service.reset(batch_configs)
        
        # Run inference
        service.run(max_steps=max_steps)
        
        # Get results
        batch_results = service.recording_to_log()
        all_results.extend(batch_results)
        
        # Log batch progress
        if wandb.run:
            wandb.log({
                "batch_progress": (i + batch_size) / len(env_configs) * 100,
                "num_environments_processed": i + len(batch_configs)
            })
    
    return all_results

def log_results_to_wandb(results: List[Dict], global_step: int = 0) -> None:
    """Log results to wandb without any aggregation."""
    # Log raw metrics for each environment
    metrics = log_metrics_by_config_id(results, mode='val')
    wandb.log(metrics)
    
    # Log generation table (unchanged)
    generations_to_log = 10  # You can make this configurable
    maybe_log_val_generations_to_wandb(results, generations_to_log, global_step)
    
    # If you still want some overall summary, just count things
    wandb.log({
        "val/total_environments": len(results),
        "val/num_successful": sum(1 for r in results if r['metrics'].get('success', 0) > 0),
        "val/num_done": sum(1 for r in results if r['metrics'].get('done', 0) > 0),
    })


def main():
    """Main entry point for inference."""
    args = parse_args()
    
    # Load configurations
    inference_config = load_yaml_config(args.inference_config_path)
    model_config = load_yaml_config(args.model_config_path)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if not inference_config.get('debug', False) else logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting inference pipeline")
    
    # Load environment configurations
    env_configs = load_environment_configs_from_parquet(args.val_files_path)
    logger.info(f"Loaded {len(env_configs)} environment configurations")
    
    # Process each model
    models = model_config.get('models', {})
    for model_name, model_cfg in models.items():
        logger.info(f"Running inference for model: {model_name}")
        
        # Setup wandb for this model
        if inference_config.get('use_wandb', True):
            setup_wandb(model_name, model_cfg, inference_config)
        
        try:
            # Create model interface
            model_interface = ModelFactory.create(model_cfg)
            
            # Create inference service
            service = InferenceRolloutService(
                config=inference_config,
                model_interface=model_interface,
                base_url=inference_config.get('server_url', 'http://localhost:5000'),
                timeout=inference_config.get('server_timeout', 600),
                max_workers=inference_config.get('server_max_workers', 48),
                split=inference_config.get('split', 'test'),
                debug=inference_config.get('debug', False)
            )
            
            # Run inference
            results = run_inference_batch(
                service=service,
                env_configs=env_configs,
                batch_size=inference_config.get('batch_size', 32),
                max_steps=inference_config.get('max_steps', 10)
            )
            
            # Log results to wandb
            if inference_config.get('use_wandb', True):
                log_results_to_wandb(results, global_step=0)
            
            # Print summary (no saving to disk)
            print(f"\n===== Results for {model_name} =====")
            print(f"Total environments: {len(results)}")
            
            success_count = sum(1 for r in results if r['metrics'].get('success', 0) > 0)
            done_count = sum(1 for r in results if r['metrics'].get('done', 0) > 0)
            
            print(f"Successful: {success_count}")
            print(f"Completed: {done_count}")
            
            # Print individual results
            print("\nIndividual results:")
            for i, result in enumerate(results):
                print(f"Environment {i}: score={result['metrics'].get('score', 0)}, done={result['metrics'].get('done', 0)}, steps={result['metrics'].get('step', 0)}")
            
        except Exception as e:
            logger.error(f"Error during inference for model {model_name}: {str(e)}")
            raise
        
        finally:
            # Cleanup
            if 'service' in locals():
                service.close()
            
            # Finish wandb run
            if wandb.run:
                wandb.finish()
    
    logger.info("Inference pipeline completed")

if __name__ == "__main__":
    main()