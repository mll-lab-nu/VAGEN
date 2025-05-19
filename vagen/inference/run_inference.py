import os
import sys
import argparse
import logging
import yaml
import wandb
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

from vagen.inference.model_interface.factory_model import ModelFactory
from vagen.rollout.inference_rollout.inference_rollout_service import InferenceRolloutService
from vagen.inference.utils.logging import log_results_to_wandb

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
    parser.add_argument("--wandb_path_name", type=str, required=True,
                        help="For clearify wandb run's name")
    
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

def setup_wandb(model_name: str, wandb_path_name: str, model_config: Dict, inference_config: Dict) -> None:
    """Initialize wandb run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{wandb_path_name}_inference"
    
    wandb.init(
        project=inference_config.get('wandb_project', 'vagen-inference'),
        name=run_name,
        config={
            "model_name": model_name,
            "model_config": model_config,
            "inference_config": inference_config
        }
    )

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
            setup_wandb(model_name, args.wandb_path_name, model_cfg, inference_config)
        
        try:
            # Create model interface
            model_interface = ModelFactory.create(model_cfg)
            
            # Create inference service with all config parameters
            service = InferenceRolloutService(
                config=inference_config,
                model_interface=model_interface,
                base_url=inference_config.get('server_url', 'http://localhost:5000'),
                timeout=inference_config.get('server_timeout', 600),
                max_workers=inference_config.get('server_max_workers', 48),
                split=inference_config.get('split', 'test'),
                debug=inference_config.get('debug', False)
            )
            
            # Reset environments and run inference
            service.reset(env_configs)
            service.run(max_steps=inference_config.get('max_steps', 10))
            results = service.recording_to_log()
            
            # Log results to wandb (using the combined logging function)
            if inference_config.get('use_wandb', True):
                log_results_to_wandb(results, inference_config)
            
            # Print summary
            print(f"\n===== Results for {model_name} =====")
            print(f"Total environments: {len(results)}")
            
            success_count = sum(1 for r in results if r['metrics'].get('success', 0) > 0)
            done_count = sum(1 for r in results if r['metrics'].get('done', 0) > 0)
            
            print(f"Successful: {success_count}")
            print(f"Completed: {done_count}")
            
            # Collect metrics by config_id for display
            metrics_by_config = defaultdict(dict)
            for item in results:
                config_id = item['config_id']
                for k, v in item['metrics'].items():
                    if isinstance(v, (int, float)):
                        if k not in metrics_by_config[config_id]:
                            metrics_by_config[config_id][k] = []
                        metrics_by_config[config_id][k].append(v)
            
            # Calculate and print averages for each config_id
            print("\nMetrics by Config ID:")
            for config_id, metrics in metrics_by_config.items():
                print(f"  {config_id}:")
                for k, values in metrics.items():
                    if values:
                        avg = sum(values) / len(values)
                        print(f"    {k}: {avg:.4f}")
            
        except Exception as e:
            logger.error(f"Error during inference for model {model_name}: {str(e)}")
            raise
        
        finally:
            # Cleanup
            if 'service' in locals():
                service.close()
            
            # Finish wandb run
            if wandb.run is not None:
                wandb.finish()
    
    logger.info("Inference pipeline completed")

if __name__ == "__main__":
    main()