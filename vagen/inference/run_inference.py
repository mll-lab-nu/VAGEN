# vagen/inference/run_inference.py

import argparse
import logging
import wandb
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any
from pathlib import Path

from ..mllm_agent.model_interface.factory_model import ModelFactory
from ..mllm_agent.inference_rollout.inference_rollout_service import InferenceRolloutService
from .utils.environment import load_environment_configs_from_parquet
from .utils.logging import setup_wandb_for_model, log_metrics_to_wandb
from .utils.config import load_yaml_config

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with multiple models")
    
    parser.add_argument("--inference_config_path", type=str, required=True,
                       help="Path to inference configuration YAML")
    parser.add_argument("--model_config_path", type=str, required=True,
                       help="Path to model configuration YAML")
    parser.add_argument("--val_files_path", type=str, required=True,
                       help="Path to validation dataset parquet file")
    
    return parser.parse_args()

def main():
    """Main entry point for parallel multi-model inference."""
    args = parse_args()
    
    # Load configurations
    inference_config = load_yaml_config(args.inference_config_path)
    model_config = load_yaml_config(args.model_config_path)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if not inference_config.get('debug', False) else logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting multi-model inference pipeline")
    
    try:
        run_parallel_model_inference(
            inference_config=inference_config,
            model_config=model_config,
            val_files_path=args.val_files_path
        )
    except Exception as e:
        logger.error(f"Inference pipeline failed: {str(e)}")
        raise
    finally:
        logger.info("Inference pipeline completed")

def run_parallel_model_inference(
    inference_config: Dict,
    model_config: Dict,
    val_files_path: str
) -> None:
    """
    Runs inference for all models in parallel.
    Each model processes all environments independently.
    Uses thread pool for API models, process pool for local models.
    """
    # Load environment configs once for all models
    env_configs = load_environment_configs_from_parquet(val_files_path)
    logger.info(f"Loaded {len(env_configs)} environment configurations")
    
    # Extract models from config
    models = model_config.get('models', {})
    
    # Determine which executor to use based on model types
    use_processes = any(
        model_cfg.get('provider') == "vllm" 
        for model_cfg in models.values()
    )
    
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    max_workers = inference_config.get('num_workers', 4)
    
    logger.info(f"Using {executor_class.__name__} with {max_workers} workers")
    
    # Run inference for each model in parallel
    with executor_class(max_workers=max_workers) as executor:
        futures = []
        
        for model_name, model_cfg in models.items():
            future = executor.submit(
                run_single_model_inference,
                model_name,
                model_cfg,
                env_configs,
                inference_config
            )
            futures.append(future)
        
        # Wait for all models to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Model inference failed: {str(e)}")

def run_single_model_inference(
    model_name: str, 
    model_config: Dict,
    env_configs: List[Dict],
    inference_config: Dict
) -> None:
    """
    Runs inference for a single model on all environments.
    Logs results directly to wandb, no local saving.
    """
    logger.info(f"Starting inference for model: {model_name}")
    
    # Setup wandb for this model
    if inference_config.get('use_wandb', True):
        setup_wandb_for_model(model_name, model_config, inference_config)
    
    try:
        # Create service for this model
        service = create_model_service(model_name, model_config, inference_config)
        
        # Process environments in batches
        batch_size = inference_config.get('batch_size', 32)
        total_batches = (len(env_configs) + batch_size - 1) // batch_size
        
        all_results = []
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(env_configs))
            batch_configs = env_configs[start_idx:end_idx]
            
            logger.info(f"Model {model_name}: Processing batch {batch_idx+1}/{total_batches}")
            
            # Reset environments for this batch
            service.reset(batch_configs)
            
            # Run inference
            service.run(max_steps=inference_config.get('max_steps', 10))
            
            # Get results
            batch_results = service.recording_to_log()
            all_results.extend(batch_results)
            
            # Log metrics to wandb
            if inference_config.get('use_wandb', True):
                log_metrics_to_wandb(batch_results, model_name)
        
        # Print summary
        print_model_summary(model_name, all_results)
        
    except Exception as e:
        logger.error(f"Error during inference for model {model_name}: {str(e)}")
        raise
    finally:
        # Cleanup
        if 'service' in locals():
            cleanup_model_resources(model_name, service)

def print_model_summary(
    model_name: str,
    results: List[Dict]
) -> None:
    """
    Prints a simple summary for the model at the end of inference.
    Just basic stats like completion count, not per-environment details.
    """
    total_envs = len(results)
    completed_envs = sum(1 for r in results if r['metrics'].get('done', False))
    
    print(f"\n===== Model: {model_name} =====")
    print(f"Total environments: {total_envs}")
    print(f"Completed environments: {completed_envs}")
    print(f"Completion rate: {completed_envs/total_envs*100:.1f}%")

def create_model_service(
    model_name: str,
    model_config: Dict,
    inference_config: Dict
) -> InferenceRolloutService:
    """
    Creates InferenceRolloutService for a specific model.
    """
    # Create model interface using factory
    model_interface = ModelFactory.create(model_config)
    
    # Create and return service
    service = InferenceRolloutService(
        config=inference_config,
        model_interface=model_interface,
        base_url=inference_config.get('server_url', 'http://localhost:5000'),
        timeout=inference_config.get('server_timeout', 600),
        max_workers=inference_config.get('server_max_workers', 48),
        split=inference_config.get('use_split', 'test'),
        debug=inference_config.get('debug', False)
    )
    
    return service

def cleanup_model_resources(
    model_name: str,
    service: InferenceRolloutService
) -> None:
    """
    Cleans up resources for a specific model.
    Closes environments and finishes wandb run.
    """
    logger.info(f"Cleaning up resources for model: {model_name}")
    
    # Close environments
    service.close()
    
    # Finish wandb run if active
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()