# vagen/inference/svg_inference.py

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
                        help="For clarify wandb run's name")
    
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

def maybe_log_val_generations_to_wandb(
    results: List[Dict[str, Any]], 
    generations_to_log: int, 
    global_step: int = 0
) -> None:
    """Log validation generation examples to wandb as a table."""
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
    
    for item in results:
        inputs.append(item['config_id'])
        outputs.append(item['output_str'])
        scores.append(item['metrics']['score'])
        images.append(item.get('image_data', None))
    
    # Find maximum number of images in any sample
    max_images_per_sample = max(
        len(img_list) if img_list else 0
        for img_list in images
    )
    
    # Create samples
    samples = list(zip(inputs, outputs, scores, images))
    
    # Sort and shuffle for consistency
    samples.sort(key=lambda x: x[0])  # Sort by input text
    rng = np.random.RandomState(42)  # Use a fixed seed for reproducibility
    rng.shuffle(samples)
    
    # Take first N samples
    samples = samples[:generations_to_log]
    
    # Create columns for the table (this part remains the same)
    columns = ["input", "output", "score"]
    
    # For SVG, add columns for each turn
    columns.extend(["turn_0_gt_image", "turn_0_gen_image", "turn_0_dino", "turn_0_dreamsim"])
    
    # Add columns for subsequent turns (assuming each has one image)
    for i in range(1, max_images_per_sample - 1):
        columns.extend([f"turn_{i}_image", f"turn_{i}_dino", f"turn_{i}_dreamsim"])
    
    # Create table
    table = wandb.Table(columns=columns)
    
    # Add each sample as a separate row
    for i, sample in enumerate(samples):
        input_text, output_text, score, sample_images = sample
        
        # Get the result for this sample to access metrics
        result_idx = results.index([r for r in results if r['config_id'] == input_text][0])
        result = results[result_idx]
        metrics = result.get('metrics', {})
        
        # Get the scores from metrics
        dino_score = metrics.get('avg_dino_score', 0.0)
        dreamsim_score = metrics.get('avg_dreamsim_score', 0.0)
        
        # Get all_dino_score and all_dreamsim_score if available
        all_dino_scores = metrics.get('all_dino_score', [0.0] * (len(sample_images) if sample_images else 0))
        all_dreamsim_scores = metrics.get('all_dreamsim_score', [0.0] * (len(sample_images) if sample_images else 0))
        
        # Ensure scores lists are long enough
        if all_dino_scores and len(all_dino_scores) < (len(sample_images) if sample_images else 0):
            all_dino_scores.extend([0.0] * ((len(sample_images) if sample_images else 0) - len(all_dino_scores)))
        if all_dreamsim_scores and len(all_dreamsim_scores) < (len(sample_images) if sample_images else 0):
            all_dreamsim_scores.extend([0.0] * ((len(sample_images) if sample_images else 0) - len(all_dreamsim_scores)))
        
        # Basic info
        row = [input_text, output_text, score]
        
        # Add images and scores
        if sample_images:
            if len(sample_images) >= 2:
                # Add first turn (GT and first gen images)
                row.append(wandb.Image(sample_images[0]))  # GT image
                row.append(wandb.Image(sample_images[1]))  # First gen image
                
                # Use individual turn scores if available, otherwise use average
                if all_dino_scores and len(all_dino_scores) > 0:
                    row.append(all_dino_scores[0])
                else:
                    row.append(dino_score)
                
                if all_dreamsim_scores and len(all_dreamsim_scores) > 0:
                    row.append(all_dreamsim_scores[0])
                else:
                    row.append(dreamsim_score)
                
                # Add subsequent turns
                for turn_idx in range(1, len(sample_images) - 1):
                    img_idx = turn_idx + 1  # Skip GT image
                    
                    if img_idx < len(sample_images):
                        row.append(wandb.Image(sample_images[img_idx]))
                        
                        # Add scores for this turn
                        if all_dino_scores and turn_idx < len(all_dino_scores):
                            row.append(all_dino_scores[turn_idx])
                        else:
                            row.append(dino_score)
                        
                        if all_dreamsim_scores and turn_idx < len(all_dreamsim_scores):
                            row.append(all_dreamsim_scores[turn_idx])
                        else:
                            row.append(dreamsim_score)
        
        # Pad with None if needed
        while len(row) < len(columns):
            row.append(None)
        
        # Add row to table
        table.add_data(*row)
    
    # Log the table
    wandb.log({"val/generations": table})

def log_results_to_wandb(results: List[Dict], inference_config: Dict) -> None:
    """Log results to wandb with improved organization."""
    # Log generations table
    val_generations_to_log = inference_config.get('val_generations_to_log_to_wandb', 10)
    maybe_log_val_generations_to_wandb(results, val_generations_to_log, 0)
    
    # Extract metrics and log by example index
    for metric_name in {k for r in results for k in r['metrics'].keys()}:
        values = []
        for i, r in enumerate(results):
            if metric_name in r['metrics'] and isinstance(r['metrics'][metric_name], (int, float)):
                values.append([i, float(r['metrics'][metric_name])])
        
        if not values:
            continue
        
        table = wandb.Table(data=values, columns=["example_idx", "value"])
        wandb.log({
            f"eval/{metric_name}_by_example": wandb.plot.line(
                table, 
                "example_idx", 
                "value",
                title=f"{metric_name} by Example Index"
            )
        })
    
    # Log summary metrics
    summary_metrics = {}
    
    # Basic counts and rates
    summary_metrics['summary/total_examples'] = len(results)
    summary_metrics['summary/num_successful'] = sum(1 for r in results if r['metrics'].get('success', 0) > 0)
    summary_metrics['summary/num_done'] = sum(1 for r in results if r['metrics'].get('done', 0) > 0)
    
    if len(results) > 0:
        summary_metrics['summary/success_rate'] = summary_metrics['summary/num_successful'] / len(results)
        summary_metrics['summary/completion_rate'] = summary_metrics['summary/num_done'] / len(results)
    
    # Calculate means and standard deviations
    metric_values = defaultdict(list)
    for result in results:
        for metric_name, value in result['metrics'].items():
            if isinstance(value, (int, float)):
                metric_values[metric_name].append(float(value))
    
    for metric_name, values in metric_values.items():
        if len(values) > 0:
            summary_metrics[f'summary/{metric_name}_mean'] = np.mean(values)
            if len(values) > 1:
                summary_metrics[f'summary/{metric_name}_std'] = np.std(values)
    
    # Add last turn DINO and DreamSim scores (these are just the avg scores)
    for result in results:
        metrics = result.get('metrics', {})
        if 'avg_dino_score' in metrics:
            summary_metrics['summary/avg_dino_score_mean'] = np.mean([r['metrics']['avg_dino_score'] for r in results if 'avg_dino_score' in r['metrics']])
            if len(results) > 1:
                summary_metrics['summary/avg_dino_score_std'] = np.std([r['metrics']['avg_dino_score'] for r in results if 'avg_dino_score' in r['metrics']])
            break
    
    for result in results:
        metrics = result.get('metrics', {})
        if 'avg_dreamsim_score' in metrics:
            summary_metrics['summary/avg_dreamsim_score_mean'] = np.mean([r['metrics']['avg_dreamsim_score'] for r in results if 'avg_dreamsim_score' in r['metrics']])
            if len(results) > 1:
                summary_metrics['summary/avg_dreamsim_score_std'] = np.std([r['metrics']['avg_dreamsim_score'] for r in results if 'avg_dreamsim_score' in r['metrics']])
            break
    
    wandb.log(summary_metrics)

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
            
            # Log results to wandb
            if inference_config.get('use_wandb', True):
                log_results_to_wandb(results, inference_config)
            
            # Print summary
            print(f"\n===== Results for {model_name} =====")
            print(f"Total environments: {len(results)}")
            
            success_count = sum(1 for r in results if r['metrics'].get('success', 0) > 0)
            done_count = sum(1 for r in results if r['metrics'].get('done', 0) > 0)
            
            print(f"Successful: {success_count}")
            print(f"Completed: {done_count}")
            
            # Print individual results with metrics
            print("\nIndividual results:")
            for i, result in enumerate(results):
                metrics = result.get('metrics', {})
                print(f"Environment {i}: score={metrics.get('score', 0):.3f}, "
                      f"dino={metrics.get('avg_dino_score', 0):.3f}, "
                      f"dreamsim={metrics.get('avg_dreamsim_score', 0):.3f}, "
                      f"done={metrics.get('done', 0)}, "
                      f"steps={metrics.get('step', 0)}")
            
        except Exception as e:
            logger.error(f"Error during inference for model {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
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