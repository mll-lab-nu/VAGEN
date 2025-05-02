#!/usr/bin/env python
"""
Main entry point for running inference using the InferenceRolloutService.
This script handles the high-level flow of the inference process.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from vagen.mllm_agent.inference_rollout.model_interface.factory_model import create_model_interface
from vagen.mllm_agent.inference_rollout.inference_rollout_service import InferenceRolloutService
from .utils.config import load_config, save_configs
from .utils.environment import setup_gpu, load_environment_configs
from .utils.logging import setup_output_dir, setup_wandb
from .utils.metrics import run_validation

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with model on environments")

    # Required arguments
    parser.add_argument("--model_config", type=str, required=True,
                       help="Path to model configuration YAML")
    parser.add_argument("--inference_config", type=str, required=True,
                       help="Path to inference configuration YAML")

    # Optional arguments
    parser.add_argument("--dataset", type=str, default=None,
                       help="Path to dataset with environment configs")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                       help="Directory to save results")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to use")
    parser.add_argument("--max_envs", type=int, default=100,
                       help="Maximum number of environments to evaluate")
    parser.add_argument("--max_steps", type=int, default=10,
                       help="Maximum number of steps per environment")
    parser.add_argument("--server_url", type=str, default="http://localhost:5000",
                       help="Environment server URL")
    parser.add_argument("--timeout", type=int, default=600,
                       help="Server request timeout in seconds")
    parser.add_argument("--max_workers", type=int, default=10,
                       help="Maximum number of parallel workers")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--global_steps", type=int, default=0,
                       help="Current global step (for wandb logging)")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")

    return parser.parse_args()

def main():
    """Main function with simplified high-level flow."""
    # Parse arguments
    args = parse_args()

    # Set up GPU if available
    setup_gpu(args.gpu_id)

    # Load configurations
    model_config = load_config(args.model_config)
    inference_config = load_config(args.inference_config)

    # Set up output directory
    output_dir = setup_output_dir(args.output_dir)
    print(f"Saving results to {output_dir}")

    # Save configurations
    save_configs(output_dir, model_config, inference_config, args)

    # Set up wandb
    if not args.no_wandb:
        setup_wandb(args, model_config, inference_config)

    try:
        # Create model interface
        model_interface = create_model_interface(model_config)

        # Load environment configs
        env_configs = load_environment_configs(args, inference_config)

        # Print model info
        model_info = model_interface.get_model_info()
        print(f"Model: {model_info['name']} (type: {model_info['type']})")
        print(f"Context length: {model_info['context_length']}")

        # Create inference rollout service
        inference_service = InferenceRolloutService(
            config=inference_config,
            model_interface=model_interface,
            base_url=args.server_url,
            timeout=args.timeout,
            max_workers=args.max_workers,
            split=args.split,
            debug=args.debug
        )

        # Run validation and get metrics
        metrics, results = run_validation(
            inference_service=inference_service,
            env_configs=env_configs,
            max_steps=args.max_steps,
            global_steps=args.global_steps,
            output_dir=output_dir,
            use_wandb=not args.no_wandb
        )

        # Print summary metrics
        print("\n===== Inference Results =====")
        print(f"Total environments: {len(env_configs)}")
        print(f"Completed environments: {sum(1 for r in results if r['metrics']['done'])}")
        print(f"Mean score: {metrics.get('mean_score', 0):.4f}")
        print(f"Completion rate: {metrics.get('percent_done', 0):.1f}%")

    finally:
        # Clean up
        if 'inference_service' in locals():
            inference_service.close()

        # Finish wandb run if active
        if not args.no_wandb:
            try:
                import wandb
                if wandb.run:
                    wandb.finish()
            except:
                pass

if __name__ == "__main__":
    main()