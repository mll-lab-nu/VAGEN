"""
Utility functions for environment setup and configuration.
"""

import torch
from typing import Dict, List
import os

def setup_gpu(gpu_id: int) -> None:
    """
    Set up GPU device if available.

    Args:
        gpu_id: GPU ID to use (-1 for CPU)
    """
    if gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        print(f"Using GPU device {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        print("Using CPU")

def load_environment_configs(args, inference_config) -> List[Dict]:
    """
    Load environment configurations.
    Supports [{"env_name": ..., "env_config": ..., "seed": ...}, ...] structure.
    """
    import os
    from typing import List, Dict

    env_configs = []

    # Try to load from dataset if provided
    if args.dataset:
        dataset_path = os.path.expandvars(args.dataset)
        print(f"Attempting to load environment configs from: {dataset_path}")

        if not os.path.exists(dataset_path):
            print(f"Warning: File not found at {dataset_path}")
            print(f"Current working directory: {os.getcwd()}")
        else:
            try:
                # Try pandas approach
                import pandas as pd
                df = pd.read_parquet(dataset_path)
                
                for i in range(min(len(df), args.max_envs)):
                    row = df.iloc[i].to_dict()
                    if 'extra_info' in row and isinstance(row['extra_info'], dict):
                        extra_info = row['extra_info']
                        config = {
                            "env_name": extra_info.get("env_name"),
                            "env_config": extra_info.get("env_config", {}),
                            "seed": extra_info.get("seed", args.seed)
                        }
                        env_configs.append(config)
                
                if env_configs:
                    print(f"Loaded {len(env_configs)} environment configs from parquet file")
                    return env_configs[:args.max_envs]
            except Exception as e:
                print(f"Failed to read parquet with pandas: {e}")

            # If pandas fails, try datasets library
            try:
                from datasets import load_dataset
                dataset = load_dataset("parquet", data_files=dataset_path, split=args.split)
                
                for i in range(min(len(dataset), args.max_envs)):
                    example = dataset[i]
                    if 'extra_info' in example and example['extra_info']:
                        extra_info = example['extra_info']
                        config = {
                            "env_name": extra_info.get("env_name"),
                            "env_config": extra_info.get("env_config", {}),
                            "seed": extra_info.get("seed", args.seed)
                        }
                        env_configs.append(config)
                
                if env_configs:
                    print(f"Loaded {len(env_configs)} environment configs using datasets library")
                    return env_configs[:args.max_envs]
            except Exception as e:
                print(f"Failed to read with datasets library: {e}")
    
    print(f"Using {len(env_configs)} environment configs for inference")
    return env_configs[:args.max_envs]