import argparse
import time
import json
import random
import statistics
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datasets import load_dataset

from vagen.env import REGISTERED_ENV

def generate_random_action(available_commands):
    """Generate a random action from available commands."""
    if not available_commands:
        return "<think>Let me explore.</think><answer>look</answer>"
        
    if isinstance(available_commands, list) and len(available_commands) > 0:
        if isinstance(available_commands[0], list):
            available_commands = available_commands[0]
            
    action = random.choice(available_commands) if available_commands else "look"
    return f"<think>I'll try this action.</think><answer>{action}</answer>"

def benchmark_environments(config_path):
    """
    Main benchmark function for environments.
    
    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get benchmark parameters
    benchmark_config = config.get('benchmark', {})
    batch_sizes = benchmark_config.get('batch_sizes', [8])
    iterations = benchmark_config.get('iterations', 3)
    step_count = benchmark_config.get('step_count', 5)
    output_dir = benchmark_config.get('output_dir', 'env_benchmark_results')
    functions = benchmark_config.get('functions', ['system_prompt', 'reset', 'step', 'compute_reward', 'close'])
    valid_commands = benchmark_config.get('valid_commands', None)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load environment configs from datasets
    datasets_config = config.get('datasets', [])
    all_env_configs = {}
    
    for dataset_config in datasets_config:
        name = dataset_config.get('name')
        train_path = dataset_config.get('train_path')
        test_path = dataset_config.get('test_path')
        use_split = dataset_config.get('use_split', 'both')
        
        # Load train dataset if needed
        if use_split in ['train', 'both'] and os.path.exists(train_path):
            try:
                train_dataset = load_dataset('parquet', data_files=train_path, split="train")
                print(f"Loaded train dataset from {train_path} with {len(train_dataset)} examples")
                
                # Extract environment configs
                for i in range(len(train_dataset)):
                    example = train_dataset[i]
                    env_config = {
                        'env_name': example['extra_info']['env_name'],
                        'env_config': example['extra_info']['env_config'],
                        'seed': example['extra_info']['seed']
                    }
                    all_env_configs.setdefault(name, []).append(env_config)
            except Exception as e:
                print(f"Failed to load train dataset from {train_path}: {e}")
        
        # Load test dataset if needed
        if use_split in ['test', 'both'] and os.path.exists(test_path):
            try:
                test_dataset = load_dataset('parquet', data_files=test_path, split="train")
                print(f"Loaded test dataset from {test_path} with {len(test_dataset)} examples")
                
                # Extract environment configs
                for i in range(len(test_dataset)):
                    example = test_dataset[i]
                    env_config = {
                        'env_name': example['extra_info']['env_name'],
                        'env_config': example['extra_info']['env_config'],
                        'seed': example['extra_info']['seed']
                    }
                    all_env_configs.setdefault(name, []).append(env_config)
            except Exception as e:
                print(f"Failed to load test dataset from {test_path}: {e}")
    
    # Dictionary to store all benchmark results
    results = {}
    
    # Run benchmark for each environment type
    for env_name, env_configs in all_env_configs.items():
        print(f"\n===== Benchmarking {env_name} =====")
        
        # Store results for this environment
        env_results = {
            'batch_sizes': [],
            'timings': {func: [] for func in functions},
        }
        
        # Test different batch sizes
        for batch_size in batch_sizes:
            if batch_size > len(env_configs):
                print(f"Skipping batch size {batch_size}: not enough configs available")
                continue
                
            print(f"\nBatch size: {batch_size}")
            
            # Run multiple iterations for statistical significance
            batch_timings = {func: [] for func in functions}
            
            for iteration in range(iterations):
                print(f"  Iteration {iteration+1}/{iterations}")
                
                # Sample environment configs for this batch
                batch_configs = random.sample(env_configs, batch_size)
                
                # Create all environments for this batch
                envs = []
                for config in batch_configs:
                    env_name = config['env_name']
                    env_cls = REGISTERED_ENV[env_name]['env_cls']
                    config_cls = REGISTERED_ENV[env_name]['config_cls']
                    env_config = config_cls(**config.get('env_config', {}))
                    env = env_cls(env_config)
                    envs.append((env, config['seed']))
                
                # --- Benchmark system_prompt ---
                print("    Testing system_prompt...", end='', flush=True)
                start_time = time.time()
                for env, _ in envs:
                    env.system_prompt()
                system_prompt_time = time.time() - start_time
                batch_timings['system_prompt'].append(system_prompt_time)
                print(f" {system_prompt_time:.4f}s total")
                
                # --- Benchmark reset ---
                print("    Testing reset...", end='', flush=True)
                start_time = time.time()
                obs_infos = []
                for env, seed in envs:
                    obs, info = env.reset(seed=seed)
                    obs_infos.append((obs, info))
                reset_time = time.time() - start_time
                batch_timings['reset'].append(reset_time)
                print(f" {reset_time:.4f}s total")
                
                # --- Benchmark step ---
                print("    Testing step...", end='', flush=True)
                step_times = []
                for step_idx in range(step_count):
                    actions = []
                    for i, (env, _) in enumerate(envs):
                        obs, info = obs_infos[i]
                        if valid_commands:
                            action = generate_random_action(valid_commands)
                        else:
                            available_commands = info.get('admissible_commands', [])
                            action = generate_random_action(available_commands)
                        actions.append(action)
                    
                    start_time = time.time()
                    for i, (env, _) in enumerate(envs):
                        obs, reward, done, info = env.step(actions[i])
                        obs_infos[i] = (obs, info)  # Update for next step
                    step_time = time.time() - start_time
                    step_times.append(step_time)
                    
                if step_times:
                    avg_step_time = statistics.mean(step_times)
                    batch_timings['step'].append(avg_step_time)
                    print(f" {avg_step_time:.4f}s total (average per step)")
                
                # --- Benchmark compute_reward ---
                print("    Testing compute_reward...", end='', flush=True)
                start_time = time.time()
                for env, _ in envs:
                    env.compute_reward()
                compute_reward_time = time.time() - start_time
                batch_timings['compute_reward'].append(compute_reward_time)
                print(f" {compute_reward_time:.4f}s total")
                
                # --- Benchmark close ---
                print("    Testing close...", end='', flush=True)
                start_time = time.time()
                for env, _ in envs:
                    env.close()
                close_time = time.time() - start_time
                batch_timings['close'].append(close_time)
                print(f" {close_time:.4f}s total")
            
            # Store results for this batch size
            env_results['batch_sizes'].append(batch_size)
            
            for func in functions:
                if func in batch_timings and batch_timings[func]:
                    avg_time = statistics.mean(batch_timings[func])
                    env_results['timings'][func].append(avg_time)
                else:
                    env_results['timings'][func].append(None)
        
        # Store results for this environment
        results[env_name] = env_results
    
    # Save results as JSON
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(output_dir, f"env_benchmark_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {results_file}")
    
    # Generate plots if we have results
    if results:
        print_summary(results, functions)
    else:
        print("\nNo benchmark results to plot or summarize.")

def print_summary(results, functions):
    """
    Print a summary of the benchmark results.
    
    Args:
        results: Dictionary of benchmark results
        functions: List of functions that were benchmarked
    """
    print("\n===== Summary =====")
    
    for env_name, env_results in results.items():
        print(f"\n{env_name}:")
        batch_sizes = env_results['batch_sizes']
        
        # Print function timings
        for func in functions:
            # Skip if this function wasn't benchmarked for this environment
            if func not in env_results['timings'] or not env_results['timings'][func]:
                continue
                
            timings = env_results['timings'][func]
            
            print(f"  {func} times:")
            for i, batch_size in enumerate(batch_sizes):
                if i < len(timings) and timings[i] is not None:
                    print(f"    Batch size {batch_size}: {timings[i]:.4f}s total")
            
            # Print scaling behavior
            if len(batch_sizes) > 1 and all(t is not None for t in timings):
                ideal_scaling = batch_sizes[-1] / batch_sizes[0]
                actual_scaling = timings[-1] / timings[0]
                efficiency = ideal_scaling / actual_scaling
                print(f"    Scaling efficiency (batch {batch_sizes[0]} → {batch_sizes[-1]}): {efficiency:.2f}x")
                
                if efficiency < 0.5:
                    print(f"    ⚠️  Poor scaling for {func}")
                elif efficiency > 0.8:
                    print(f"    ✅ Good scaling for {func}")

def create_config_file(output_path="env_benchmark_config.yaml"):
    """Create a default configuration file for the benchmark."""
    config = {
        "benchmark": {
            "functions": ["system_prompt", "reset", "step", "compute_reward", "close"],
            "iterations": 3,
            "step_count": 5,
            "batch_sizes": [8, 16, 32],
            "output_dir": "env_benchmark_results",
            "valid_commands": ["Left", "Right", "Up", "Down"]
        },
        "datasets": [
            {
                "name": "frozenlake-vision",
                "train_path": "data/frozenlake-vision-benchmark/train.parquet",
                "test_path": "data/frozenlake-vision-benchmark/test.parquet",
                "use_split": "both"
            }
        ]
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created default configuration file at {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark environment functions")
    parser.add_argument("--config", type=str, help="Path to configuration YAML")
    parser.add_argument("--create-config", action="store_true", help="Create a default configuration file")
    args = parser.parse_args()
    
    if args.create_config:
        config_path = create_config_file()
    elif args.config:
        config_path = args.config
    else:
        # Look for default config file
        default_config = "env_benchmark_config.yaml"
        if os.path.exists(default_config):
            config_path = default_config
        else:
            print("No configuration file provided. Creating default configuration.")
            config_path = create_config_file()
    
    benchmark_environments(config_path)