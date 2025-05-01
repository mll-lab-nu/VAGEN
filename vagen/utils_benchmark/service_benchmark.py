import argparse
import time
import json
import random
import statistics
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from datasets import load_dataset
from tqdm import tqdm

from vagen.server.client import BatchEnvClient
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

def benchmark_service(config_path):
    """
    Main benchmark function.
    
    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize client
    server_config = config.get('server', {})
    client = BatchEnvClient(
        base_url=server_config.get('base_url', 'http://localhost:5000'),
        timeout=server_config.get('timeout', 600),
        max_workers=server_config.get('max_workers', 48)
    )
    
    # Get benchmark parameters
    benchmark_config = config.get('benchmark', {})
    batch_sizes = benchmark_config.get('batch_sizes', [128])
    iterations = benchmark_config.get('iterations', 3)
    step_count = benchmark_config.get('step_count', 5)
    output_dir = benchmark_config.get('output_dir', 'benchmark_results')
    functions = benchmark_config.get('functions', [
        'create_environments_batch',
        'reset_batch', 
        'step_batch',
        'compute_reward_batch',
        'get_system_prompts_batch',
        'close_batch'
    ])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if server is alive
    if not client.wait_for_server():
        print("Server not available. Exiting.")
        return
    
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
            'per_env_timings': {func: [] for func in functions}
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
                env_ids = [f"{env_name}_{i}" for i in range(batch_size)]
                
                # Dictionary of environment configurations
                ids2configs = {env_id: config for env_id, config in zip(env_ids, batch_configs)}
                
                # Dictionary of seeds for resetting environments
                ids2seeds = {env_id: config['seed'] for env_id, config in zip(env_ids, batch_configs)}
                
                # ----- Benchmark create_environments_batch -----
                if 'create_environments_batch' in functions:
                    print("    Creating environments...", end='', flush=True)
                    start_time = time.time()
                    client.create_environments_batch(ids2configs)
                    end_time = time.time()
                    create_time = end_time - start_time
                    batch_timings['create_environments_batch'].append(create_time)
                    print(f" {create_time:.4f}s ({create_time/batch_size:.6f}s per env)")
                
                # ----- Benchmark reset_batch -----
                if 'reset_batch' in functions:
                    print("    Resetting environments...", end='', flush=True)
                    start_time = time.time()
                    reset_results = client.reset_batch(ids2seeds)
                    end_time = time.time()
                    reset_time = end_time - start_time
                    batch_timings['reset_batch'].append(reset_time)
                    print(f" {reset_time:.4f}s ({reset_time/batch_size:.6f}s per env)")
                
                # ----- Benchmark step_batch -----
                if 'step_batch' in functions:
                    step_times = []
                    
                    # Need to reset to get initial observations
                    if 'reset_batch' not in functions:
                        reset_results = client.reset_batch(ids2seeds)
                    
                    # Run multiple steps
                    for step in range(step_count):
                        print(f"    Step {step+1}/{step_count}...", end='', flush=True)
                        
                        # Generate actions for each environment
                        ids2actions = {}
                        for env_id in env_ids:
                            if env_id in reset_results:
                                observation, info = reset_results[env_id]
                                available_commands = info.get('admissible_commands', [])
                                ids2actions[env_id] = generate_random_action(available_commands)
                        
                        # Measure step time
                        start_time = time.time()
                        step_results = client.step_batch(ids2actions)
                        end_time = time.time()
                        step_time = end_time - start_time
                        step_times.append(step_time)
                        print(f" {step_time:.4f}s ({step_time/batch_size:.6f}s per env)")
                        
                        # Use step results as observations for next step
                        reset_results = {env_id: (obs, info) for env_id, (obs, _, _, info) in step_results.items()}
                    
                    # Store average step time
                    avg_step_time = statistics.mean(step_times)
                    batch_timings['step_batch'].append(avg_step_time)
                    print(f"    Average step time: {avg_step_time:.4f}s ({avg_step_time/batch_size:.6f}s per env)")
                
                # ----- Benchmark compute_reward_batch -----
                if 'compute_reward_batch' in functions:
                    print("    Computing rewards...", end='', flush=True)
                    start_time = time.time()
                    client.compute_reward_batch(env_ids)
                    end_time = time.time()
                    reward_time = end_time - start_time
                    batch_timings['compute_reward_batch'].append(reward_time)
                    print(f" {reward_time:.4f}s ({reward_time/batch_size:.6f}s per env)")
                
                # ----- Benchmark get_system_prompts_batch -----
                if 'get_system_prompts_batch' in functions:
                    print("    Getting system prompts...", end='', flush=True)
                    start_time = time.time()
                    client.get_system_prompts_batch(env_ids)
                    end_time = time.time()
                    prompt_time = end_time - start_time
                    batch_timings['get_system_prompts_batch'].append(prompt_time)
                    print(f" {prompt_time:.4f}s ({prompt_time/batch_size:.6f}s per env)")
                
                # ----- Benchmark close_batch -----
                if 'close_batch' in functions:
                    print("    Closing environments...", end='', flush=True)
                    start_time = time.time()
                    client.close_batch(env_ids)
                    end_time = time.time()
                    close_time = end_time - start_time
                    batch_timings['close_batch'].append(close_time)
                    print(f" {close_time:.4f}s ({close_time/batch_size:.6f}s per env)")
                else:
                    # Make sure environments are closed even if not benchmarking close
                    client.close_batch(env_ids)
            
            # Compute average timings for this batch size
            env_results['batch_sizes'].append(batch_size)
            
            for func in functions:
                avg_time = statistics.mean(batch_timings[func])
                env_results['timings'][func].append(avg_time)
                env_results['per_env_timings'][func].append(avg_time / batch_size)
        
        # Store results for this environment
        results[env_name] = env_results
    
    # Save results as JSON
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
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
        
        for func in functions:
            # Skip if this function wasn't benchmarked for this environment
            if func not in env_results['timings'] or not env_results['timings'][func]:
                continue
                
            timings = env_results['timings'][func]
            per_env_timings = env_results['per_env_timings'][func]
            
            print(f"  {func}:")
            for i, batch_size in enumerate(batch_sizes):
                print(f"    Batch size {batch_size}: {timings[i]:.4f}s total, {per_env_timings[i]:.6f}s per env")
            
            # Print scaling behavior
            if len(batch_sizes) > 1:
                scaling = timings[-1] / timings[0] * batch_sizes[0] / batch_sizes[-1]
                print(f"    Scaling efficiency (batch {batch_sizes[0]} -> {batch_sizes[-1]}): {scaling:.2f}x")
                
                if scaling < 0.5:
                    print(f"    ⚠️  Poor scaling for {func} in {env_name}")
                elif scaling > 0.8:
                    print(f"    ✅ Good scaling for {func} in {env_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark service functions")
    parser.add_argument("--config", type=str, default="benchmark_config.yaml", help="Path to configuration YAML")
    args = parser.parse_args()
    
    benchmark_service(args.config)