import time
import random
import concurrent.futures
import os
import traceback
from typing import List, Dict, Any, Tuple
from vagen.env.navigation.env import NavigationEnv
from vagen.env.navigation.config import NavigationConfig


def _safe_reset_env(env, seed):
   
    try:
        result = env.reset(seed=seed)
        return {"success": True, "data": result}
    except Exception as e:
        error_msg = f"Reset error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"success": False, "error": error_msg}


def _safe_step_env(env, action):
    
    try:
        result = env.step(action)
        return {"success": True, "data": result}
    except Exception as e:
        error_msg = f"Step error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"success": False, "error": error_msg}


def _safe_close_env(env):
    
    try:
        env.close()
        return {"success": True}
    except Exception as e:
        error_msg = f"Close error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"success": False, "error": error_msg}


def thread_pool_benchmark_navigation_envs(
    k: int = 32,
    m: int = 15,
    reset_cnt: int = 1,
    seed: int = 42,
    max_workers: int = None,
    timeout: float = 30.0,
    batch_size: int = 8
):
    
    print(f"Starting thread pool benchmark with {k} environments and {m} steps per environment")
    random.seed(seed)
    
    
    max_workers = max_workers or min(32, (os.cpu_count() or 1) * 2)
    print(f"Using thread pool with {max_workers} workers")
    
   
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    
    init_start_time = time.time()
    
    
    environments = []
    configs = []
    
    for i in range(0, k, batch_size):
        batch_end = min(i + batch_size, k)
        print(f"Initializing environments {i} to {batch_end-1}")
        
       
        batch_configs = []
        for j in range(i, batch_end):
            config = NavigationConfig()
            configs.append(config)
            batch_configs.append(config)
        
        
        batch_envs = []
        for config in batch_configs:
            env = NavigationEnv(config)
            environments.append(env)
            batch_envs.append(env)
    
    init_end_time = time.time()
    init_time = init_end_time - init_start_time
    print(f"Initialized {k} environments in {init_time:.2f} seconds")
    
    
    action_options = ["moveahead", "moveback", "moveright", "moveleft", 
                     "rotateright", "rotateleft", "lookup", "lookdown"]
    
    
    all_reset_times = []
    all_step_times = []
    all_steps_executed = 0
    all_env_successes = 0
    all_env_rewards = 0.0
    
    
    for reset_num in range(reset_cnt):
        print(f"\nReset iteration {reset_num + 1}/{reset_cnt}")
        
       
        reset_start_time = time.time()
        observations = [None] * k
        
        for i in range(0, k, batch_size):
            batch_end = min(i + batch_size, k)
            print(f"Resetting environments {i} to {batch_end-1}")
            
            
            reset_futures = {}
            for j in range(i, batch_end):
                
                reset_seed = seed + j + (reset_num * k)
                reset_futures[j] = executor.submit(_safe_reset_env, environments[j], reset_seed)
            
            
            for j, future in reset_futures.items():
                try:
                    result = future.result(timeout=timeout)
                    if result["success"]:
                        observations[j] = result["data"][0] 
                    else:
                        print(f"Failed to reset environment {j}: {result.get('error')}")
                        observations[j] = None
                except concurrent.futures.TimeoutError:
                    print(f"Timeout resetting environment {j}")
                    observations[j] = None
                except Exception as e:
                    print(f"Error resetting environment {j}: {e}")
                    observations[j] = None
        
        reset_end_time = time.time()
        reset_time = reset_end_time - reset_start_time
        all_reset_times.append(reset_time)
        print(f"Reset {k} environments in {reset_time:.2f} seconds")
        
       
        active_envs = list(range(k))  
        env_steps = [0] * k  
        env_rewards = [0.0] * k  
        env_successes = [False] * k  
        
        iteration_step_times = []
        
       
        step_count = 0
        while active_envs and step_count < m:
            step_count += 1
            print(f"Executing step {step_count}/{m} for {len(active_envs)} active environments")
            
            step_start_time = time.time()
            next_active_envs = []
            
            
            for i in range(0, len(active_envs), batch_size):
                batch_end = min(i + batch_size, len(active_envs))
                batch_indices = active_envs[i:batch_end]
                
                
                step_futures = {}
                for env_idx in batch_indices:
                    
                    action = random.choice(action_options)
                    formatted_action = f"<think>Let me navigate toward the target.</think><answer>{action}</answer>"
                    
                    
                    step_futures[env_idx] = executor.submit(
                        _safe_step_env, 
                        environments[env_idx], 
                        formatted_action
                    )
                
                
                for env_idx, future in step_futures.items():
                    try:
                        result = future.result(timeout=timeout)
                        if result["success"]:
                            obs, reward, done, info = result["data"]
                            
                            
                            env_steps[env_idx] += 1
                            env_rewards[env_idx] += reward
                            env_successes[env_idx] = env_successes[env_idx] or info.get('task_success', False)
                            
                            
                            if not done and env_steps[env_idx] < m:
                                next_active_envs.append(env_idx)
                        else:
                            print(f"Step failed for env {env_idx}: {result.get('error')}")
                            
                    except concurrent.futures.TimeoutError:
                        print(f"Timeout during step for env {env_idx}")
                        
                    except Exception as e:
                        print(f"Error during step for env {env_idx}: {e}")
                        
            
            active_envs = next_active_envs
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            iteration_step_times.append(step_time)
            print(f"Step {step_count} completed in {step_time:.2f} seconds")
        
        
        iteration_steps_executed = sum(env_steps)
        all_steps_executed += iteration_steps_executed
        all_step_times.extend(iteration_step_times)
        all_env_successes += sum(env_successes)
        all_env_rewards += sum(env_rewards)
        
        print(f"Reset {reset_num + 1} completed with {iteration_steps_executed} total steps executed")
        print(f"Success rate for this reset: {sum(env_successes) / k:.2%}")
    
   
    total_environments_run = k * reset_cnt
    total_step_time = sum(all_step_times)
    avg_step_time = total_step_time / max(1, len(all_step_times))
    steps_per_second = all_steps_executed / max(0.001, total_step_time)
    avg_reset_time = sum(all_reset_times) / len(all_reset_times)
    
  
    close_start_time = time.time()
    
    for i in range(0, k, batch_size):
        batch_end = min(i + batch_size, k)
        print(f"Closing environments {i} to {batch_end-1}")
        
        
        close_futures = {}
        for j in range(i, batch_end):
            close_futures[j] = executor.submit(_safe_close_env, environments[j])
        
        
        for j, future in close_futures.items():
            try:
                result = future.result(timeout=timeout)
                if not result["success"]:
                    print(f"Warning: Failed to close environment {j}")
            except Exception as e:
                print(f"Error closing environment {j}: {e}")
    
    close_end_time = time.time()
    close_time = close_end_time - close_start_time
    
    
    executor.shutdown(wait=False)
    
   
    total_time = init_time + sum(all_reset_times) + total_step_time + close_time
    
    
    success_rate = all_env_successes / total_environments_run
    
    
    results = {
        "num_environments": k,
        "reset_count": reset_cnt,
        "steps_per_env": m,
        "max_workers": max_workers,
        "batch_size": batch_size,
        "total_environments_initialized": k,
        "total_environments_run": total_environments_run,
        "total_steps_executed": all_steps_executed,
        "initialization_time": init_time,
        "avg_reset_time": avg_reset_time,
        "total_reset_time": sum(all_reset_times),
        "step_time_total": total_step_time,
        "step_time_average": avg_step_time,
        "steps_per_second": steps_per_second,
        "environments_per_second": steps_per_second / m if m > 0 else 0,
        "close_time": close_time,
        "total_execution_time": total_time,
        "success_rate": success_rate,
        "average_reward": all_env_rewards / total_environments_run
    }
    
    print("\nThread Pool Benchmark Results:")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Environments: {k}")
    print(f"Reset iterations: {reset_cnt}")
    print(f"Total environment runs: {total_environments_run}")
    print(f"Steps per environment: {m}")
    print(f"Total steps executed: {all_steps_executed}")
    print(f"Thread pool workers: {max_workers}")
    print(f"Batch size: {batch_size}")
    print(f"Average reset time: {avg_reset_time:.4f} seconds")
    print(f"Average step time: {avg_step_time:.4f} seconds")
    print(f"Steps per second: {steps_per_second:.2f}")
    print(f"Average success rate: {success_rate:.2%}")
    
    return results


if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser(description='Thread Pool Benchmark for Navigation Environments')
    parser.add_argument('-k', '--environments', type=int, default=32, 
                        help='Number of environments to initialize')
    parser.add_argument('-m', '--steps', type=int, default=15, 
                        help='Number of steps to execute per environment')
    parser.add_argument('-r', '--resets', type=int, default=2,
                        help='Number of times to reset each environment after completion')
    parser.add_argument('-s', '--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('-w', '--workers', type=int, default=None,
                        help='Number of worker threads in thread pool (default: auto)')
    parser.add_argument('-t', '--timeout', type=float, default=30.0,
                        help='Timeout for operations in seconds (default: 30.0)')
    parser.add_argument('-b', '--batch', type=int, default=16,
                        help='Batch size for environment operations (default: 8)')
    args = parser.parse_args()
    
    results = thread_pool_benchmark_navigation_envs(
        k=args.environments,
        m=args.steps,
        reset_cnt=args.resets,
        seed=args.seed,
        max_workers=args.workers,
        timeout=args.timeout,
        batch_size=args.batch
    )
    print(results)