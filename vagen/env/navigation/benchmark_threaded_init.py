import time
import random
import concurrent.futures
import os
from typing import List
from vagen.env.navigation.env import NavigationEnv
from vagen.env.navigation.config import NavigationConfig


def simple_parallel_benchmark(k: int = 32, m: int = 15, reset_cnt: int = 1, seed: int = 42, max_workers: int = None):
    
    print(f"Starting simple parallel benchmark with {k} environments and {m} steps per environment")
    random.seed(seed)
    
    
    max_workers = max_workers or min(32, (os.cpu_count() or 1) * 2)
    print(f"Using thread pool with {max_workers} workers")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    

    print(f"Initializing {k} environments in parallel...")
    init_start_time = time.time()
    
   
    init_futures = []
    for i in range(k):
        init_futures.append(executor.submit(lambda: (NavigationConfig(), NavigationEnv(NavigationConfig()))))
    
  
    configs = []
    environments = []
    for future in concurrent.futures.as_completed(init_futures):
        config, env = future.result()
        configs.append(config)
        environments.append(env)
    
    init_end_time = time.time()
    init_time = init_end_time - init_start_time
    print(f"Initialized {len(environments)} environments in {init_time:.2f} seconds")
    
 
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
        
      
        reset_seeds = [seed + i + (reset_num * k) for i in range(len(environments))]
        reset_futures = [executor.submit(env.reset, seed=s) for env, s in zip(environments, reset_seeds)]
        
        observations = []
        for future in concurrent.futures.as_completed(reset_futures):
            obs, _ = future.result()
            observations.append(obs)
        
        reset_end_time = time.time()
        reset_time = reset_end_time - reset_start_time
        all_reset_times.append(reset_time)
        print(f"Reset {len(environments)} environments in {reset_time:.2f} seconds")
        
        
        active_envs = list(range(len(environments)))
        env_steps = [0] * len(environments)
        env_rewards = [0.0] * len(environments)
        env_successes = [False] * len(environments)
        
        iteration_step_times = []
        
      
        step_count = 0
        while active_envs and step_count < m:
            step_count += 1
            print(f"Executing step {step_count}/{m} for {len(active_envs)} active environments")
            step_start_time = time.time()
            
           
            actions = []
            for _ in active_envs:
                action = random.choice(action_options)
                formatted_action = f"<think>Let me navigate toward the target.</think><answer>{action}</answer>"
                actions.append(formatted_action)
            
          
            step_futures = [executor.submit(environments[env_idx].step, actions[i]) 
                           for i, env_idx in enumerate(active_envs)]
            
        
            next_active_envs = []
            step_results = []
            
            for i, future in enumerate(concurrent.futures.as_completed(step_futures)):
                env_idx = active_envs[i]
                obs, reward, done, info = future.result()
                step_results.append((env_idx, obs, reward, done, info))
            
        
            for env_idx, obs, reward, done, info in step_results:
                env_steps[env_idx] += 1
                env_rewards[env_idx] += reward
                env_successes[env_idx] = env_successes[env_idx] or info.get('task_success', False)
                
                if not done and env_steps[env_idx] < m:
                    next_active_envs.append(env_idx)
            
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
        print(f"Success rate for this reset: {sum(env_successes) / len(environments):.2%}")
    
   
    total_environments_run = len(environments) * reset_cnt
    total_step_time = sum(all_step_times)
    avg_step_time = total_step_time / max(1, len(all_step_times))
    steps_per_second = all_steps_executed / max(0.001, total_step_time)
    avg_reset_time = sum(all_reset_times) / len(all_reset_times)
    
    
    close_start_time = time.time()
    close_futures = [executor.submit(env.close) for env in environments]
    
  
    for future in concurrent.futures.as_completed(close_futures):
        future.result()  
    
    close_end_time = time.time()
    close_time = close_end_time - close_start_time
    
    
    executor.shutdown()
    
   
    total_time = init_time + sum(all_reset_times) + total_step_time + close_time
    
    
    success_rate = all_env_successes / total_environments_run
    
    
    results = {
        "num_environments": len(environments),
        "reset_count": reset_cnt,
        "steps_per_env": m,
        "workers": max_workers,
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
    
    print("\nSimple Parallel Benchmark Results:")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Environments: {len(environments)}")
    print(f"Reset iterations: {reset_cnt}")
    print(f"Steps per environment: {m}")
    print(f"Total steps executed: {all_steps_executed}")
    print(f"Worker threads: {max_workers}")
    print(f"Average reset time: {avg_reset_time:.4f} seconds")
    print(f"Average step time: {avg_step_time:.4f} seconds")
    print(f"Steps per second: {steps_per_second:.2f}")
    print(f"Average success rate: {success_rate:.2%}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Parallel Benchmark for Navigation Environments')
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
    args = parser.parse_args()
    
    results = simple_parallel_benchmark(
        k=args.environments,
        m=args.steps,
        reset_cnt=args.resets,
        seed=args.seed,
        max_workers=args.workers
    )
    print(results)