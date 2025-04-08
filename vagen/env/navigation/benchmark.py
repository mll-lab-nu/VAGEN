import time
import random
from typing import List
from vagen.env.navigation.env import NavigationEnv
from vagen.env.navigation.config import NavigationConfig

def benchmark_navigation_envs(k: int = 32, m: int = 15, reset_cnt: int = 1, seed: int = 42):
    """
    Benchmark the performance of multiple NavigationEnv instances.
    
    Args:
        k: Number of environments to initialize
        m: Number of steps to execute for each environment
        reset_cnt: Number of times to reset each environment after completion
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"Starting benchmark with {k} environments and {m} steps per environment")
    random.seed(seed)
    
    # Timing for initialization
    init_start_time = time.time()
    
    # Initialize k environments
    environments = []
    configs = []
    for i in range(k):
        config = NavigationConfig()
        configs.append(config)
        env = NavigationEnv(config)
        environments.append(env)
    
    init_end_time = time.time()
    init_time = init_end_time - init_start_time
    print(f"Initialized {k} environments in {init_time:.2f} seconds")
    
    # Simple actions to test with
    action_options = ["moveahead", "moveback", "moveright", "moveleft", 
                     "rotateright", "rotateleft", "lookup", "lookdown"]
    
    # Statistics tracking across all resets
    all_reset_times = []
    all_step_times = []
    all_steps_executed = 0
    all_env_successes = 0
    all_env_rewards = 0.0
    
    # Run the benchmark reset_cnt times for each environment
    for reset_num in range(reset_cnt):
        print(f"\nReset iteration {reset_num + 1}/{reset_cnt}")
        
        # Reset all environments
        reset_start_time = time.time()
        observations = []
        for i, env in enumerate(environments):
            # Use different seeds for diversity, but make them deterministic across reset iterations
            obs, _ = env.reset(seed=seed + i + (reset_num * k))
            observations.append(obs)
        
        reset_end_time = time.time()
        reset_time = reset_end_time - reset_start_time
        all_reset_times.append(reset_time)
        print(f"Reset {k} environments in {reset_time:.2f} seconds")
        
        # Execute m steps for each environment
        active_envs = list(range(k))  # Indices of environments that haven't terminated
        env_steps = [0] * k  # Track steps for each environment
        env_rewards = [0.0] * k  # Track total reward for each environment
        env_successes = [False] * k  # Track if environment reached success state
        
        iteration_step_times = []
        
        # Loop until all environments have taken m steps or terminated
        step_count = 0
        while active_envs and step_count < m:
            step_count += 1
            print(f"Executing step {step_count}/{m} for {len(active_envs)} active environments")
            
            step_start_time = time.time()
            next_active_envs = []
            
            for env_idx in active_envs:
                # Choose a random action
                action = random.choice(action_options)
                formatted_action = f"<think>Let me navigate toward the target.</think><answer>{action}</answer>"
                
                # Execute the action
                obs, reward, done, info = environments[env_idx].step(formatted_action)
                
                # Update tracking variables
                env_steps[env_idx] += 1
                env_rewards[env_idx] += reward
                env_successes[env_idx] = env_successes[env_idx] or info.get('task_success', False)
                
                # Keep track of active environments
                if not done and env_steps[env_idx] < m:
                    next_active_envs.append(env_idx)
            
            active_envs = next_active_envs
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            iteration_step_times.append(step_time)
            print(f"Step {step_count} completed in {step_time:.2f} seconds")
        
        # Update overall statistics for this reset iteration
        iteration_steps_executed = sum(env_steps)
        all_steps_executed += iteration_steps_executed
        all_step_times.extend(iteration_step_times)
        all_env_successes += sum(env_successes)
        all_env_rewards += sum(env_rewards)
        
        print(f"Reset {reset_num + 1} completed with {iteration_steps_executed} total steps executed")
        print(f"Success rate for this reset: {sum(env_successes) / k:.2%}")
    
    # Calculate average statistics across all resets
    total_environments_run = k * reset_cnt
    total_step_time = sum(all_step_times)
    avg_step_time = total_step_time / max(1, len(all_step_times))
    steps_per_second = all_steps_executed / max(0.001, total_step_time)
    avg_reset_time = sum(all_reset_times) / len(all_reset_times)
    
    # Close all environments
    close_start_time = time.time()
    for env in environments:
        env.close()
    close_end_time = time.time()
    close_time = close_end_time - close_start_time
    
    # Total execution time
    total_time = init_time + sum(all_reset_times) + total_step_time + close_time
    
    # Average success rate across all reset iterations
    success_rate = all_env_successes / total_environments_run
    
    # Compile results
    results = {
        "num_environments": k,
        "reset_count": reset_cnt,
        "steps_per_env": m,
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
    
    print("\nBenchmark Results:")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Environments: {k}")
    print(f"Reset iterations: {reset_cnt}")
    print(f"Total environment runs: {total_environments_run}")
    print(f"Steps per environment: {m}")
    print(f"Total steps executed: {all_steps_executed}")
    print(f"Average reset time: {avg_reset_time:.4f} seconds")
    print(f"Average step time: {avg_step_time:.4f} seconds")
    print(f"Steps per second: {steps_per_second:.2f}")
    print(f"Average success rate: {success_rate:.2%}")
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark Navigation Environments')
    parser.add_argument('-k', '--environments', type=int, default=32, 
                        help='Number of environments to initialize')
    parser.add_argument('-m', '--steps', type=int, default=15, 
                        help='Number of steps to execute per environment')
    parser.add_argument('-r', '--resets', type=int, default=2,
                        help='Number of times to reset each environment after completion')
    parser.add_argument('-s', '--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    results = benchmark_navigation_envs(
        k=args.environments,
        m=args.steps,
        reset_cnt=args.resets,
        seed=args.seed
    )
    print(results)