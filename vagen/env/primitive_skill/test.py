import time
import random
import numpy as np
from typing import Dict, List, Any
import uuid
import statistics

from vagen.env.primitive_skill.service import PrimitiveSkillService
from vagen.env.primitive_skill.env_config import PrimitiveSkillEnvConfig
from vagen.env.primitive_skill.service_config import PrimitiveSkillConfig

def generate_random_action():
    """Generate a random action string (pick, place, or push with random coordinates)."""
    action_types = ["pick", "place", "push"]
    action_type = random.choice(action_types)
    
    # Generate random coordinates within typical workspace limits
    # Assuming workspace limits around [-1, 1] for x, y, z (multiplied by 1000 per the parsing logic)
    x = random.uniform(-0.8, 0.8) * 1000
    y = random.uniform(-0.8, 0.8) * 1000
    z = random.uniform(0, 0.5) * 1000
    
    if action_type in ["pick", "place"]:
        action_str = f"{action_type}({x:.2f}, {y:.2f}, {z:.2f})"
    else:  # push
        # Generate second point for push
        x2 = random.uniform(-0.8, 0.8) * 1000
        y2 = random.uniform(-0.8, 0.8) * 1000
        z2 = random.uniform(0, 0.5) * 1000
        action_str = f"{action_type}({x:.2f}, {y:.2f}, {z:.2f}, {x2:.2f}, {y2:.2f}, {z2:.2f})"
    
    # Format according to the expected response format
    return f"<think>Executing {action_type} action.</think><answer>{action_str}</answer>"


def main(num_envs=32, num_steps=3,max_process_workers=8,max_thread_workers=8):
    # Tasks to benchmark - create 32 environments for each task
    tasks = ["AlignTwoCube", "PlaceTwoCube", "PutAppleInDrawer", "StackThreeCube"]
    
    all_stats = {}
    total_envs = num_envs * len(tasks)  # 32 envs per task
    
    print(f"Creating {num_envs} environments for each of {len(tasks)} tasks ({total_envs} total environments)")
    
    # Create configurations for all environments
    env_configs = {}
    env_ids_by_task = {}
    
    for task in tasks:
        # Generate 32 environment IDs for this task
        task_env_ids = [f"{task}_{i}" for i in range(num_envs)]
        env_ids_by_task[task] = task_env_ids
        
        # Create configurations for this task
        for env_id in task_env_ids:
            env_configs[env_id] = {
                'env_name': 'primitive_skill',
                'env_config': {
                    'env_id': task,
                    'render_mode': 'vision',
                    'record_video': False,
                }
            }
    
    # Create service configuration with higher number of workers for parallelism
    service_config = PrimitiveSkillConfig(max_process_workers=max_process_workers, max_thread_workers=max_thread_workers)
    service = PrimitiveSkillService(service_config)
    
    # Timing dictionary
    timings = {
        'create_environments': [],
        'reset': [],
        'steps': [],
        'compute_reward': [],
        'get_system_prompts': [],
        'close': []
    }
    
    print(f"\n===== Creating all environments =====")
    
    # ----- Create all environments at once -----
    start_time = time.time()
    service.create_environments_batch(env_configs)
    end_time = time.time()
    create_time = end_time - start_time
    timings['create_environments'].append(create_time)
    print(f"Created {total_envs} environments in {create_time:.2f} seconds ({create_time/total_envs:.4f} seconds per env)")
    
    # ----- Reset all environments -----
    print("\n===== Resetting all environments =====")
    all_env_ids = list(env_configs.keys())
    ids2seeds = {env_id: random.randint(0, 10000) for env_id in all_env_ids}
    start_time = time.time()
    reset_results = service.reset_batch(ids2seeds)
    end_time = time.time()
    reset_time = end_time - start_time
    timings['reset'].append(reset_time)
    print(f"Reset {total_envs} environments in {reset_time:.2f} seconds ({reset_time/total_envs:.4f} seconds per env)")
    
    # ----- Run steps on all environments -----
    print(f"\n===== Running 5 steps on all environments =====")
    step_times = []
    
    for step in range(num_steps):  # 5 steps per environment
        # Generate random actions for all environments
        ids2actions = {env_id: generate_random_action() for env_id in all_env_ids}
        
        # Execute step
        start_time = time.time()
        step_results = service.step_batch(ids2actions)
        end_time = time.time()
        step_time = end_time - start_time
        step_times.append(step_time)
        
        print(f"Step {step+1}: Completed {total_envs} environment steps in {step_time:.2f} seconds ({step_time/total_envs:.4f} seconds per env)")
        
        # Check validity of actions and any terminated environments
        valid_actions = sum(1 for env_id, (_, _, _, info) in step_results.items() if info.get('action_is_valid', False))
        terminated = sum(1 for env_id, (_, _, done, _) in step_results.items() if done)
        
        print(f"  - Valid actions: {valid_actions}/{total_envs}")
        print(f"  - Terminated environments: {terminated}/{total_envs}")
    
    timings['steps'] = step_times
    
    # ----- Compute rewards for all environments -----
    print("\n===== Computing rewards for all environments =====")
    start_time = time.time()
    rewards = service.compute_reward_batch(all_env_ids)
    end_time = time.time()
    reward_time = end_time - start_time
    timings['compute_reward'].append(reward_time)
    print(f"Computed rewards for {total_envs} environments in {reward_time:.2f} seconds ({reward_time/total_envs:.4f} seconds per env)")
    
    # ----- Get system prompts for all environments -----
    print("\n===== Getting system prompts for all environments =====")
    start_time = time.time()
    prompts = service.get_system_prompts_batch(all_env_ids)
    end_time = time.time()
    prompt_time = end_time - start_time
    timings['get_system_prompts'].append(prompt_time)
    print(f"Retrieved system prompts for {total_envs} environments in {prompt_time:.2f} seconds ({prompt_time/total_envs:.4f} seconds per env)")
    
    # ----- Close all environments -----
    print("\n===== Closing all environments =====")
    start_time = time.time()
    service.close_batch(all_env_ids)
    end_time = time.time()
    close_time = end_time - start_time
    timings['close'].append(close_time)
    print(f"Closed {total_envs} environments in {close_time:.2f} seconds ({close_time/total_envs:.4f} seconds per env)")
    
    # Compute average step time
    avg_step_time = sum(timings['steps']) / len(timings['steps'])
    print(f"\nAverage step time: {avg_step_time:.2f} seconds total, {avg_step_time/total_envs:.4f} seconds per env")
    
    # Compile statistics
    stats = {}
    for operation, times in timings.items():
        if operation == 'steps':
            stats[operation] = {
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'min': min(times),
                'max': max(times),
                'per_env_mean': statistics.mean(times) / total_envs
            }
        else:
            if times:
                stats[operation] = {
                    'total': times[0],
                    'per_env': times[0] / total_envs
                }
    
    # Print summary statistics
    print("\n===== Summary Statistics =====")
    print(f"Total environments: {total_envs} (32 each of {', '.join(tasks)})")
    print(f"Environment creation: {stats['create_environments']['total']:.2f}s total, {stats['create_environments']['per_env']:.4f}s per env")
    print(f"Environment reset: {stats['reset']['total']:.2f}s total, {stats['reset']['per_env']:.4f}s per env")
    print(f"Environment steps: {stats['steps']['mean']:.2f}s mean, {stats['steps']['median']:.2f}s median, {stats['steps']['min']:.2f}s min, {stats['steps']['max']:.2f}s max, {stats['steps']['per_env_mean']:.4f}s per env")
    print(f"Compute rewards: {stats['compute_reward']['total']:.2f}s total, {stats['compute_reward']['per_env']:.4f}s per env")
    print(f"Get system prompts: {stats['get_system_prompts']['total']:.2f}s total, {stats['get_system_prompts']['per_env']:.4f}s per env")
    print(f"Close environments: {stats['close']['total']:.2f}s total, {stats['close']['per_env']:.4f}s per env")
    
    # Calculate overall throughput
    total_operations = total_envs * (1 + 1 + num_steps + 1 + 1 + 1)  # create, reset, 5 steps, reward, prompt, close
    total_time = (
        stats['create_environments']['total'] +
        stats['reset']['total'] +
        stats['steps']['mean'] * num_steps +
        stats['compute_reward']['total'] +
        stats['get_system_prompts']['total'] +
        stats['close']['total']
    )
    ops_per_second = total_operations / total_time
    print(f"\nTotal operations: {total_operations}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Operations per second: {ops_per_second:.2f}")

if __name__ == "__main__":
    for i in range(10):
        main()