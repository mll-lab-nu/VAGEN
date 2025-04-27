import os
import sys
import unittest
from dataclasses import asdict

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vagen.env.alfworld.env import ALFWorldEnv
from vagen.env.alfworld.env_config import ALFWorldEnvConfig

def test_alfworld_env():
    """Basic test for ALFWorldEnv functionality."""
    print("=== Testing ALFWorldEnv ===")
    
    # Create configuration with path to your config file
    config = ALFWorldEnvConfig(
        alf_config_path="/workspace/VAGEN/vagen/env/alfworld/data/alf-config.py",  # Update with actual path
        max_actions_per_step=1,
        action_only_prompt=False,
        render_mode="text"  # Start with text mode for simplicity
    )
    
    try:
        # Test initialization
        print("1. Testing initialization...")
        env = ALFWorldEnv(config)
        print("✓ Initialization successful")
        
        # Test system prompt
        print("\n2. Testing system prompt...")
        prompt = env.system_prompt()
        print(f"System prompt (first 100 chars): {prompt[:100]}...")
        print("✓ System prompt generated")
        
        # Test reset
        print("\n3. Testing environment reset...")
        obs, info = env.reset()
        print("✓ Reset successful")
        print(f"Observation keys: {obs.keys()}")
        print(f"Observation preview: {obs['obs_str'][:100]}...")
        
        # Test available actions
        print("\n4. Available actions:")
        if env.prev_admissible_commands:
            for i, cmd in enumerate(env.prev_admissible_commands):
                print(f"  {i+1}. {cmd}")
        
        # Test step with a valid action
        print("\n5. Testing step with first available action...")
        if env.prev_admissible_commands and len(env.prev_admissible_commands) > 0:
            # Use the first available action
            action = env.prev_admissible_commands[0]
            action_json = '{"action": "' + action + '"}'
            
            print(f"Taking action: {action}")
            next_obs, reward, done, step_info = env.step(action_json)
            
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            print(f"Action valid: {step_info['metrics']['turn_metrics']['action_is_valid']}")
            print(f"Action effective: {step_info['metrics']['turn_metrics']['action_is_effective']}")
            print(f"New observation preview: {next_obs['obs_str'][:100]}...")
            print("✓ Step successful")
        else:
            print("No admissible commands available to test step")
        
        # Test compute_reward
        print("\n6. Testing compute_reward...")
        total_reward = env.compute_reward()
        print(f"Total reward: {total_reward}")
        print("✓ Compute reward successful")
        
        # Test close
        print("\n7. Testing environment close...")
        env.close()
        print("✓ Close successful")
        
        print("\n=== All tests passed! ===")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_alfworld_env()