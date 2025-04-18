# How to Create New Environments
> NOTICE: Once you've implemented your environment following this guide, VAGEN can be used directly, with the service layer being optional for training acceleration.

This guide explains how to create new environments for VAGEN's architecture. Creating custom environments is the foundation for building specialized VLM agent training scenarios. 

## Environment Structure Overview

VAGEN uses an object-oriented approach for environment management:
- 'BaseEnv': Abstract base class that defines the interface all environments must implement
- 'BaseEnvConfig': Configuration class for environment parameters
- 'Environment'-specific implementations (e.g., SvgEnv)

This architecture enables standardized interaction patterns while allowing for customization across different domains and tasks.

## Directory Structure
```
vagen/
├── env/
|   ├── create_dataset.py         # Store train/test data configs, not real data
│   ├── base/
│   │   ├── base_env.py           # Abstract base class defining the 
│   │   └── base_env_config.py    # Base configuration class for environments
│   ├── [your_env]/               # Your environment implementation
│       ├── env.py                # Your environment class
│       ├── env_config.py         # Your environment configuration
│       └── data/                 # Environment-specific resources (Optional)
|   
├── examples/
|   ├── [your_env]/
│       ├── env_config.yaml       # Your data&env config for create_dataset.py
│       ├── run.sh                # Script for training
```
## Creating a New Environment Step by Step

### Step 1: Create Environment Configuration

Create a new class that inherits from BaseEnvConfig. This class will define all parameters specific to your environment in `create_dataset.py` by combining your unique requirements in `env_config.yaml` and default requirements in `env_config.py`:

```python
@dataclass
class MyNewEnvConfig(BaseEnvConfig):
    """Configuration for My New Environment"""
    dataset_name: str = "path/to/dataset"
    data_dir: str = "vagen/env/my_new_env/data"
    seed: int = 42
    # Add your environment-specific parameters here
    
    def config_id(self) -> str:
        """Generate a unique identifier for this configuration"""
        return f"MyNewEnvConfig(dataset={self.dataset_name},seed={self.seed})"
```

### Step 2: Implement Environment Class

Create a new class that inherits from BaseEnv. This class must implement all required methods:
```python
from vagen.env.base.base_env import BaseEnv
from typing import Dict, Tuple
import random

class MyNewEnv(BaseEnv):
    def __init__(self, config):
        self.config = config
        self.done = False
        
    def step(self, llm_raw_response) -> Tuple[Dict, float, bool, Dict]:
        """Process an action from the LLM and return the next state"""
        parsed_action = parse_llm_raw_response(llm_raw_response)
        action_valid = parsed_action['is_valid']
        action_effective = action_valid  # Simplification for example
        
        # Update environment state based on action
        
        obs = {
            'obs_str': "Observation after action",
            'multi_modal_data': {}  # Add any images or audio here
        }
        
        reward = 0.0 if not action_valid else 0.5
        self.done = False  # Update based on task completion
        
        info = {
            "metrics": {
                'success': False,
                'action_is_effective': action_effective,
                'action_is_valid': action_valid,
            },
            "llm_raw_response": llm_raw_response,
            "llm_response": parsed_action,
        }
        
        return obs, reward, self.done, info
    
    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state"""
        if seed is not None:
            random.seed(seed)
            
        self.done = False
        
        obs = {
            'obs_str': "Initial observation text",
            'multi_modal_data': {}
        }
                
        return obs, info #(Optional, could be empty)
    
    def system_prompt(self) -> str:
        """Define the system prompt for the LLM"""
        return "You are an agent in the MyNewEnv environment. Your goal is to [describe task]."
    
    def compute_reward(self) -> float:
        """Calculate final episode reward"""
        return 0.0  # Calculate based on task completion
        
    def close(self):
        """Clean up any resources"""
        pass
```
### Step 3: Make Sure Input/Output Format Details

#### Environment Observations
Step Observations must follow this structure:
```python
{
    'obs_str': "Text observation with <image> or <audio> placeholders",
    'multi_modal_data': {
        '<image>': [image_data_1, image_data_2, ...],
        '<audio>': [audio_data_1, audio_data_2, ...],
    }
}
```
**Notice**: number of `image_place_holder(<image>)` in `obs_str` must match with number of `image_data` in `multi_modal_data`

#### Environment Info Dictionary
The info dictionary provides additional context and metrics:
```python
{
    "metrics": {
        'success': bool,  # Did the agent complete the task?
        'action_is_effective': bool,  # Was the action meaningful?
        'action_is_valid': bool,  # Was the action syntactically correct?
        # Add additional custom metrics
    },
    "llm_raw_response": str,  # Original response from LLM
    "llm_response": dict,  # Parsed response with structured format
}
```

### Step 4: Testing Your Environment

Create a basic script below your `env.py` to test your environment:
```python
# Create environment
config = MyNewEnvConfig()
env = MyNewEnv(config)

# Reset environment
obs, info = env.reset(seed=42)
print("Initial observation:", obs['obs_str'])

# Test step with mock LLM response
mock_llm_response = "Action1, Action2, Action3"
next_obs, reward, done, info = env.step(mock_llm_response)

print("Next observation:", next_obs['obs_str'])
print("Reward:", reward)
print("Done:", done)
print("Action valid:", info['metrics']['action_is_valid'])
print("Action effective:", info['metrics']['action_is_effective'])

# Clean up
env.close()
```

### Step 5: Integration with Service Layer (Optional)

For training acceleration and distributed processing, you can integrate your environment with the VAGEN service layer. This step is optional but recommended for large-scale training. See the "[Create your Own Service](create-service.md)" section for details.
