# How to Create Services

This guide explains how to create service layers for VAGEN environments using **Blackjack** as an example. Services enable batch processing and distributed training by managing multiple environment instances simultaneously.

## Directory Structure

```
vagen/env/
├── __init__.py               # Main environment registry (REGISTERED_ENV)
└── blackjack/
    ├── service.py            # BlackjackService - batch environment manager
    ├── service_config.py     # BlackjackServiceConfig - service configuration
    ├── env.py                # BlackjackEnv (dependency)
    ├── env_config.py         # BlackjackEnvConfig (dependency)
    └── __init__.py           # Environment info definition (BLACKJACK_ENV_INFO)
```

**File Responsibilities:**
- `blackjack/service_config.py`: Configuration parameters for batch processing and service management
- `blackjack/service.py`: Implementation of batch environment operations following BaseService interface
- `blackjack/__init__.py`: Defines BLACKJACK_ENV_INFO with service classes included
- `env/__init__.py`: Main registry where BLACKJACK_ENV_INFO gets added to REGISTERED_ENV

## Component Hierarchy

```
VAGEN Service Architecture
│
├── Infrastructure Layer (vagen/server/)
│   ├── BatchEnvServer                  # Flask server hosting all services
│   │   ├── Service Management          # Route requests to appropriate services
│   │   ├── HTTP API Endpoints          # RESTful interface (/batch/*, /environments)
│   │   ├── Request Serialization       # Handle complex observation data
│   │   └── Multi-Service Coordination  # Manage different environment types
│   │
│   └── BatchEnvClient                  # Client for connecting to servers
│       ├── HTTP Communication          # Connect to remote BatchEnvServer
│       ├── Batch Method Wrappers       # Same interface as BaseService
│       ├── Automatic Serialization     # Handle observation deserialization
│       └── Convenience Methods         # Single-environment operations
│
├── Service Interface Layer (vagen/env/base/)
│   └── BaseService (Abstract Interface)
│       ├── create_environments_batch() # Create multiple env instances
│       ├── reset_batch()              # Reset multiple environments
│       ├── step_batch()               # Execute actions across environments
│       ├── compute_reward_batch()     # Calculate final rewards
│       ├── get_system_prompts_batch() # Get prompts for environments
│       └── close_batch()              # Clean up environment resources
│
└── Implementation Layer (vagen/env/your_env/)
    └── YourService (e.g., BlackjackService)
        ├── Environment Instance Pool   # Manage multiple env instances
        ├── Configuration Management    # Handle per-environment configs
        ├── State Tracking             # Monitor environment states
        └── Optional: LLM-as-Judge     # State reward evaluation
```

## Understanding BaseService Interface

Services inherit from `BaseService`, which defines batch operations for environment management. Here's what each required method does:

### Core Methods Overview

**`create_environments_batch(ids2configs)`** - Initialize environment pool
- Takes a dictionary mapping environment IDs to their configurations
- Creates multiple environment instances with different settings
- Each environment can have unique parameters (render mode, difficulty, etc.)

**`reset_batch(ids2seeds)`** - Reset multiple environments
- Resets specified environments with given seeds for reproducibility
- Returns initial observations for all environments
- Handles serialization for network transfer

**`step_batch(ids2actions)`** - Execute actions across environments
- Takes LLM responses for multiple environments simultaneously
- Processes each action in its respective environment
- Returns observations, rewards, done flags, and info for all environments

**`compute_reward_batch(env_ids)`** - Calculate final rewards
- Computes episode-ending rewards for specified environments
- Usually returns 0.0 unless you need custom final scoring

**`get_system_prompts_batch(env_ids)`** - Retrieve system prompts
- Gets the system prompt for each specified environment
- Used by training system to set up LLM context

**`close_batch(env_ids)`** - Clean up resources
- Closes specified environments and frees resources
- If no IDs provided, closes all managed environments

## Implementation Components

### 1. Service Configuration (`service_config.py`)
Defines service-level parameters:

```python
from vagen.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass

@dataclass
class BlackjackServiceConfig(BaseServiceConfig):
    # Inherited: max_workers = 10
    
    # Optional state reward features
    use_state_reward: bool = False
```

### 2. Service Implementation (`service.py`)
The main service class implementing BaseService:

```python
from typing import Dict, List, Tuple, Optional, Any
from vagen.env.base.base_service import BaseService
from vagen.server.serial import serialize_observation
from .env import BlackjackEnv
from .env_config import BlackjackEnvConfig

class BlackjackService(BaseService):
    def __init__(self, config: BaseServiceConfig):
        self.environments = {}    # env_id -> BlackjackEnv instance
        self.env_configs = {}     # env_id -> BlackjackEnvConfig
        self.config = config
        
        # Optional: Initialize state reward tracking
        if self.config.use_state_reward:
            from vagen.env.utils.top_string_tracker import TopKStringTracker
            self.top_strings_tracker = TopKStringTracker(self.config.top_strings_m)
    
    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        """Create multiple Blackjack environments with different configurations"""
        for env_id, config in ids2configs.items():
            # Extract environment-specific config
            env_config_dict = config.get('env_config', {})
            env_config = BlackjackEnvConfig(**env_config_dict)
            
            # Create environment instance
            env = BlackjackEnv(env_config)
            
            # Store in service pools
            self.environments[env_id] = env
            self.env_configs[env_id] = env_config
    
    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        """Reset multiple environments and return serialized observations"""
        results = {}
        
        for env_id, seed in ids2seeds.items():
            env = self.environments[env_id]
            observation, info = env.reset(seed=seed)
            
            # Serialize for network transfer
            serialized_observation = serialize_observation(observation)
            results[env_id] = (serialized_observation, info)
        
        return results
    
    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        """Execute LLM actions across multiple environments"""
        results = {}
        
        for env_id, action in ids2actions.items():
            env = self.environments[env_id]
            observation, reward, done, info = env.step(action)
            
            # Serialize observation for network transfer
            serialized_observation = serialize_observation(observation)
            results[env_id] = (serialized_observation, reward, done, info)
        
        return results
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[Any, float]:
        """Get final rewards for multiple environments"""
        results = {}
        for env_id in env_ids:
            env = self.environments[env_id]
            results[env_id] = env.compute_reward()
        return results
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[Any, str]:
        """Get system prompts for multiple environments"""
        results = {}
        for env_id in env_ids:
            env = self.environments[env_id]
            results[env_id] = env.system_prompt()
        return results
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """Clean up multiple environments"""
        if env_ids is None:
            env_ids = list(self.environments.keys())
        
        for env_id in env_ids:
            if env_id in self.environments:
                self.environments[env_id].close()
                del self.environments[env_id]
                del self.env_configs[env_id]
```

## Service Registration

Update your `blackjack/__init__.py` to include service information:

```python
from .env import BlackjackEnv
from .env_config import BlackjackEnvConfig
from .service import BlackjackService
from .service_config import BlackjackServiceConfig

# Complete registration with service support
BLACKJACK_ENV_INFO = {
    "env_cls": BlackjackEnv,
    "config_cls": BlackjackEnvConfig,
    "service_cls": BlackjackService,
    "service_config_cls": BlackjackServiceConfig,
    "description": "Classic Blackjack with batch processing support"
}
```

Then add to the main registry in `vagen/env/__init__.py`:

```python
from .blackjack import BLACKJACK_ENV_INFO

REGISTERED_ENV = {
    # ... other environments ...
    "blackjack": BLACKJACK_ENV_INFO,
    # ... other environments ...
}
```

## Server Configuration

Add your service to the server config in `vagen/server/config/server.yaml`:

```yaml
blackjack:
  max_workers: 48
  use_state_reward: false
```

## Advanced Features: State Rewards (Optional)

If you want LLM-as-judge evaluation, add these methods to your service:

```python
# Add to your service class if using state rewards

@service_state_reward_wrapper  # Decorator enables LLM judging
def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
    # Same implementation as above
    # Decorator automatically calls reward methods when needed
    pass

def gen_reasoning_prompt(self, content, **kwargs) -> str:
    """Generate prompt for LLM judge to evaluate reasoning"""
    return f"""Evaluate this Blackjack decision reasoning:
{content}

Rate the strategic quality from 0.0 to 1.0."""

def calculate_reasoning_reward(self, **kwargs) -> float:
    """Calculate reward based on LLM judge evaluation"""
    response = kwargs.get("response")
    content = kwargs.get("content")
    
    # Parse LLM judge response
    try:
        reward = response.get("quality", 0.0) if isinstance(response, dict) else 0.5
    except:
        reward = 0.0
    
    # Anti-repetition penalty
    if hasattr(self, 'top_strings_tracker'):
        top_k_strings = self.top_strings_tracker.get_top_k(self.config.top_strings_k)
        if content in top_k_strings and reward < 0.6:
            return -0.1  # Penalty for repetitive low-quality responses
    
    return reward
```

## Usage Patterns

### Direct Service Usage
```python
from vagen.env.blackjack import BlackjackService, BlackjackServiceConfig

# Create service with 4 environments
config = BlackjackServiceConfig(max_workers=4)
service = BlackjackService(config)

# Create environments with different settings
ids2configs = {
    "easy": {"env_config": {"natural": False, "render_mode": "text"}},
    "hard": {"env_config": {"natural": True, "render_mode": "vision"}}
}
service.create_environments_batch(ids2configs)

# Batch operations
observations = service.reset_batch({"easy": 42, "hard": 123})
results = service.step_batch({
    "easy": "<answer>Stand</answer>", 
    "hard": "<answer>Hit</answer>"
})

service.close_batch()
```

### Client-Server Usage
```python
from vagen.server.client import BatchEnvClient

# Connect to remote service
client = BatchEnvClient(base_url="http://localhost:5000")

# Same interface as direct service
client.create_environments_batch(ids2configs)
observations = client.reset_batch(ids2seeds)
results = client.step_batch(ids2actions)
client.close_batch()
```

## Testing Your Service

```python
# Test service functionality
config = BlackjackServiceConfig()
service = BlackjackService(config)

# Test environment creation
ids2configs = {"test_env": {"env_config": {"render_mode": "text"}}}
service.create_environments_batch(ids2configs)

# Test batch operations
obs = service.reset_batch({"test_env": 42})
results = service.step_batch({"test_env": "<answer>Hit</answer>"})

print("Service test passed:", len(obs) == 1 and len(results) == 1)
service.close_batch()
```

The service layer enables efficient scaling from single environments to hundreds of parallel instances, essential for large-scale LLM training while maintaining clean separation from your core game logic.