# How to Create New Services

This guide explains how to create new environment services for VAGEN's service-based architecture. The service architecture provides a standardized way to manage multiple environments for VLM agent training with enhanced scalability and flexibility.

## Service Architecture Overview

VAGEN uses a client-server architecture for environment management:

- `BaseService`: Abstract base class that defines the interface all services must implement
- `BatchEnvClient`: Client that communicates with environment servers (fixed)
- `BatchEnvServer`: Server implementation for hosting environments (fixed)

This architecture enables efficient parallel processing across distributed systems and seamless integration of both rule-based rewards and reward models.

## Directory Structure

```
vagen/
├── env/
│   ├── base_service.py        # Abstract base class defining the service interface
│   ├── client.py              # Client for interacting with environment server
│   ├── server.py              # Server implementation for hosting environments
│   └── REGISTERED_ENV         # Registry mapping environment names to services
├── utils/
│   ├── serial.py              # Handle observation serialization from service to client
```
## Component Hierarchy
```
BaseService (ABC)
├── **Batch Methods**
    ├── create_environments_batch()
    ├── reset_batch()
    ├── step_batch()
    ├── compute_reward_batch()
    ├── get_system_prompts_batch()
    └── close_batch()

BatchEnvClient
├── **HTTP Communication**
├── **Batch Methods**
└── **Convenience Methods**

BatchEnvServer
├── **Service Management**
├── **Request Routing**
├── **Batch Method Implementation**
└── **Server Management**
```

## Creating a New Service Step by Step

### Step 1: Inherit from BaseService

Create a new class that inherits from `BaseService`. This class must implement all required methods for interacting with environments:

```python
class MyNewService(BaseService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your environment-specific components here
        
    def create_environments_batch(self, ids2configs: Dict[str, Any]) -> None:
        # Initialize batch of environments
        # Return initialization status
        
    def reset_batch(self, ids2seeds: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        # Reset environments and return observations
        
    def step_batch(self, ids2actions: Dict[str, Any]) -> Dict[str, Tuple[Dict, float, bool, Dict]]:
        # Process actions and return (observations, dones)
        
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[str, float]:
        # Calculate rewards for each environment
        # Return rewards and any additional info
        
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[str, str]:
        # Return system prompts for each environment
        
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        # Clean up resources for environments
```
### Step 2: Observation Serialization
When passing observations between the service and client, ensure proper serialization:
```
from vagen.utils.serial import serialize_observation, deserialize_observation

# On the service side
## at the end of reset_batch()
serialized_obs = serialize_observation(original_observation)
## at the end of step_batch()
serialized_step = serialize_step_result(observation, reward, done, info)
```

### Step 3: Register your Environment
Register your env in `env/__init__.py`
```
from vagen.env.NEW_service import MyNewService
# Register your service
REGISTERED_ENV["my_new_env"] = MyNewService
```

### Step 4: Define your env config and script
> Please refer to "[Configuration](../configs/general-config.md)"
Define your env config and running script in `examples/`


Please refer to `Frozenlake/service.py` for better service structure understanding