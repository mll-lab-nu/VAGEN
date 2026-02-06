# Custom Environment

This guide explains how to create your own environment for VAGEN.

## Overview

VAGEN environments extend `GymImageEnv`, which provides a simple API for multi-modal (image + text) observations.

## Step 1: Create Your Environment Class

Use `GymImageEnv` as the base class:

* [`vagen/envs/gym_image_env.py`](https://github.com/RAGEN-AI/VAGEN/blob/main/vagen/envs/gym_image_env.py)

Refer to Sokoban for a full implementation example:

* [`vagen/envs/sokoban/sokoban_env.py`](https://github.com/RAGEN-AI/VAGEN/blob/main/vagen/envs/sokoban/sokoban_env.py)

### Base Class

```python
from vagen.envs.gym_image_env import GymImageEnv
from typing import Dict, Any, Tuple
from PIL import Image

class MyEnv(GymImageEnv):
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        # Your initialization code
```

### Required Methods

#### `reset() -> Tuple[Dict, Dict]`

Reset the environment and return the initial observation.

```python
def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Reset your environment state

    obs = {
        "obs_str": "You see: <image>\nWhat action will you take?",
        "multi_modal_input": {
            "<image>": [self.render()]  # List of PIL Images
        }
    }

    info = {
        "success": False  # Used for logging
    }

    return obs, info
```

#### `step(action: str) -> Tuple[Dict, float, bool, bool, Dict]`

Execute an action and return the result.

```python
def step(self, action: str) -> Tuple[Dict, float, bool, bool, Dict]:
    # Parse and execute the action
    reward = self._execute_action(action)
    done = self._check_done()
    truncated = self._check_truncated()

    obs = {
        "obs_str": f"Action result: <image>",
        "multi_modal_input": {
            "<image>": [self.render()]
        }
    }

    info = {
        "success": self._check_success()
    }

    return obs, reward, done, truncated, info
```

#### `system_prompt() -> str`

Return the system prompt for the agent.

```python
def system_prompt(self) -> str:
    return """You are an agent in a grid world.
Available actions: up, down, left, right
Respond with your chosen action."""
```

## Observation Format

### With Images

```python
obs = {
    "obs_str": "Description with <image> placeholder",
    "multi_modal_input": {
        "<image>": [pil_image_1, pil_image_2, ...]
    }
}
```

- The number of `<image>` placeholders in `obs_str` must match the number of images
- Images are PIL.Image objects

### Without Images (Text-only)

```python
obs = {
    "obs_str": "Text-only observation"
}
```

## Step 2: Register the Environment

Add your environment entry to [`vagen/configs/env_registry.yaml`](https://github.com/RAGEN-AI/VAGEN/blob/main/vagen/configs/env_registry.yaml):

```yaml
env_registry:
  Sokoban: vagen.envs.sokoban.sokoban_env.Sokoban
  FrozenLake: vagen.envs.frozenlake.frozenlake_env.FrozenLake
  MyEnv: vagen.envs.myenv.my_env.MyEnv  # Add this line
```

## Step 3: Create Configuration Files

Prepare training and validation configs:

* `train.yaml`
* `val.yaml`

You can follow the Sokoban examples as templates:

* [`examples/sokoban/train_sokoban_vision.yaml`](https://github.com/RAGEN-AI/VAGEN/blob/main/examples/sokoban/train_sokoban_vision.yaml)
* [`examples/sokoban/val_sokoban_vision.yaml`](https://github.com/RAGEN-AI/VAGEN/blob/main/examples/sokoban/val_sokoban_vision.yaml)

### Example Training Config

```yaml
env_name: MyEnv
env_config:
  grid_size: 8
  max_steps: 50

num_envs: 128  # Number of parallel environments
```

## Step 4: Create a Training Script

Write your training script based on [`examples/sokoban/train_ppo_qwen25vl3b.sh`](https://github.com/RAGEN-AI/VAGEN/blob/main/examples/sokoban/train_ppo_qwen25vl3b.sh):

```bash
#!/bin/bash
set -x

PROJECT_NAME="verl_vagen"
EXPERIMENT_NAME="myenv_ppo"

BASEDIR=$(pwd)
EXPERIMENT_DIR=${BASEDIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}
DATASET_TRAIN=path/to/train.yaml
DATASET_VAL=path/to/val.yaml

mkdir -p ${EXPERIMENT_DIR}

python3 -m vagen.main_ppo \
    --config-path=${BASEDIR}/vagen/configs \
    --config-name='vagen_multiturn' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=${DATASET_VAL} \
    data.train_batch_size=128 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    # ... other parameters
```

## Tips

1. **Action Parsing**: Design a robust action parser that handles various agent output formats
2. **Reward Shaping**: Start with sparse rewards (success/failure), add shaping if needed
3. **Max Steps**: Set appropriate episode length to balance exploration and training efficiency
4. **Image Size**: Smaller images (e.g., 84x84, 128x128) train faster but may lose detail
