# Custom Environment

This guide explains how to create your own environment for VAGEN.

## Overview

Place each concrete environment in `vagen/envs/<environment>/`. Shared environment
contracts and protocol helpers belong under `vagen/envs/_common/`; an implementation must
not place its environment-specific control flow there.

Image-capable environments extend `GymImageEnv`, imported from the public facade:

```python
from vagen.envs import GymImageEnv
```

## Step 1: Create Your Environment Class

Use `GymImageEnv` as the base class. Its implementation lives at:

* [`vagen/envs/_common/gym_image.py`](https://github.com/mll-lab-nu/VAGEN/blob/main/vagen/envs/_common/gym_image.py)

Refer to Sokoban for a full implementation example:

* [`vagen/envs/sokoban/sokoban_env.py`](https://github.com/mll-lab-nu/VAGEN/blob/main/vagen/envs/sokoban/sokoban_env.py)

Implement `system_prompt()`, `reset(seed)`, `step(action_str)`, and `close()`. Return image
observations as `{"obs_str": "... <image> ...", "multi_modal_input": {"<image>": [image]}}`;
text-only observations need only `obs_str`.

For response parsing, reuse `vagen.envs._common.response_format`. Structured WM output is
always `<perception><reasoning><prediction><answer>` in that order, while free-think output
is `<think><answer>`. Keep only the environment's action vocabulary and reward semantics in
the concrete package.

## Step 2: Register the Environment

Add your environment entry to [`vagen/configs/env_registry.yaml`](https://github.com/mll-lab-nu/VAGEN/blob/main/vagen/configs/env_registry.yaml):

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

* [`examples/train/sokoban/train_sokoban_vision.yaml`](https://github.com/mll-lab-nu/VAGEN/blob/main/examples/train/sokoban/train_sokoban_vision.yaml)
* [`examples/train/sokoban/val_sokoban_vision.yaml`](https://github.com/mll-lab-nu/VAGEN/blob/main/examples/train/sokoban/val_sokoban_vision.yaml)

## Step 4: Create a Training Script

Write your training script based on [`examples/train/sokoban/train_default_gae_qwen25vl3b.sh`](https://github.com/mll-lab-nu/VAGEN/blob/main/examples/train/sokoban/train_default_gae_qwen25vl3b.sh)

Use the same environment config for evaluation. Training and evaluation share the harness
and environment construction path, so a separate evaluation-only parser or runner is not
needed.
