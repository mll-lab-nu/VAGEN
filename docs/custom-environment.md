# Custom Environment

This guide explains how to create your own environment for VAGEN.

## Overview

VAGEN environments extend `GymImageEnv`, which provides a simple API for multi-modal (image + text) observations.

## Step 1: Create Your Environment Class

Use `GymImageEnv` as the base class:

* [`vagen/envs/gym_image_env.py`](https://github.com/RAGEN-AI/VAGEN/blob/main/vagen/envs/gym_image_env.py)

Refer to Sokoban for a full implementation example:

* [`vagen/envs/sokoban/sokoban_env.py`](https://github.com/RAGEN-AI/VAGEN/blob/main/vagen/envs/sokoban/sokoban_env.py)

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

## Step 4: Create a Training Script

Write your training script based on [`examples/sokoban/train_ppo_qwen25vl3b.sh`](https://github.com/RAGEN-AI/VAGEN/blob/main/examples/sokoban/train_ppo_qwen25vl3b.sh)
