
# VAGEN-Lite

**VAGEN-Lite** is a lightweight implementation of **VAGEN** built on top of the **VERL agent-loop framework**

## Installation

```bash
conda create -n vagen python=3.12 -y
conda activate vagen

git clone https://github.com/mll-lab-nu/VAGEN.git
cd VAGEN
git checkout vagen-lite
git submodule update --init --recursive

pip install -e .

cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .

pip install "trl==0.26.2"
```


## Quick Start

VAGEN-Lite currently supports PPO / GRPO with two multi-turn training paradigms:

### 1. Multi-turn Concatenated Training

All turns in a trajectory are concatenated into a single training instance.

```bash
cd VAGEN
bash examples/sokoban/train_ppo_qwen25vl3b.sh
```

### 2. Multi-turn Non-Concatenated Training

Each trajectory is split into multiple turn-level training instances.

```bash
cd VAGEN
bash examples/sokoban/train_ppo_no_concat_qwen25vl3b.sh
```


## Customizing Your Environment

To train on your own environment, follow the steps below.

### 1. Create Your Environment Class

* Use `GymImageEnv` as the base class:

  * [`vagen/envs/gym_image_env.py`](vagen/envs/gym_image_env.py)
* Refer to Sokoban for a full implementation example:

  * [`vagen/envs/sokoban/sokoban_env.py`](vagen/envs/sokoban/sokoban_env.py)


### 2. Register the Environment

Add your environment entry to:

```yaml
vagen/configs/env_registry.yaml
```

### 3. Create Configuration Files

Prepare training and validation configs:

* `train.yaml`
* `val.yaml`

You can follow the Sokoban examples as templates:

* [`examples/sokoban/train_sokoban_vision.yaml`](examples/sokoban/train_sokoban_vision.yaml)
* [`examples/sokoban/val_sokoban_vision.yaml`](examples/sokoban/val_sokoban_vision.yaml)

---

### 4. Create a Training Script

Write your training script based on:

* [`examples/sokoban/train_ppo_qwen25vl3b.sh`](examples/sokoban/train_ppo_qwen25vl3b.sh)

