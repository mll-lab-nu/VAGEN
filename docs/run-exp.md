# Run Experiments

This guide provides instructions for running experiments with VAGEN, a multi-turn reinforcement learning framework for training VLM Agents. VAGEN leverages the TRICO algorithm to efficiently train VLMs for visual agentic tasks.

## Prerequisites

Before running experiments, ensure you have set up the environment properly:

```bash
# Create a new conda environment
conda create -n vagen python=3.10 -y
conda activate vagen

# Install verl
git clone https://github.com/JamesKrW/verl.git
cd verl
pip install -e .
cd ../

# Install VAGEN
git clone https://github.com/RAGEN-AI/VAGEN.git
cd VAGEN
bash scripts/install.sh

# go to release branch of verl
cd ../verl
git checkout release
cd ../VAGEN

# Login to wandb for experiment tracking
wandb login
```

## Running Experiments
### Basic Approach
The simplest way to reproduce our experiments is to use the provided scripts:
```
# run one of the experiment scripts
bash vagen/examples/release_experiments/grpo.sh  # rico-grpo
bash vagen/examples/release_experiments/mask_gae_mask_loss_turnwise_reward_bi_level.sh  # trico
bash vagen/examples/release_experiments/mask_gae_mask_loss.sh  # aico
```

**Handling Training Instabilities**
Each run takes approximately 4 hours to reach 150 steps on 4 H100s. You can decrease testing frequency to speed up training. Note that training might be unstable due to loss spikes; we recommend restoring from the latest checkpoint when encountering such cases.

### Service Approach
The simplest way to reproduce our experiments is to use the provided scripts:
```
# EXPERIMENT TOBE ADDED
```

## Support Environments
The following environments are currently registered:
> NOTICE: SVG and Navigation envs are commented in `env/__init__.py`for better package management. Please comment them out for future use, and refer `vagen/env/README.md` to install dependencies

- FrozenLake: A simple grid-based environment
- Sokoban: A visual puzzle-solving environment with box pushing
- SVG: An environment that generate svg code fot provided image. Supports reward model integration
- Navigation: An environment of visual navigation task for embodied AI

For information on creating new environment services, please refer to our "[Create your Own Environment](create-env.md)" guide.