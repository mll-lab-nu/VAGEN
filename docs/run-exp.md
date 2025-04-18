# Installation and Run Experiments

This guide provides instructions for installing environment and running experiments with VAGEN, a multi-turn reinforcement learning framework for training VLM Agents. VAGEN leverages the TRICO algorithm to efficiently train VLMs for visual agentic tasks.

## Installation

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
The simplest way to run an experiments is to use the provided scripts:
```
# run one of the experiment scripts
bash vagen/examples/frozen_lake_aico/run.sh  #aico-frozenlake
bash vagen/examples/frozen_lake_trico/run.sh  #trico-frozenlake
```

**Handling Training Instabilities**
Each run takes approximately 4 hours to reach 150 steps on 4 H100s. You can decrease testing frequency to speed up training. Note that training might be unstable due to loss spikes; we recommend restoring from the latest checkpoint when encountering such cases.

### Service Approach
The simplest way to reproduce our experiments is to use the provided scripts:
```
bash vagen/examples/frozen_lake_aico_service/run.sh  # aico-frozenlake
```

## Support Environments
The following environments are currently registered:
> NOTICE: SVG and Navigation envs are commented in `env/__init__.py`for better package management. Please comment them out for future use, and refer `vagen/env/README.md` to install dependencies

- FrozenLake: A simple grid-based environment
- Sokoban: A visual puzzle-solving environment with box pushing
- SVG: An environment that generate svg code fot provided image. Supports reward model integration
- Navigation: An environment of visual navigation task for embodied AI

For information on creating new environment, please refer to our "[Create your Own Environment](create-env.md)" guide.

For information on creating service for new enviornment, please refer to our "[Create your Own Service](create-service.md) guide"