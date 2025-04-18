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

# Login to wandb for experiment tracking
wandb login
```

## Running Experiments

### Basic Approach
```
# run one of the experiment scripts
bash vagen/examples/frozen_lake_aico/run.sh  
bash vagen/examples/frozen_lake_trico/run.sh  
bash vagen/examples/sokoban_aico/run.sh 
bash vagen/examples/sokoban_trico/run.sh  
```

### Service Approach
1. Start the server process in a tmux session 
```
python vagen/server/server.py
```

2. After the server is running, open new terminal to run the training processe:
```
bash vagen/examples/frozen_lake_aico_service/run.sh
bash vagen/examples/svg_aico/run.sh
bash vagen/examples/svg_trico/run.sh
```

## Support Environment
- FrozenLake: A simple grid-based environment
- Sokoban: A visual puzzle-solving environment with box pushing
- SVG: An environment that generate svg code fot provided image. Supports reward model integration
- Navigation: An environment of visual navigation task for embodied AI

For information on creating new environment, please refer to our "[Create your Own Environment](create-env.md)" guide.

For information on creating service for new enviornment, please refer to our "[Create your Own Service](create-service.md) guide"
