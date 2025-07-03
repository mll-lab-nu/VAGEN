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
# Login to wandb
wandb login

# You can run different environments and algorithms:
bash scripts/examples/masked_grpo/frozenlake/grounding_worldmodeling/run_tmux.sh
bash scripts/examples/finegrained/sokoban/grounding_worldmodeling/run_tmux.sh
bash scripts/examples/masked_turn_ppo/frozenlake/grounding_worldmodeling/run_tmux.sh

# Use Visual Reasoning Reward
# Setup OPENAI_API_KEY in the Environment
bash scripts/examples/state_reward_finegrained/sokoban/grounding_worldmodeling/run_tmux.sh
```

## Support Environment
- FrozenLake: A simple grid-based environment
- Sokoban: A visual puzzle-solving environment with box pushing
- SVG: An environment that generate svg code fot provided image. Supports reward model integration
- Navigation: An environment of visual navigation task for embodied AI
- Primitive-skill: An environment of primitive skill for embodied AI
- Blackjack: A simple card game environment

For information on creating new environment, please refer to our "[Create your Own Environment](envs/create-env.md)" guide.

For information on creating service for training based on your new environment, please refer to our "[Create your Own Service](envs/create-service.md) guide"
