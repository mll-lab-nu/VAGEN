# Quick Start

## Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU
- Conda (recommended)

### Setup

```bash
# Create conda environment
conda create -n vagen python=3.12 -y
conda activate vagen

# Clone repository
git clone https://github.com/mll-lab-nu/VAGEN.git
cd VAGEN
git submodule update --init --recursive

# Install VAGEN
pip install -e .

# Install VERL
cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .

# Additional dependencies
pip install "trl==0.26.2"
```

## Quick Start

### Training Paradigms

VAGEN supports two multi-turn training paradigms:

#### 1. Concatenated Training

All turns in a trajectory are concatenated into a single training instance. The context grows as the agent interacts with the environment:

```
sys + obs_0 + response_0 + obs_1 + response_1 + ...
```

**Run:**
```bash
cd VAGEN
wandb login
bash examples/sokoban/train_ppo_qwen25vl3b.sh
```

#### 2. Non-Concatenated Training

Each turn is treated as an independent training instance with its own context:

```
Turn 0: sys + obs_0 → response_0
Turn 1: sys + obs_1 → response_1
...
```

This paradigm uses custom GAE for cross-turn credit assignment.

**Run:**
```bash
cd VAGEN
wandb login
bash examples/sokoban/train_ppo_no_concat_qwen25vl3b.sh
```
