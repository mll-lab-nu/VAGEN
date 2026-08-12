# Quick Start

## Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU
- Conda (recommended)

### Setup

```bash
conda create -n vagen python=3.12 -y
conda activate vagen

git clone --recursive https://github.com/mll-lab-nu/VAGEN.git
cd VAGEN
bash scripts/install.sh
```

`scripts/install.sh` is idempotent and verifies the result. `SKIP_ENGINE=1` skips the
vLLM/SGLang step if you already have them.

To do it by hand, the order matters — the engine stack first, then verl, then VAGEN:

```bash
git submodule update --init --recursive

cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .    # --no-deps: verl's pins would downgrade the stack above
cd ..
pip install -e .              # after verl, so VAGEN's transformers/torchao floors win
pip install "trl==0.26.2"
```

## Quick Start

### Training Paradigms

VAGEN supports three multi-turn training paradigms:

#### 1. Concatenated Training

All turns in a trajectory are concatenated into a single training instance. The context grows as the agent interacts with the environment:

```
sys + obs_0 + response_0 + obs_1 + response_1 + ...
```

**Run:**
```bash
cd VAGEN
wandb login
bash examples/train/sokoban/train_ppo_qwen25vl3b.sh
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
bash examples/train/sokoban/train_ppo_no_concat_qwen25vl3b.sh
```
