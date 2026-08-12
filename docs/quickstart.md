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

`scripts/install.sh` is idempotent and verifies the result. It installs vLLM by default;
`BACKEND=sglang bash scripts/install.sh` picks SGLang instead, and `SKIP_ENGINE=1` skips
the engine if you already have one. Install **one** engine per environment — each pins a
different `flashinfer` patch version, so pip refuses the two together.

To do it by hand, the order matters — VAGEN with its engine first, then verl:

```bash
git submodule update --init --recursive

pip install -e ".[vllm]"           # or ".[sglang]" -- pick one, never both
pip install --no-deps -e ./verl    # --no-deps: verl's pins would undo the line above
pip install accelerate codetiming datasets dill hydra-core numpy pandas peft pyarrow \
            pybind11 pylatexenc ray tensordict torchdata wandb
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
