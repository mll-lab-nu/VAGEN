# Quick Start

## Installation

### Prerequisites

- Python 3.12 exactly — `scripts/install.sh` checks for it and stops otherwise
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

Because an episode is now spread over several rows, this **requires** a trajectory-level
advantage estimator. verl's own estimators score one row at a time and would treat each
turn as a complete episode; the trainer refuses that combination at startup rather than
training on it:

```
ValueError: algorithm.adv_estimator=... scores one row at a time, but
trainer.harness=... splits an episode across rows
```

**Run:**
```bash
cd VAGEN
wandb login
bash examples/train/sokoban/train_ppo_no_concat_qwen25vl3b.sh
```

#### 3. Compaction

The conversation is summarised and reopened when it grows past `trainer.compact_budget`,
so a long episode keeps its context without one row having to hold all of it. Like
non-concatenated, it splits an episode across rows and so needs a trajectory estimator.

```
conversation 1: sys + obs_0 + resp_0 + obs_1 + resp_1 + <summary>
conversation 2: sys + <summary> + obs_2 + resp_2 + ...
```

**Run:**
```bash
cd VAGEN
wandb login
bash examples/train/sokoban/train_default_gae_compact_qwen25vl3b.sh
```

!!! tip "Size `compact_budget` against `max_turns`"
    If a whole episode fits inside one conversation, compaction never fires and the run is
    silently `concat` under another name. See [Configuration](configuration.md).

All three are selected by one key, `trainer.harness`, and a custom policy can be plugged in
without editing the trainer — see [Configuration](configuration.md).
