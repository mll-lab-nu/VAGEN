# Quick Start

## Installation

### Prerequisites

- Python 3.12 exactly — `scripts/install.sh` checks for it and stops otherwise
- CUDA 13 runtime and driver; a CUDA 13 toolkit with `nvcc` is needed only when a
  compiled dependency has no matching wheel
- **GPUs.** The shipped scripts ask for `trainer.n_gpus_per_node` of 4 (most), 8
  (navigation, spatial_gym, and the state-reward judge), 2, or 1 — check the script you
  intend to run. `default_gae` and `ppo` also train a critic the size of the actor, so a
  3B run holds two 3B models plus the rollout engine; A100-80G class cards are what these
  were developed on. Lower `n_gpus_per_node` and `data.train_batch_size` together.
- Conda (recommended)

### Setup

```bash
conda create -n vagen python=3.12 -y
conda activate vagen

git clone --branch dev/bi_level https://github.com/JamesKrW/VAGEN.git
cd VAGEN
bash scripts/install.sh
```

`scripts/install.sh` initializes only the required `verl` submodule. It installs Torch
2.11.0 / SGLang 0.5.13 / Transformers 5.8.1 by default.
`BACKEND=vllm bash scripts/install.sh` selects vLLM; `SKIP_ENGINE=1` keeps and verifies an
existing engine. Install **one** engine per environment because the backends require
different `flashinfer` builds.

For a manual SGLang install, preserve the same two-pass order (the second pass reuses
the installed Torch while building `causal-conv1d`):

```bash
git submodule update --init -- verl
pip install -r requirements/locks/sglang-cu130.txt
pip install --no-build-isolation --no-deps \
    -r requirements/locks/sglang-cu130-post.txt
pip install --no-deps -e . -e ./verl
pip install codetiming dill pybind11 pylatexenc fire ninja cachetools \
    gym-sokoban gymnasium "uvicorn<0.41"
```

On hosts that cannot fetch GitHub release assets, `causal-conv1d` builds locally from
PyPI. Ensure `CUDA_HOME` points to a toolkit with `nvcc`; use `MAX_JOBS=8` if RAM is tight.

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
bash examples/train/sokoban/train_default_gae_qwen25vl3b.sh
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
