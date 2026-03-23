<h1 align="center">MindCube RL Training with VAGEN</h1>

<p align="center">
  RL fine-tuning for spatial reasoning on the <a href="https://mind-cube.github.io/">MindCube</a> benchmark, built on the <a href="https://github.com/RAGEN-AI/VAGEN">VAGEN</a> framework.
</p>

## Overview

This branch contains the code for training VLMs on the MindCube crossview spatial reasoning task using reinforcement learning (GRPO). The model is given 4 images of a 3D scene and must answer questions about spatial relationships.

## Installation

```bash
# Create a new conda environment
conda create -n vagen python=3.10 -y
conda activate vagen

# verl (MindCube branch)
git clone -b MindCube https://github.com/JamesKrW/verl.git
cd verl
pip install -e .
cd ../

# vagen
git clone https://github.com/RAGEN-AI/VAGEN.git
cd VAGEN
bash scripts/install.sh
```

## MindCube: Spatial Reasoning with RL

### Setup

**1. Install crossview environment dependencies:**
```bash
bash scripts/install.sh  # ensure crossview is uncommented
```

**2. Download MindCube dataset:**

Download the dataset from HuggingFace: https://huggingface.co/datasets/MLL-Lab/MindCube

Place the RL training data (`.jsonl` files) under `vagen/env/crossview/MindCube_RL_Data/`, and create a symlink for the image data:
```bash
ln -s /path/to/your/MindCube/images vagen/env/crossview/other_all_image
```

**3. Prepare SFT checkpoints (for SFT-initialized experiments):**

For experiments that start from an SFT checkpoint, set the `SFT_CKPT_DIR` environment variable to your SFT results directory:
```bash
export SFT_CKPT_DIR=/path/to/your/sft/results
```

The expected checkpoint structure is:
```
$SFT_CKPT_DIR/
  ff_rsn/checkpoint-57           # baseRL SFT (think+answer format)
  aug_cgmap_ffr_out/checkpoint-45  # cogmap+reasoning SFT (augmented cogmap)
  plain_cgmap_ffr_out/checkpoint-50 # cogmap+reasoning_plain SFT (plain cogmap)
```

### Training

Three training strategies are supported, each with a base (from scratch) and SFT-initialized variant:

| Experiment | Description | Script |
|---|---|---|
| `baseRL` | Standard RL with think+answer format | `scripts/examples/crossview/baseRL/run_tmux.sh` |
| `cogmap_reasoning` | RL with augmented cognitive map | `scripts/examples/crossview/cogmap_reasoning/run_tmux.sh` |
| `cogmap_reasoning_plain` | RL with plain cognitive map | `scripts/examples/crossview/cogmap_reasoning_plain/run_tmux.sh` |
| `baseRL_sft` | Same as baseRL but starts from SFT checkpoint | `scripts/examples/crossview/baseRL_sft/run_tmux.sh` |

**Run a single experiment (interactive, 2-GPU):**
```bash
bash scripts/examples/crossview/baseRL/run_tmux.sh
```

**Run all SFT-initialized experiments sequentially (4-GPU relay):**
```bash
export SFT_CKPT_DIR=/path/to/sft/results
nohup bash scripts/run_all_crossview.sh > logs/run_all.log 2>&1 &
```

### Evaluation

To evaluate the best checkpoint of each experiment on the TinyBench (1050 samples):
```bash
export SFT_CKPT_DIR=/path/to/sft/results
nohup bash scripts/eval_all_crossview.sh > logs/eval_all.log 2>&1 &
```

Inference results are saved to `eval_results/{experiment_name}/step_{N}_responses.jsonl`.
