# SpatialGym (Theory of Space Environment)

SpatialGym is a spatial reasoning environment built on top of the [Theory of Space](https://github.com/mll-lab-nu/Theory-of-Space.git) framework.

## Installation

1. Download the dataset:

```bash
cd VAGEN

huggingface-cli download yw12356/spatial_gym_dataset \
  --repo-type dataset \
  --local-dir vagen/envs/spatial_gym/room_data
```

2. Install the additional dependencies:

```bash
pip install -r vagen/envs/spatial_gym/requirements.txt
```

## Evaluation

Run evaluation with OpenAI-compatible backends:

```bash
python -m vagen.evaluate.run_eval --config examples/evaluate/spatial_gym/config.yaml
```

Available eval configs:
- `config.yaml` - 2-room active exploration 

## Training

GRPO training
```bash
bash examples/spatial_gym/train_grpo_qwen25vl3b.sh
bash examples/spatial_gym/train_grpo_qwen25vl7b.sh
```

PPO training

```bash
bash examples/spatial_gym/train_ppo_qwen25vl3b.sh
bash examples/spatial_gym/train_ppo_qwen25vl7b.sh
```
