# SpatialGym (Theory of Space Environment)

SpatialGym is a spatial reasoning environment built on top of the [Theory of Space](https://github.com/mll-lab-nu/Theory-of-Space.git) framework.

## Installation

1. Download the dataset:

```bash
cd VAGEN

hf download yw12356/spatial_gym_dataset \
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

| config | task | rooms scored |
|---|---|---|
| `config.yaml` | 2-room active exploration | all 20 of `room_data/2-room`. Nothing under `examples/train/` reads 2-room, so none of them were trained on |
| `config_1room.yaml` | 1-room, matching the training task | the four held out: training takes rooms 0-15 of `room_data/1-room` and validation 16-19 |

`config_1room.yaml` is what the shipped launcher runs by default:

```bash
MODEL_PATH=/path/to/checkpoint \
  bash examples/evaluate/spatial_gym/vllm/eval_qwen25_vl_3b.sh
```

## Training

GRPO training
```bash
bash examples/train/spatial_gym/train_grpo_qwen25vl3b.sh
bash examples/train/spatial_gym/train_grpo_qwen25vl7b.sh
```

PPO training -- `default_gae` with a critic, which is what PPO is here

```bash
bash examples/train/spatial_gym/train_default_gae_qwen25vl3b.sh
bash examples/train/spatial_gym/train_default_gae_qwen25vl7b.sh
```

Bi-level GAE

```bash
bash examples/train/spatial_gym/train_bi_level_gae_qwen25vl3b.sh
```
