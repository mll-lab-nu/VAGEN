# SpatialGym (Theory of Space Environment)

SpatialGym is a spatial reasoning environment built on top of the [Theory of Space](https://github.com/mll-lab-nu/Theory-of-Space.git) framework.

## Installation

Install the base VAGEN package first:

```bash
pip install -e .
```

Then install the additional dependencies required by SpatialGym:

```bash
pip install numpy matplotlib scipy pillow tqdm imageio omegaconf
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

## Reference

This environment is based on:
- [Theory of Space](https://github.com/mll-lab-nu/Theory-of-Space.git)
