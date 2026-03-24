# ManiSkill (Primitive Skill Robot Manipulation)

ManiSkill is a robot manipulation environment where the LLM controls a Franka Emika robot arm via pick/place/push commands. Built on [ManiSkill](https://github.com/haosulab/ManiSkill).

Supported tasks: AlignTwoCube, PlaceTwoCube, PutAppleInDrawer, StackThreeCube.

## Installation

Install the additional dependencies:

```bash
pip install mani_skill fire
```

ManiSkill requires GPU rendering (`render_backend="gpu"`), so the server must run on a machine with a GPU.

## Evaluation

Start the ManiSkill server, then run evaluation:

```bash
# Terminal 1: start server
python -m vagen.envs.maniskill.serve

# Terminal 2: run eval
python -m vagen.evaluate.run_eval --config examples/evaluate/maniskill/config.yaml
```

## Training

Start the ManiSkill server, then run training:

```bash
# Terminal 1: start server
python -m vagen.envs.maniskill.serve

# Terminal 2: run training
bash examples/train/maniskill/train_ppo_qwen25vl3b.sh
```
