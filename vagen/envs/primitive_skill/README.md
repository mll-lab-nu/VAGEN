# Primitive Skill Environment

ManiSkill-based robot manipulation environment. The agent controls a Franka Emika robot arm via high-level primitive skills (`pick`, `place`, `push`) to complete tabletop manipulation tasks.

## Tasks

| Task | Description |
|------|-------------|
| `AlignTwoCube` | Align red and green cubes along the x-axis to the origin |
| `PlaceTwoCube` | Place red cube on left target, green cube on right target |
| `StackThreeCube` | Stack red on green, purple on red |
| `PutAppleInDrawer` | Grasp apple, place in drawer, close drawer |

## Installation

```bash
pip install "mani_skill<=3.0.0b22" gymnasium "numpy<=2.2"
python -m mani_skill.utils.download_asset partnet_mobility_cabinet
python -m mani_skill.utils.download_asset ycb

# ManiSkill requires GPU rendering
# See https://maniskill.readthedocs.io/ for full setup
```


## Server


```bash
# Auto-detect GPUs, default settings
python -m vagen.envs.primitive_skill.serve 2>&1 | tee serve_$(date +%Y%m%d_%H%M%S).log

# Custom settings
python -m vagen.envs.primitive_skill.serve --devices='[0,1]' --max_envs=32 --port=8001 2>&1 | tee serve_$(date +%Y%m%d_%H%M%S).log
```

Key parameters:
- `devices`: GPU IDs (default: auto-detect)
- `max_envs`: max concurrent environments, bounds GPU memory (default: 64)
- `thread_pool_size`: should be >= max_envs (default: 64)

## Evaluation

```bash
# Terminal 1: start server

# Terminal 2: run eval
bash examples/evaluate/primitive_skill/run_eval.sh
```

Config: `examples/evaluate/primitive_skill/config.yaml`

## Training

```bash
# Terminal 1: start server

# Terminal 2: run training
cd VAGEN
bash examples/train/primitive_skill/train_ppo_qwen25vl3b.sh
```

## Prompt Formats

| Format | Structure |
|--------|-----------|
| `free_think` | `<think>...</think><answer>...</answer>` |
| `wm` | `<observation>...</observation><think>...</think><answer>...</answer><prediction>...</prediction>` |

## Action Space

```
pick(x, y, z)                        # Grasp object at (x,y,z)
place(x, y, z)                       # Place held object at (x,y,z)
push(x1, y1, z1, x2, y2, z2)        # Push object from (x1,y1,z1) to (x2,y2,z2)
```
