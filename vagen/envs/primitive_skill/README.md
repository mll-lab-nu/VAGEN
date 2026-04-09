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
pip install mani_skill gymnasium

# ManiSkill requires GPU rendering
# See https://maniskill.readthedocs.io/ for full setup
```

ManiSkill uses GPU rendering (`render_backend="gpu"`), so environments must run on a GPU-equipped machine.

## Server

The primitive skill environment runs as a separate server (via `envs_remote` framework).

```bash
# Auto-detect GPUs, default settings
python -m vagen.envs.primitive_skill.serve

# Custom settings
python -m vagen.envs.primitive_skill.serve --devices='[0,1]' --max_envs=32 --port=8001
```

Key parameters:
- `devices`: GPU IDs (default: auto-detect)
- `max_envs`: max concurrent environments, bounds GPU memory (default: 64)
- `thread_pool_size`: should be >= max_envs (default: 64)

## Benchmark

```bash
# Terminal 1: start server
python -m vagen.envs.primitive_skill.serve

# Terminal 2: run benchmark
python -m vagen.envs.primitive_skill.benchmark \
    --base_url http://localhost:8000 \
    --num_rounds 10 \
    --num_clients 32 \
    --num_steps 3
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

Coordinates are in millimeters (integers). Multiple actions per step separated by `|`.
