# Navigation Environment

AI2-THOR based indoor navigation environment. The agent receives egocentric RGB images and follows natural language instructions to navigate to target locations.

## Installation

```bash
pip install ai2thor
```

AI2-THOR runs a Unity backend process per environment instance. It requires a GPU with cloud rendering support.

## Server

The navigation environment runs as a separate server (via `envs_remote` framework) because AI2-THOR controllers are heavy GPU processes that can't share a process with training.

```bash
# Auto-detect GPUs, default settings
python -m vagen.envs.navigation.serve

# Custom settings
python -m vagen.envs.navigation.serve --devices='[0,1]' --max_envs=64 --port=8001
```

Key parameters:
- `devices`: GPU IDs (default: auto-detect)
- `max_envs`: max alive environments, bounds GPU memory (default: 128)
- `thread_pool_size`: should be >= max_envs (default: 128)

## Evaluation

Start the server, then run eval:

```bash
# Terminal 1: start server
python -m vagen.envs.navigation.serve

# Terminal 2: run eval
bash examples/evaluate/navigation/run_eval.sh
```

Config: `examples/evaluate/navigation/config.yaml`

## Training

Start the server, then run training:

```bash
# Terminal 1: start server
python -m vagen.envs.navigation.serve

# Terminal 2: run training
cd VAGEN
bash examples/train/navigation/train_grpo_qwen25vl3b.sh
```

Configs: `examples/train/navigation/`

## Prompt Formats

- `free_think`: `<think>...</think><action>...</action>`
- `wm`: `<observation>...</observation><think>...</think><action>...</action><prediction>...</prediction>`
- `no_think`: `<action>...</action>` (strict)
- `eval_mode`: `<action>...</action>` (lenient, allows extra text)

## Datasets

Located in `datasets/`. Each eval set has 60 tasks:
- `base` — standard navigation
- `common_sense` — requires common sense reasoning
- `complex_instruction` — multi-step instructions
- `visual_appearance` — target described by appearance
- `long_horizon` — longer navigation paths

## Interactive Test

```bash
python -m vagen.envs.navigation.navigation_env --eval_set base --seed 0
```
