# EB-ALFRED Environment — Setup & Run Guide

EB-ALFRED integrates [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench)'s
AI2-THOR household tasks into the VAGEN framework.

---

## Installation (First-Time Only)

### 1. Environment Installation

```bash
cd ERA-rl/VAGEN/vagen/envs/eb_alfred/Embench_new
conda env create -f conda_envs/environment.yaml
conda activate embench
pip install -e .
```

### 2. Additional Installation

Download the dataset from HuggingFace:

```bash
conda activate embench
git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED
mv EB-ALFRED embodiedbench/envs/eb_alfred/data/json_2.1.0
```

---

## One-Time Setup (per machine restart)

These steps must be done **once per machine boot** before any evaluation or training run.

### 1. Start GPU X Server

AI2-THOR (Unity) requires a real display. We use a GPU-accelerated Xorg:

```bash
# GPU 0 (PCI:1:0:0) → Display :2
Xorg -noreset +extension GLX +extension RANDR +extension RENDER \
     -config /tmp/xorg.conf :2 &
```

> **Note:** `/tmp/xorg.conf` is created during machine setup. Its contents:
> ```
> Section "Device"
>     Identifier "Device0"
>     Driver "nvidia"
>     BusID "PCI:1:0:0"
> EndSection
> Section "Screen"
>     Identifier "Screen0"
>     Device "Device0"
>     DefaultDepth 24
>     Option "AllowEmptyInitialConfiguration" "True"
>     SubSection "Display"
>         Depth 24
>         Virtual 1024 768
>     EndSubSection
> EndSection
> Section "ServerLayout"
>     Identifier "Layout0"
>     Screen 0 "Screen0" 0 0
> EndSection
> ```
> If `/tmp/xorg.conf` is missing (e.g. after reboot), recreate it with the above.

> **Alternative (slower):** Software rendering with Xvfb:
> ```bash
> Xvfb :1 -screen 0 1024x768x24 -ac &
> # Then use DISPLAY=:1 everywhere below
> ```
> Xvfb is ~7× slower per reset but works without NVIDIA Xorg drivers.

### 2. Start EB-ALFRED Environment Server

The server manages AI2-THOR processes and load-balances sessions across GPUs.

```bash
export PATH="/opt/miniforge3/bin:/usr/bin:/bin:$PATH"

DISPLAY=:2 conda run -n embench \
    python -m vagen.envs.eb_alfred.serve \
    --port 8000 \
    --x-displays 2
```

> **`--x-displays 2`** is required. Without it, the server auto-detects GPU indices
> (0, 1, ...) from `nvidia-smi` and tries displays `:0`, `:1`, which may not exist.
> Always specify the actual display number explicitly.

Wait until you see:
```
Starting EB-ALFRED service on 0.0.0.0:8000
GPU displays: [:2] (auto-balanced)
Health check: http://localhost:8000/health
```

You can verify the server is up:
```bash
curl http://localhost:8000/health
```

---

## Running Evaluations

With the server running, launch evaluations from a **separate terminal**:

```bash
export PATH="/opt/miniforge3/bin:/usr/bin:/bin:$PATH"

conda run -n vagen \
    python -m vagen.evaluate.run_eval \
    --config tests/<config_file>.yaml
```

### Available Configs

| Config | Episodes | Concurrency | Resolution | Notes |
|--------|:--------:|:-----------:|:----------:|-------|
| `tests/eval_eb_alfred_gpt41_20ep.yaml` | 20 | 3 | 300 | Quick reference run |
| `tests/eval_eb_alfred_gpt41_10ep_serial_500.yaml` | 10 | 1 (serial) | 500 | Serial baseline |
| `tests/eval_eb_alfred_gpt41_128ep_parallel_500.yaml` | 128 | 100 (parallel) | 500 | High-throughput parallel |

---

## Environment Notes

### Conda Environments

| Env | Python | Purpose |
|-----|--------|---------|
| `embench` | 3.9 | AI2-THOR + EmbodiedBench (env server) |
| `vagen` | 3.10 | VAGEN framework (eval runner, training) |

### Key Constraints

- **Flask 1.1.4 + Werkzeug 1.0.1** must stay pinned in `embench` — newer versions break AI2-THOR's socket server.
- **GPU Xorg** requires the nvidia driver version to exactly match the kernel module (`580.95.05`). Do **not** `apt upgrade` the nvidia driver without redoing the `xorg.conf` setup.
- **Only one GPU can host an X server** in this container environment. All Unity instances share Xorg `:2` (GPU 0). GPU 1 is available for CUDA workloads but not for display.
- **Multiple concurrent Unity instances** are fine on the same Xorg. 20+ instances tested successfully; ~100 is feasible (CPU-bound at ~150% per instance on 160-core machine).

### Speed Reference (this machine, 300×300)

| Backend | Avg Reset | Avg Step |
|---------|:---------:|:--------:|
| GPU Xorg :2 (NVIDIA) | 0.6s | 0.039s |
| Xvfb :1 (software) | 4.3s | 0.211s |

---

## Startup Checklist

Before every eval run:

- [ ] `ps aux | grep Xorg` — confirm Xorg :2 is running
- [ ] `curl http://localhost:8000/health` — confirm env server is up
- [ ] `ps aux | grep thor` — no leftover Unity processes from previous runs
- [ ] `nvidia-smi` — GPU memory is mostly free

If Xorg or the server crashed, restart them per steps 1 & 2 above.
