# EB-ALFRED Environment — Setup & Run Guide

EB-ALFRED integrates [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench)'s
AI2-THOR household tasks into the VAGEN framework.

- 301 evaluation episodes across 6 eval sets
- 162 discrete actions (find / pick / put / open / close / slice / toggle)
- GPU-accelerated Xorg required — Xvfb does **not** work (no hardware OpenGL)

---

## Installation (First-Time Only)

### 1. Install EmbodiedBench

```bash
# Clone EmbodiedBench
git clone https://github.com/EmbodiedBench/EmbodiedBench.git /root/EmbodiedBench

# REQUIRED: the package is missing __init__.py — editable install breaks without it
touch /root/EmbodiedBench/embodiedbench/__init__.py

pip install -e /root/EmbodiedBench
```

### 2. Install Required Packages (order matters)

AI2-THOR 2.1.0 has strict version requirements. Install in this order:

```bash
# 1. PyTorch (match your CUDA driver; cu126 for driver >= 525)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 2. ai2thor + core deps
pip install "ai2thor==2.1.0" "gym==0.23.0" "numpy<2.0" \
    scipy Pillow networkx revtok vocab h5py tqdm natsort pyquaternion

# 3. Pin flask/werkzeug — ai2thor 2.1.0 uses an internal Flask server
#    that is incompatible with Flask 2+ / Werkzeug 2+
pip install "flask==1.1.4" "werkzeug==1.0.1" \
    "markupsafe<2.1" "jinja2<3.0" "itsdangerous<2.0"

# 4. opencv — must be <4.9 (4.9+ requires numpy>=2, which conflicts with gym 0.23.0)
pip install "opencv-python-headless<4.9"

# 5. Re-pin numpy (opencv/hydra may have upgraded it)
pip install "numpy<2.0"
```

> **Critical version constraints:**
> | Package | Required | Reason |
> |---------|----------|--------|
> | `flask` | `==1.1.4` | ai2thor 2.1.0 internal web server |
> | `werkzeug` | `==1.0.1` | same — Werkzeug 2+ breaks the socket bridge |
> | `numpy` | `<2.0` | gym 0.23.0 incompatible with numpy 2.x |
> | `opencv-python-headless` | `<4.9` | 4.9+ requires numpy>=2 |
> | `gym` | `==0.23.0` | required by EmbodiedBench |

### 3. Download Dataset

```bash
git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED
mv EB-ALFRED /path/to/EmbodiedBench/embodiedbench/envs/eb_alfred/data/json_2.1.0
```

---

## One-Time Setup (per machine restart)

These steps must be done **once per machine boot** before any evaluation or training run.

### 1. Start GPU X Server

AI2-THOR (Unity) requires a **GPU-accelerated Xorg** for OpenGL rendering.
**Xvfb will not work** — Unity falls back to CPU rendering and hangs.

The Xorg config must include a `Monitor` section with a `Modeline` and `Modes` directive.
Without it, Unity sees "Desktop is 0 x 0 @ 0 Hz" and freezes.

#### Single-GPU setup

```bash
# Find your GPU BusID (convert hex to decimal: e.g. 0x41 → 65)
nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader

cat > /tmp/xorg.conf << 'EOF'
Section "ServerFlags"
    Option "AllowEmptyInput" "True"
EndSection

Section "Device"
    Identifier  "Device0"
    Driver      "nvidia"
    BusID       "PCI:1:0:0"          # replace with your GPU BusID
    Option      "AllowEmptyInitialConfiguration" "True"
EndSection

Section "Monitor"
    Identifier  "Monitor0"
    HorizSync   28.0-80.0
    VertRefresh 48.0-75.0
    Modeline    "1920x1080" 172.80 1920 2040 2248 2576 1080 1081 1084 1118
    Option      "DPMS"
EndSection

Section "Screen"
    Identifier  "Screen0"
    Device      "Device0"
    Monitor     "Monitor0"
    DefaultDepth 24
    Option      "AllowEmptyInitialConfiguration" "True"
    Option      "UseDisplayDevice" "none"
    SubSection  "Display"
        Depth   24
        Modes   "1920x1080"
        Virtual 1920 1080
    EndSubSection
EndSection

Section "ServerLayout"
    Identifier "Layout0"
    Screen 0 "Screen0" 0 0
EndSection
EOF

Xorg -noreset +extension GLX +extension RANDR +extension RENDER \
     -config /tmp/xorg.conf :1 &>/tmp/xorg1.log &

# Verify resolution was detected
grep "Virtual screen size" /var/log/Xorg.1.log
# Expected: "Virtual screen size configured to be 1920 x 1080"
```

#### Dual-GPU setup (one Xorg per GPU)

```bash
# GPU 0 → display :0    GPU 1 → display :1
# Write a single-GPU config for each (see above template, change BusID/Identifier)

Xorg -noreset +extension GLX +extension RANDR +extension RENDER \
     -config /tmp/xorg_gpu0.conf :0 &>/tmp/xorg0.log &

Xorg -noreset +extension GLX +extension RANDR +extension RENDER \
     -config /tmp/xorg_gpu1.conf :1 &>/tmp/xorg1.log &
```

### 2. Start EB-ALFRED Environment Server

```bash
python -m vagen.envs.eb_alfred.serve \
    --port 8000 \
    --capacity 128 \
    --startup-concurrency 4 \
    --x-displays 1          # single GPU: --x-displays 1
                             # dual GPU:   --x-displays 0,1
```

> **`--startup-concurrency 4`** staggers Unity process startup to avoid CPU spikes.
> Even with `--capacity 128`, only 4 Unity instances start simultaneously; the rest queue.

Wait for:
```
Starting EB-ALFRED service on 0.0.0.0:8000
GPU displays: [:0, :1] (auto-balanced)
```

Verify:
```bash
curl http://localhost:8000/health
```

---

## Running Evaluations

With the server running, launch from a separate terminal:

```bash
python -m vagen.evaluate.run_eval --config tests/<config_file>.yaml
```

---

## Environment Notes

### Conda Environments

| Env | Python | Purpose |
|-----|--------|---------|
| `embench` | 3.9 | AI2-THOR + EmbodiedBench (env server) |
| `vagen` | 3.10 | VAGEN framework (eval runner, training) |

### Known Issues

**Unity hangs with "Desktop is 0 x 0 @ 0 Hz"**
The Xorg config is missing `Monitor` + `Modeline` + `Modes`. The `Virtual` directive alone
is not enough. Use the full config shown above.

**`embodiedbench` import fails after editable install**
The package directory has no `__init__.py`, which breaks pip's editable-install finder.
Fix: `touch embodiedbench/__init__.py` then `pip install -e .`

**`opencv-python-headless` version conflict**
`opencv>=4.9` requires `numpy>=2`, but `gym==0.23.0` requires `numpy<2`.
Fix: `pip install "opencv-python-headless<4.9"`

**AI2-THOR first-run download**
Unity binary (~390 MB) downloads to `~/.ai2thor/releases/` on first run.
Subsequent runs use the cache.

### Startup Checklist

Before every run:
- [ ] `ps aux | grep Xorg` — Xorg is running on the expected display(s)
- [ ] `curl http://localhost:8000/health` — env server is up
- [ ] `ps aux | grep thor` — no leftover Unity processes from previous runs
- [ ] `nvidia-smi` — GPU memory is mostly free
