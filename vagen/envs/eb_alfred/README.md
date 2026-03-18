# EB-ALFRED Environment

AI2-THOR based household robot task environment from [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench). The agent receives egocentric RGB images and executes multi-step tasks (cleaning, heating, slicing, storing objects).

## Running the Service

The environment runs on a **separate GPU machine** with a physical or virtual display. AI2-THOR requires X11 rendering (CloudRendering is not supported on ai2thor 2.1.0).

**One-time setup — create the conda environment:**

```bash
conda create -n embodiedbench python=3.9 -y
conda activate embodiedbench

git clone https://github.com/EmbodiedBench/EmbodiedBench.git /root/EmbodiedBench
touch /root/EmbodiedBench/embodiedbench/__init__.py
pip install -e /root/EmbodiedBench

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install "ai2thor==2.1.0" "gym==0.23.0" "numpy<2.0" \
    scipy Pillow networkx revtok vocab h5py tqdm natsort pyquaternion
pip install "flask==1.1.4" "werkzeug==1.0.1" \
    "markupsafe<2.1" "jinja2<3.0" "itsdangerous<2.0"
pip install "opencv-python-headless<4.9"
```

**Download dataset** (`eval_set` selects the split: `base` — standard tasks, `long` — longer horizon):

```bash
git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED
mv EB-ALFRED /root/EmbodiedBench/embodiedbench/envs/eb_alfred/data/json_2.1.0
```

**Start the server** (GPUs and Xorg are auto-detected and started):

```bash
conda activate embodiedbench
python -m vagen.envs.eb_alfred.serve
```

Key parameters:
- `devices`: GPU indices (default: auto-detect via `CUDA_VISIBLE_DEVICES` or `nvidia-smi`)
- `capacity`: max concurrent Unity environments (default: 16)
- `startup_concurrency`: max Unity processes starting simultaneously, prevents CPU spikes (default: 8)
- `session_timeout`: idle session cleanup in seconds (default: 3600)

```bash
# Example: 2 GPUs, higher capacity
python -m vagen.envs.eb_alfred.serve --devices='[0,1]' --capacity=90 --startup_concurrency=6 --port=8000
```

**SSH tunnel** (if training machine is remote):

```bash
# Run on the env server — forwards port 8000 to training machine
ssh -p <PORT> -R 8000:localhost:8000 \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=5 \
    -N -f user@training-machine-ip

# On training machine, allow many tunnels — add to /etc/ssh/sshd_config:
# MaxSessions 200
# then: service ssh reload
```

## Evaluation

```bash
conda activate vagen

# Terminal 1 (env server): start service
python -m vagen.envs.eb_alfred.serve --devices='[0,1]' --capacity=90

# Terminal 2 (training machine): run eval
python -m vagen.evaluate.run_eval --config examples/evaluate/eb_alfred/config.yaml
```

Config: `examples/evaluate/eb_alfred/config.yaml`

## Training

```bash
conda activate vagen

# Terminal 1 (env server): start service
python -m vagen.envs.eb_alfred.serve --devices='[0,1]' --capacity=90

# Terminal 2 (training machine): run training
cd VAGEN
bash examples/train/eb_alfred/train_grpo_qwen25vl3b.sh
```

Configs: `examples/train/eb_alfred/`

