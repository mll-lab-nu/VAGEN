# EB-ALFRED Environment

## Installation (First-Time Only)

```bash
git clone https://github.com/EmbodiedBench/EmbodiedBench.git /root/EmbodiedBench
touch /root/EmbodiedBench/embodiedbench/__init__.py
pip install -e /root/EmbodiedBench

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install "ai2thor==2.1.0" "gym==0.23.0" "numpy<2.0" \
    scipy Pillow networkx revtok vocab h5py tqdm natsort pyquaternion
pip install "flask==1.1.4" "werkzeug==1.0.1" \
    "markupsafe<2.1" "jinja2<3.0" "itsdangerous<2.0"
pip install "opencv-python-headless<4.9"
pip install "numpy<2.0"

git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED
mv EB-ALFRED /path/to/EmbodiedBench/embodiedbench/envs/eb_alfred/data/json_2.1.0
```

---

## Per-Boot Setup

### 1. Start Xorg + Server

```bash
bash vagen/envs/eb_alfred/start_server.sh
```

### 2. SSH Tunnel (if training machine is remote)

```bash
# On env server
ssh -p <PORT> -R 8000:localhost:8000 \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=5 \
    -N -f user@training-machine-ip

# On training machine — set once in /etc/ssh/sshd_config:
# MaxSessions 200
# then: service ssh reload
```

---

## Checklist

- [ ] `ps aux | grep Xorg`
- [ ] `curl http://localhost:8000/health`
- [ ] `ps aux | grep thor` — no leftover Unity processes
- [ ] `nvidia-smi` — GPU memory free
