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

### 1. Start Xorg (GPU-accelerated, Xvfb will not work)

```bash
# Get GPU BusID (hex → decimal, e.g. 0x41 → 65)
nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader

cat > /tmp/xorg0.conf << 'EOF'
Section "Device"
    Identifier  "GPU0"
    Driver      "nvidia"
    BusID       "PCI:65:0:0"
    Option      "AllowEmptyInitialConfiguration" "True"
EndSection
Section "Monitor"
    Identifier  "Monitor0"
    HorizSync   28.0-80.0
    VertRefresh 48.0-75.0
    Modeline    "1920x1080" 172.80 1920 2040 2248 2576 1080 1081 1084 1118
EndSection
Section "Screen"
    Identifier  "Screen0"
    Device      "GPU0"
    Monitor     "Monitor0"
    DefaultDepth 24
    SubSection  "Display"
        Depth   24
        Modes   "1920x1080"
        Virtual 1920 1080
    EndSubSection
EndSection
Section "ServerLayout"
    Identifier "Layout0"
    Screen 0 "Screen0"
EndSection
EOF

# Single GPU
Xorg -noreset +extension GLX -config /tmp/xorg0.conf :0 &

# Dual GPU (repeat with second config for :1)
Xorg -noreset +extension GLX -config /tmp/xorg1.conf :1 &

# Verify
grep "Virtual screen size" /var/log/Xorg.0.log
```

### 2. Start Server

```bash
python -m vagen.envs.eb_alfred.serve \
    --port 8000 \
    --capacity 90 \
    --startup-concurrency 6 \
    --x-displays 0,1

curl http://localhost:8000/health
```

### 3. SSH Tunnel (if training machine is remote)

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
