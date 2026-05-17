# VAGEN-WEBAGENT Full Setup Guide

End-to-end setup for a fresh **cloud VM / bare-metal server** with 2× GPU
(≥ 48 GB VRAM each) and root access. Goal: run the WebAgent-R1 GRPO
training on Qwen2.5-3B in `examples/train/webarena/`.

If you're on a SLURM cluster without root, skip §3 and §6 — use the SSH
tunnel approach (§3 alt) and module/container-based system deps.

Time budget: ~2-3 hours of setup, ~30 GB disk for Docker, ~30 GB for HF
cache. WebArena Docker stack alone is 10-20 GB.

---

## 1. Hardware & OS

| | Recommended |
|---|---|
| GPU | 2× NVIDIA, ≥ 48 GB VRAM each (Blackwell / Ada / A100 / H100) |
| CUDA | 12.x driver (we test on 12.8) |
| RAM | 192 GB+ |
| Disk | 200 GB+ free |
| OS | Ubuntu 22.04 / Rocky 9 / Debian 12 |
| Python | 3.10 (webarena env) + 3.12 (vagen env) |

---

## 2. System packages

```bash
# (Ubuntu) build tools + curl/git/etc.
sudo apt-get update
sudo apt-get install -y build-essential git curl wget jq tmux vim \
                        ca-certificates gnupg

# Verify NVIDIA driver + CUDA visible
nvidia-smi  # should list both GPUs
nvcc --version  # should be 12.x

# Docker (for WebArena stack)
curl -fsSL https://get.docker.com | sudo bash
sudo usermod -aG docker $USER
newgrp docker  # or log out/in
docker run --rm hello-world  # verify

# (Optional) NVIDIA container toolkit if you want GPU-in-container (not
# required for our setup — training runs natively, only WebArena uses
# Docker for app servers which are CPU-only)
```

If on a cluster without root, replace this section with:
- `module load cuda/12.x`
- `module load python/3.12`
- Skip Docker (use SSH tunnel to a remote WebArena host instead — see §3 alt)

---

## 3. WebArena Docker stack (local option)

WebArena ships 6 services as Docker images. We host all of them on the
same machine (or a sibling) so the env server (§5) can hit `localhost`.

```bash
# ~30 GB pull. The exact image tags are pinned in the official repo:
# https://github.com/web-arena-x/webarena/blob/main/environment_docker/

# Shopping (port 7770)
docker run -d --name shopping -p 7770:80 shopping_final_0712 \
  /bin/bash -c "/var/www/magento2/bin/magento setup:store-config:set --base-url='http://localhost:7770' && \
                /var/www/magento2/bin/magento cache:flush && \
                apache2ctl -D FOREGROUND"

# Shopping admin (port 7780)
docker run -d --name shopping_admin -p 7780:80 shopping_admin_final_0719 \
  /bin/bash -c "apache2ctl -D FOREGROUND"

# Reddit / Postmill (port 9999)
docker run -d --name forum -p 9999:80 postmill-populated-exposed-withimg

# GitLab (port 8023, ~10 GB image)
docker run -d --name gitlab -p 8023:8023 \
  --shm-size 256m gitlab-populated-final-port8023 \
  /opt/gitlab/embedded/bin/runsvdir-start

# Wikipedia (kiwix, port 8888)
docker run -d --name kiwix -p 8888:80 kiwix \
  --port=80 /data/wikipedia_en_all_maxi_2022-05.zim

# OpenStreetMap (port 3000) — OPTIONAL, skip if you don't need map tasks
docker run -d --name map -p 3000:80 openstreetmap-website-image

# Homepage (port 4399) — small router page; you can also build from source
docker run -d --name homepage -p 4399:80 webarena_homepage
```

Verify all 6 are up:
```bash
for p in 7770 7780 9999 8023 8888 4399; do
  curl -s -o /dev/null -w "$p: %{http_code}\n" http://localhost:$p/
done
```

### 3-alt. SSH tunnel option (no local Docker)

If WebArena is hosted on a remote machine (e.g. Amazon-provided VM),
tunnel the ports instead:
```bash
ssh -i ~/.ssh/id_ed25519 -p <REMOTE_PORT> root@<REMOTE_HOST> \
    -L 7770:localhost:7770 -L 7780:localhost:7780 \
    -L 9999:localhost:9999 -L 8023:localhost:8023 \
    -L 8888:localhost:8888 -L 4399:localhost:4399 \
    -L 3000:localhost:3000 -N &  # 3000 optional
```

Same `localhost:PORT` works for the env server either way.

---

## 4. Code: clone repos

```bash
mkdir -p ~/work && cd ~/work
git clone https://github.com/<your-fork>/VAGEN-WEBAGENT.git
cd VAGEN-WEBAGENT

# verl is required as a sibling. The vagen_multiturn.yaml does
# `hydra.searchpath: file:../../verl/verl/trainer/config`, so verl/ must
# live inside the VAGEN-WEBAGENT root.
git clone https://github.com/volcengine/verl.git verl_src
ln -s verl_src verl   # symlink: VAGEN-WEBAGENT/verl -> verl_src

# Sanity: this file should resolve
ls verl/verl/trainer/config/ppo_trainer.yaml
```

Note: this repo's [examples/train/webarena/](examples/train/webarena/)
already contains the WebArena training scripts/yamls. The smoke-test
fixes (verl path, config symlink, model paths) are baked in.

---

## 5. Python envs

We use **two conda envs** because webarena's playwright pins Python 3.10
and old fastapi/lxml versions, while vagen/verl needs Python 3.12 +
torch 2.8 + sglang 0.5 + vllm 0.11.

### 5a. vagen env (training)

```bash
conda create -n vagen python=3.12 -y
conda activate vagen

# Core stack — pin the versions we tested
pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install \
  ray[default]==2.53.0 \
  sglang==0.5.2 \
  vllm==0.11.0 \
  flash-attn==2.8.1 --no-build-isolation \
  flashinfer-python==0.3.1 \
  transformers==4.56.1 \
  accelerate==1.11.0 \
  hydra-core==1.3.2 omegaconf==2.3.0 \
  triton==3.4.0 \
  pypdf==6.10.2 \
  wandb httpx fire pillow numpy pandas

# Install verl (editable from cloned source)
pip install -e ./verl_src

# Install vagen (editable)
pip install -e .

# Sanity
python -c "import verl; from verl.trainer.constants_ppo import get_ppo_ray_runtime_env; print('verl OK', verl.__file__)"
python -c "import vagen; print('vagen OK')"
python -c "import sglang, vllm; print('sglang', sglang.__version__, 'vllm', vllm.__version__)"
```

### 5b. webarena env (env server only)

```bash
conda create -n webarena python=3.10 -y
conda activate webarena

pip install \
  playwright==1.32.1 \
  fastapi==0.136.0 \
  "uvicorn>=0.44,<0.50" \
  lxml==6.0.4 \
  beautifulsoup4==4.14.3 \
  scikit-image==0.25.2 \
  matplotlib==3.10.8 \
  dashscope anthropic \
  "openai>=1.0" \
  numpy pandas python-multipart httpx aiofiles

playwright install chromium  # downloads ~150 MB Chromium

# Install vagen so `python -m vagen.envs.webarena.serve` resolves
cd ~/work/VAGEN-WEBAGENT
pip install -e .  # in the webarena env too

# Sanity
python -c "from vagen.envs.webarena.handler import WebArenaHandler; print('OK')"
```

---

## 6. Models — Hugging Face cache

```bash
export HF_HOME=$HOME/hf_cache   # or wherever you have ~30 GB
mkdir -p $HF_HOME

conda activate vagen
huggingface-cli login   # if any models are gated; weizhepei's is public

# RL initialization policy (the "weizhepei baseline")
huggingface-cli download weizhepei/Qwen2.5-3B-WebArena-Lite-SFT-epoch-5 \
  --local-dir-use-symlinks False

# Optional: also pre-cache the base Qwen for tokenizer/architecture lookups
huggingface-cli download Qwen/Qwen2.5-3B-Instruct
```

Make sure to `export HF_HOME=$HOME/hf_cache` in your training shell too,
otherwise verl/HF will redownload to `~/.cache/huggingface`.

---

## 7. WebArena env vars + auth bootstrap

```bash
cd ~/work/VAGEN-WEBAGENT
conda activate webarena
source vagen/envs/webarena/setup_vars.sh

# Bootstrap auth cookies once (~3 min). This logs into shopping/shopping_admin/
# reddit/gitlab via Playwright and caches storage_state.json files.
mkdir -p .wa_auth
PYTHONPATH=. python vagen/envs/webarena/browser_env/auto_login.py \
  --auth_folder $(pwd)/.wa_auth \
  --site_list shopping,shopping_admin,reddit,gitlab

# Verify cookies generated
ls .wa_auth/  # should show shopping_state.json, etc.
```

If `auto_login.py` fails: usually a Docker/tunnel problem. Verify §3
ports respond before retrying.

---

## 8. Validate with smoke test (15 min)

This catches install issues before committing to a 16-hour training run.

```bash
# Terminal 1: webarena server (small pool for smoke)
cd ~/work/VAGEN-WEBAGENT
conda activate webarena
source vagen/envs/webarena/setup_vars.sh
PYTHONPATH=. python -m vagen.envs.webarena.serve \
  --task_config_file=vagen/envs/webarena/config_files/normalized_test.json \
  --n_browsers=2 --max_contexts_per_browser=2 \
  --port=8002 --auth_cache_dir=./.wa_auth

# Terminal 2: tiny train smoke (uses Qwen2.5-0.5B, batch=1, 2 steps)
# This is the same script we used during development.
cd ~/work/VAGEN-WEBAGENT
conda activate vagen
export HF_HOME=$HOME/hf_cache
bash examples/train/webarena/train_smoke_qwen25_05b.sh
```

Expected: ~5 min init, then `Training Progress: 1/2` after ~30s, then
`Training Progress: 2/2`. If you see metrics like `actor/entropy`,
`critic/vf_loss` printed, the pipeline works end-to-end.

If smoke fails, see [§10 Troubleshooting](#10-troubleshooting) below.

---

## 9. Full training

After smoke passes, scale up to the real run.

```bash
# Restart webarena servers fresh + bigger pool (kills any leaked sessions)
pkill -9 -f "vagen.envs.webarena.serve" || true
pkill -9 -u $USER -f chromium || true
sleep 3

cd ~/work/VAGEN-WEBAGENT
conda activate webarena
source vagen/envs/webarena/setup_vars.sh

# Train server (port 8002, normalized_train.json)
nohup env PYTHONPATH=. python -m vagen.envs.webarena.serve \
  --task_config_file=vagen/envs/webarena/config_files/normalized_train.json \
  --n_browsers=8 --max_contexts_per_browser=8 \
  --port=8002 --auth_cache_dir=./.wa_auth \
  > examples/evaluate/webarena/logs/webarena_train_server.log 2>&1 &

# Val server (port 8003, normalized_test.json)
nohup env PYTHONPATH=. python -m vagen.envs.webarena.serve \
  --task_config_file=vagen/envs/webarena/config_files/normalized_test.json \
  --n_browsers=4 --max_contexts_per_browser=8 \
  --port=8003 --auth_cache_dir=./.wa_auth \
  > examples/evaluate/webarena/logs/webarena_val_server.log 2>&1 &

# Wait for ready
until curl -s --fail http://localhost:8002/health >/dev/null \
   && curl -s --fail http://localhost:8003/health >/dev/null; do sleep 3; done
echo "both servers ready"

# Kick off training (16h+ for 200 steps on 2× 96GB GPUs)
conda activate vagen
export HF_HOME=$HOME/hf_cache
export WANDB_API_KEY=<your_key>

# IMPORTANT: edit train yaml's base_urls if the GPU machine is not the
# same as the env server machine. With local Docker stack on same VM,
# leave it as `http://localhost:8002` (default in this setup) but change
# from `http://dt-login03:8002` (the Delta-cluster default in the yaml).
sed -i 's|http://dt-login03:8002|http://localhost:8002|g' \
  examples/train/webarena/train_webarena_full.yaml
sed -i 's|http://dt-login03:8003|http://localhost:8003|g' \
  examples/train/webarena/val_webarena_full.yaml

bash examples/train/webarena/train_webarena_grpo_qwen25_3b.sh \
  2>&1 | tee /tmp/train_full.log
```

See [examples/train/webarena/RUN_INSTRUCTIONS.md](examples/train/webarena/RUN_INSTRUCTIONS.md)
for the full operations guide (resume, monitoring, sanity checks).

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: verl` | editable install path stale | Re-run `pip install -e ./verl_src` |
| `Could not load 'ppo_trainer'` (Hydra) | `verl/` symlink missing | `cd VAGEN-WEBAGENT && ln -sf verl_src verl` |
| `AssertionError: only support equal chunk. Got size N and chunk M` | `agent.num_workers` mismatched with batch | Set `actor_rollout_ref.rollout.agent.num_workers=<batch_size>` |
| `Connect failed: localhost:8002` (from GPU) | env server is on a different host | Change `base_urls` in yaml to a hostname/IP reachable from GPU |
| `/connect 500` (random) | map task hits localhost:3000 (not running) | Use `seed_list` excluding map seeds (already done in `train_webarena_full.yaml`); or run map docker (§3) |
| `max_new_tokens must be at least 0, got -N` | `data.max_prompt_length + max_response_length < actual_prompt_tokens` | Bump `max_prompt_length` to ≥ 16000 |
| sglang hangs on long prompt (multi-turn concat) | sglang 0.5.2 + 22k+ token prompt edge case | Switch to vllm rollout (`rollout.name=vllm`) — already set in 3B training |
| Train slow (>2 min/step on 0.5B) | FSDP offload on, model is small | Set `*.fsdp_config.param_offload=False` and `optimizer_offload=False` |
| Leaked webarena sessions over time | crashed train didn't `env.close()` | `pkill -9 -f webarena.serve && pkill -9 chromium` and restart server before each train run |
| `OpenBLAS: pthread_create failed for thread N of 64` | login node thread limit | Set `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` |

---

## 11. What's in this repo

```
VAGEN-WEBAGENT/
├── verl -> verl_src/          # symlink (created in §4)
├── verl_src/                  # cloned verl
├── vagen/
│   ├── main_ppo.py            # train entry
│   ├── ray_trainer.py
│   ├── envs/webarena/         # env impl + Playwright + serve.py
│   ├── envs_remote/           # generic HTTP gym client/server framework
│   ├── agent_loop/            # multi-turn agent loop (RL rollout driver)
│   └── configs/
│       ├── vagen_multiturn.yaml   # base trainer config (inherits verl ppo_trainer)
│       └── env_registry.yaml      # env name → class mapping
└── examples/
    ├── train/webarena/        # ★ all WebArena training assets
    │   ├── train_webarena_grpo_qwen25_3b.sh    # main 3B training (paper)
    │   ├── train_webarena_full.yaml            # 540 non-map train seeds
    │   ├── val_webarena_full.yaml              # 130 non-map val seeds
    │   ├── train_smoke_qwen25_05b.sh           # smoke test (0.5B, batch=1, 2 steps)
    │   └── RUN_INSTRUCTIONS.md                 # operational runbook
    └── evaluate/webarena/     # eval-only configs (no training)
```
