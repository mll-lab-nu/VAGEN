#!/usr/bin/env bash
# Automated setup for VAGEN-WEBAGENT on a fresh machine.
#
# Covers SETUP.md sections 2 (system pkgs), 4 (clone), 5 (envs), 6 (HF
# models). Sections 3 (WebArena Docker) and 7 (auth bootstrap) require
# user-specific assets (Docker image tars or SSH credentials) and are
# left as manual steps with prompts at the end.
#
# Idempotent: re-runnable. Each step skips if its output already exists.
#
# Usage:
#   bash setup.sh                       # default: all steps, verbose
#   bash setup.sh --skip-system         # skip apt-get steps (no root)
#   bash setup.sh --skip-models         # skip HF download
#   bash setup.sh --only=envs           # only build conda envs
#   REPO_ROOT=/path/to/repo bash setup.sh
#
# Env vars (override defaults):
#   REPO_ROOT       directory to clone into [default: $PWD]
#   HF_HOME         HF cache location      [default: $HOME/hf_cache]
#   CONDA_BIN       path to conda binary   [default: auto-detect]
#   CUDA_INDEX_URL  pytorch wheel index    [default: cu128]

set -euo pipefail

# ---------------------------------------------------------------- defaults
REPO_ROOT="${REPO_ROOT:-$PWD}"
HF_HOME="${HF_HOME:-$HOME/hf_cache}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
SKIP_SYSTEM=0
SKIP_DOCKER_NOTES=0
SKIP_MODELS=0
SKIP_SMOKE=1   # smoke needs WebArena up; off by default
ONLY=""

for arg in "$@"; do
  case "$arg" in
    --skip-system) SKIP_SYSTEM=1 ;;
    --skip-docker-notes) SKIP_DOCKER_NOTES=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --run-smoke) SKIP_SMOKE=0 ;;
    --only=*) ONLY="${arg#--only=}" ;;
    -h|--help)
      sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "Unknown flag: $arg"; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- helpers
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; }
err()  { echo -e "${RED}✗${NC}  $*" >&2; exit 1; }
step() { echo; echo -e "${BLUE}═══ $* ═══${NC}"; }
have() { command -v "$1" >/dev/null 2>&1; }
should_run() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

# ---------------------------------------------------------------- 0. preflight
step "Preflight"
log "REPO_ROOT=$REPO_ROOT"
log "HF_HOME=$HF_HOME"
mkdir -p "$REPO_ROOT" "$HF_HOME"

if have nvidia-smi; then
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
  ok "Found $GPU_COUNT GPU(s)"
  nvidia-smi -L 2>&1 | head -4
else
  warn "nvidia-smi not found — training will not work, only env server can run"
fi

if have conda; then
  CONDA_BIN="${CONDA_BIN:-$(command -v conda)}"
  CONDA_BASE=$(conda info --base)
  ok "conda at $CONDA_BIN (base: $CONDA_BASE)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  err "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
fi

# ---------------------------------------------------------------- 1. system packages
if should_run system && [ "$SKIP_SYSTEM" = 0 ]; then
  step "System packages (skip with --skip-system)"
  if have sudo && have apt-get; then
    log "Running apt-get install ..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
      build-essential git curl wget jq tmux vim ca-certificates gnupg
    ok "system pkgs installed"
  else
    warn "no sudo or apt-get — skipping. Install build-essential, git, curl manually."
  fi
fi

# ---------------------------------------------------------------- 2. clone code
if should_run code; then
  step "Code: clone VAGEN-WEBAGENT + verl"

  cd "$REPO_ROOT"
  if [ ! -d VAGEN-WEBAGENT/.git ] && [ ! -f VAGEN-WEBAGENT/setup.py ]; then
    if [ -f setup.py ] && grep -q "name=\"vagen\"" setup.py 2>/dev/null; then
      log "Already inside VAGEN-WEBAGENT (setup.py present)"
      VAGEN_DIR="$REPO_ROOT"
    else
      err "VAGEN-WEBAGENT not found at $REPO_ROOT. Clone it first or cd into it."
    fi
  else
    VAGEN_DIR="$REPO_ROOT/VAGEN-WEBAGENT"
  fi
  cd "$VAGEN_DIR"
  log "VAGEN_DIR=$VAGEN_DIR"

  if [ ! -d verl_src/.git ]; then
    log "Cloning verl into ./verl_src ..."
    git clone --depth 1 https://github.com/volcengine/verl.git verl_src
    ok "verl cloned"
  else
    ok "verl_src already exists"
  fi

  if [ ! -e verl ]; then
    ln -s verl_src verl
    ok "Created symlink: verl -> verl_src"
  elif [ -L verl ]; then
    ok "verl symlink already in place"
  elif [ -d verl ] && [ -z "$(ls -A verl 2>/dev/null)" ]; then
    rmdir verl && ln -s verl_src verl
    ok "Replaced empty verl/ dir with symlink"
  fi

  if [ -f verl/verl/trainer/config/ppo_trainer.yaml ]; then
    ok "verl Hydra config reachable"
  else
    err "verl/verl/trainer/config/ppo_trainer.yaml NOT found. Check verl clone + symlink."
  fi
fi

# ---------------------------------------------------------------- 3. vagen conda env
if should_run vagen-env; then
  step "Conda env: vagen (Python 3.12, training)"

  if conda env list | awk '{print $1}' | grep -qx vagen; then
    ok "vagen env already exists (skipping create)"
  else
    log "Creating conda env vagen ..."
    conda create -n vagen python=3.12 -y -q
  fi

  conda activate vagen

  log "Installing pytorch (cu128 wheels) ..."
  pip install -q \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url "$CUDA_INDEX_URL"

  log "Installing core RL stack ..."
  pip install -q \
    'ray[default]==2.53.0' \
    sglang==0.5.2 \
    vllm==0.11.0 \
    flashinfer-python==0.3.1 \
    transformers==4.56.1 \
    accelerate==1.11.0 \
    hydra-core==1.3.2 omegaconf==2.3.0 \
    triton==3.4.0 \
    pypdf==6.10.2 \
    wandb httpx fire pillow numpy pandas

  log "Installing flash-attn (no-build-isolation, slow) ..."
  pip install -q flash-attn==2.8.1 --no-build-isolation || \
    warn "flash-attn build failed — try later with matching CUDA toolkit; training degrades but works without it"

  log "Installing verl (editable from ./verl_src) ..."
  pip install -q -e "$VAGEN_DIR/verl_src"

  log "Installing vagen (editable) ..."
  pip install -q -e "$VAGEN_DIR"

  log "Sanity checks ..."
  python -c "import verl; from verl.trainer.constants_ppo import get_ppo_ray_runtime_env" \
    && ok "verl importable" \
    || err "verl import failed"
  python -c "import vagen; from vagen.envs.webarena.handler import WebArenaHandler" \
    && ok "vagen importable (incl. webarena handler)" \
    || err "vagen import failed"
  python -c "import sglang, vllm, ray, torch; print('sglang', sglang.__version__, 'vllm', vllm.__version__, 'ray', ray.__version__, 'torch', torch.__version__)"

  conda deactivate
fi

# ---------------------------------------------------------------- 4. webarena conda env
if should_run webarena-env; then
  step "Conda env: webarena (Python 3.10, env server)"

  if conda env list | awk '{print $1}' | grep -qx webarena; then
    ok "webarena env already exists (skipping create)"
  else
    log "Creating conda env webarena ..."
    conda create -n webarena python=3.10 -y -q
  fi

  conda activate webarena

  log "Installing playwright + fastapi stack ..."
  pip install -q \
    playwright==1.32.1 \
    fastapi==0.136.0 \
    'uvicorn>=0.44,<0.50' \
    lxml==6.0.4 \
    beautifulsoup4==4.14.3 \
    scikit-image==0.25.2 \
    matplotlib==3.10.8 \
    dashscope anthropic \
    'openai>=1.0' \
    numpy pandas python-multipart httpx aiofiles

  log "Installing Chromium for Playwright (~150 MB) ..."
  playwright install chromium

  log "Installing vagen (editable, for the webarena env server entry point) ..."
  pip install -q -e "$VAGEN_DIR"

  python -c "from vagen.envs.webarena.handler import WebArenaHandler" \
    && ok "vagen.envs.webarena imports OK in webarena env" \
    || err "webarena env import failed"

  conda deactivate
fi

# ---------------------------------------------------------------- 5. HF models
if should_run models && [ "$SKIP_MODELS" = 0 ]; then
  step "Hugging Face models (weizhepei SFT baseline + Qwen base)"

  conda activate vagen
  export HF_HOME

  if have huggingface-cli; then
    log "Downloading weizhepei/Qwen2.5-3B-WebArena-Lite-SFT-epoch-5 ..."
    huggingface-cli download weizhepei/Qwen2.5-3B-WebArena-Lite-SFT-epoch-5 \
      --local-dir-use-symlinks False >/dev/null 2>&1 \
      && ok "weizhepei SFT model cached" \
      || warn "weizhepei download failed — retry manually"

    log "Downloading Qwen/Qwen2.5-0.5B-Instruct (smoke test) ..."
    huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct >/dev/null 2>&1 \
      && ok "Qwen 0.5B cached" \
      || warn "Qwen 0.5B download failed"
  else
    warn "huggingface-cli not found in vagen env (should have been installed by transformers). Skipping."
  fi
  conda deactivate
fi

# ---------------------------------------------------------------- 6. WebArena docker reminder
if should_run docker && [ "$SKIP_DOCKER_NOTES" = 0 ]; then
  step "WebArena Docker stack (manual)"
  cat <<'EOF'
The 6 WebArena services are not on Docker Hub. They are distributed by
the WebArena authors as image archives. You have two options:

  A) If your WebArena Docker host is REMOTE (e.g. provided by a partner):
     SSH-tunnel the 6 ports from that host to localhost — see
     SETUP.md §3-alt.

  B) If you want to host them LOCALLY:
     1. Get the image archives from
        https://github.com/web-arena-x/webarena/blob/main/environment_docker/
     2. `docker load -i <archive.tar>` for each
     3. `docker run -d -p <port>:80 <image>` per SETUP.md §3

After EITHER option, verify all 6 ports respond:
   for p in 7770 7780 9999 8023 8888 4399; do
     curl -s -o /dev/null -w "$p: %{http_code}\n" http://localhost:$p/
   done

Then continue with §7 (auth bootstrap):
   conda activate webarena
   cd $VAGEN_DIR
   source vagen/envs/webarena/setup_vars.sh
   mkdir -p .wa_auth
   PYTHONPATH=. python vagen/envs/webarena/browser_env/auto_login.py \
     --auth_folder $(pwd)/.wa_auth \
     --site_list shopping,shopping_admin,reddit,gitlab
EOF
fi

# ---------------------------------------------------------------- 7. smoke test (opt-in)
if [ "$SKIP_SMOKE" = 0 ] && should_run smoke; then
  step "Smoke test (assumes WebArena server already running on :8002)"
  if ! curl -s --max-time 3 --fail http://localhost:8002/health >/dev/null; then
    warn "http://localhost:8002/health not responding — start the env server first"
    warn "See examples/train/webarena/RUN_INSTRUCTIONS.md §1"
  else
    conda activate vagen
    export HF_HOME
    cd "$VAGEN_DIR"
    bash examples/train/webarena/train_smoke_qwen25_05b.sh \
      | tee /tmp/vagen_smoke.log \
      | grep --line-buffered -E "Training Progress|step:|reward|Error"
    conda deactivate
  fi
fi

# ---------------------------------------------------------------- summary
step "Summary"
ok "Setup complete (the parts this script can automate)."
echo
cat <<EOF
Next steps:
  1. Bring up the WebArena Docker stack (SETUP.md §3) — manual
  2. Run auto_login.py to bootstrap .wa_auth/ cookies (SETUP.md §7)
  3. Validate with smoke test (SETUP.md §8):
       cd $VAGEN_DIR
       conda activate webarena
       source vagen/envs/webarena/setup_vars.sh
       PYTHONPATH=. python -m vagen.envs.webarena.serve \\
         --task_config_file=vagen/envs/webarena/config_files/normalized_test.json \\
         --n_browsers=2 --max_contexts_per_browser=2 --port=8002 \\
         --auth_cache_dir=./.wa_auth &
       conda activate vagen
       export HF_HOME=$HF_HOME
       bash examples/train/webarena/train_smoke_qwen25_05b.sh
  4. Full training (SETUP.md §9 / RUN_INSTRUCTIONS.md):
       bash examples/train/webarena/train_webarena_grpo_qwen25_3b.sh
EOF
