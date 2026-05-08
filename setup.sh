#!/usr/bin/env bash
# Automated setup for VAGEN-WEBAGENT on a fresh machine.
#
# Covers SETUP.md sections 2 (system pkgs), 4 (clone), 5 (envs), 6 (HF
# models), plus a GitHub SSH-key bootstrap step. Sections 3 (WebArena
# Docker) and 7 (auth bootstrap) require user-specific assets (Docker
# image tars or webarena-host credentials) and are left as manual
# steps with prompts at the end.
#
# Default paths land under /workspace so data survives instance destroy
# when /workspace is backed by a Vast.ai Volume (NOT default container
# storage — see https://docs.vast.ai/guides/instances/storage/types).
#
# Idempotent: re-runnable. Each step skips if its output already exists.
#
# Usage:
#   bash setup.sh                       # default: all steps, verbose
#   bash setup.sh --skip-system         # skip apt-get steps (no root)
#   bash setup.sh --skip-ssh            # skip GitHub SSH bootstrap
#   bash setup.sh --skip-models         # skip HF download
#   bash setup.sh --only=envs           # only build conda envs
#   REPO_ROOT=/path/to/repo bash setup.sh
#
# Env vars (override defaults):
#   REPO_ROOT       directory to clone into [default: /workspace]
#   VAGEN_REPO_URL  VAGEN repo to clone    [default: git@github.com:YuRuiii/VAGEN.git]
#   VAGEN_REPO_BRANCH branch / tag to use  [default: main]
#   HF_HOME         HF cache location      [default: /workspace/hf_cache]
#   VAGEN_ENV       conda env for training [default: vagen]
#                     Creates a fresh Python 3.12 env if missing. Override
#                     to install into an existing env (e.g. Vast.ai's
#                     "main"), but if that env already has a different
#                     vllm version the script will refuse without
#                     ALLOW_CLOBBER_VLLM=1 — VAGEN/verl are tied to
#                     vllm==0.11.0 and a downgrade may break the host env.
#   CONDA_BIN       path to conda binary   [default: auto-detect]
#   CUDA_INDEX_URL  pytorch wheel index    [default: cu128]

set -euo pipefail

# ---------------------------------------------------------------- defaults
REPO_ROOT="${REPO_ROOT:-/workspace}"
HF_HOME="${HF_HOME:-/workspace/hf_cache}"
VAGEN_ENV="${VAGEN_ENV:-vagen}"
VAGEN_REPO_URL="${VAGEN_REPO_URL:-git@github.com:YuRuiii/VAGEN.git}"
VAGEN_REPO_BRANCH="${VAGEN_REPO_BRANCH:-main}"
ALLOW_CLOBBER_VLLM="${ALLOW_CLOBBER_VLLM:-0}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
SKIP_SYSTEM=0
SKIP_SSH=0
SKIP_DOCKER_NOTES=0
SKIP_MODELS=0
SKIP_SMOKE=1   # smoke needs WebArena up; off by default
ONLY=""

for arg in "$@"; do
  case "$arg" in
    --skip-system) SKIP_SYSTEM=1 ;;
    --skip-ssh) SKIP_SSH=1 ;;
    --skip-docker-notes) SKIP_DOCKER_NOTES=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --run-smoke) SKIP_SMOKE=0 ;;
    --only=*) ONLY="${arg#--only=}" ;;
    -h|--help)
      sed -n '2,37p' "$0"; exit 0 ;;
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

# conda is usually a shell function defined by `conda init` in .bashrc, so it
# does NOT propagate to this `bash setup.sh` subshell. Try, in order:
#   1) $CONDA_EXE (exported by conda init in some setups)
#   2) source ~/.bashrc to pull in the conda init block
#   3) scan common install dirs
# Then source conda.sh so `conda activate` works in this script.
if ! have conda && [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ]; then
  export PATH="$(dirname "$CONDA_EXE"):$PATH"
  log "Picked up conda from \$CONDA_EXE: $CONDA_EXE"
fi

if ! have conda && [ -f "$HOME/.bashrc" ]; then
  log "Sourcing ~/.bashrc to load conda init block ..."
  set +u
  # shellcheck disable=SC1091
  source "$HOME/.bashrc" >/dev/null 2>&1 || true
  set -u
fi

if ! have conda; then
  for p in /opt/conda /opt/miniconda3 /opt/anaconda3 \
           "$HOME/miniconda3" "$HOME/anaconda3" \
           /root/miniconda3 /root/anaconda3 /usr/local/anaconda3; do
    if [ -x "$p/bin/conda" ]; then
      export PATH="$p/bin:$PATH"
      log "Found conda at $p/bin/conda — added to PATH"
      break
    fi
  done
fi

if have conda; then
  CONDA_BIN="${CONDA_BIN:-$(command -v conda)}"
  CONDA_BASE=$(conda info --base)
  ok "conda at $CONDA_BIN (base: $CONDA_BASE)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  err "conda not found. Try:  type conda  in your shell to see where it lives,
       then re-run with PATH explicitly:
         PATH=\$(dirname \$(type -P conda)):\$PATH bash setup.sh ..."
fi

# ---------------------------------------------------------------- 0.5 GitHub SSH key
if should_run ssh && [ "$SKIP_SSH" = 0 ]; then
  step "GitHub SSH key (skip with --skip-ssh)"

  SSH_DIR="$HOME/.ssh"
  SSH_KEY="$SSH_DIR/id_ed25519"
  mkdir -p "$SSH_DIR" && chmod 700 "$SSH_DIR"

  if [ ! -f "$SSH_KEY" ]; then
    log "Generating ed25519 key (no passphrase) ..."
    ssh-keygen -t ed25519 -N "" -f "$SSH_KEY" -C "vagen-$(hostname)-$(date +%Y%m%d)" -q
    ok "Generated $SSH_KEY"
  else
    ok "Existing key found at $SSH_KEY"
  fi

  if ! grep -q "^github.com " "$SSH_DIR/known_hosts" 2>/dev/null; then
    ssh-keyscan -t ed25519,rsa github.com 2>/dev/null >> "$SSH_DIR/known_hosts"
    ok "Pinned github.com in known_hosts"
  fi

  github_ssh_ok() {
    ssh -o BatchMode=yes -o ConnectTimeout=5 -T git@github.com 2>&1 \
      | grep -q "successfully authenticated"
  }

  github_port22_blocked() {
    ! timeout 5 bash -c '</dev/tcp/github.com/22' 2>/dev/null
  }

  enable_github_443_fallback() {
    if grep -q "Hostname ssh.github.com" "$SSH_DIR/config" 2>/dev/null; then
      ok "github.com → ssh.github.com:443 fallback already configured"
      return
    fi
    log "Outbound port 22 blocked — adding ssh.github.com:443 fallback to ~/.ssh/config"
    cat >> "$SSH_DIR/config" <<'EOF'

Host github.com
  Hostname ssh.github.com
  Port 443
  User git
EOF
    chmod 600 "$SSH_DIR/config"
    ssh-keyscan -p 443 -t ed25519,rsa ssh.github.com 2>/dev/null >> "$SSH_DIR/known_hosts"
    ok "443 fallback configured"
  }

  if ! github_ssh_ok && github_port22_blocked; then
    enable_github_443_fallback
  fi

  if github_ssh_ok; then
    ok "SSH to GitHub works"
  else
    warn "GitHub SSH not yet authorized. Add this public key at"
    warn "  https://github.com/settings/keys  (type: Authentication Key)"
    echo
    echo "  ──────────── BEGIN PUBLIC KEY ────────────"
    sed 's/^/  /' "$SSH_KEY.pub"
    echo "  ─────────────  END PUBLIC KEY  ───────────"
    echo
    if [ -t 0 ]; then
      read -r -p "  Press Enter once added to GitHub (or Ctrl-C to abort) ... " _
      if github_ssh_ok; then
        ok "SSH to GitHub now works"
      else
        err "Still cannot authenticate. Verify the key was pasted correctly, then re-run."
      fi
    else
      err "Not a TTY — cannot pause for key upload. Add key to GitHub then re-run setup.sh."
    fi
  fi
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
      log "Cloning $VAGEN_REPO_URL → $REPO_ROOT/VAGEN-WEBAGENT (branch $VAGEN_REPO_BRANCH) ..."
      git clone --branch "$VAGEN_REPO_BRANCH" "$VAGEN_REPO_URL" VAGEN-WEBAGENT
      VAGEN_DIR="$REPO_ROOT/VAGEN-WEBAGENT"
      ok "VAGEN-WEBAGENT cloned"
    fi
  else
    VAGEN_DIR="$REPO_ROOT/VAGEN-WEBAGENT"
  fi
  cd "$VAGEN_DIR"
  log "VAGEN_DIR=$VAGEN_DIR"

  # verl is a git submodule of VAGEN, pinned to JamesKrW/verl @ vagen-lite.
  # See .gitmodules. The official volcengine/verl does NOT contain the
  # commit VAGEN needs (compute_reward was refactored away upstream).
  if [ -f .gitmodules ] && grep -q "submodule \"verl\"" .gitmodules; then
    log "Initializing verl submodule (JamesKrW/verl @ vagen-lite) ..."
    git submodule update --init --recursive
    ok "verl submodule at $(cd verl && git rev-parse --short HEAD)"
  else
    warn "No verl submodule defined — falling back to manual clone of JamesKrW/verl@vagen-lite"
    if [ ! -d verl/.git ]; then
      rm -rf verl verl_src
      git clone --branch vagen-lite https://github.com/JamesKrW/verl.git verl
      ok "verl cloned (JamesKrW/verl @ vagen-lite)"
    else
      ok "verl/ already exists"
    fi
  fi

  if [ -f verl/verl/trainer/config/ppo_trainer.yaml ] \
     && grep -q "^def compute_reward" verl/verl/trainer/ppo/reward.py; then
    ok "verl Hydra config + compute_reward present"
  else
    err "verl tree looks wrong — missing ppo_trainer.yaml or compute_reward. Check submodule URL/branch."
  fi
fi

# ---------------------------------------------------------------- 3. vagen conda env
if should_run vagen-env; then
  step "Conda env: $VAGEN_ENV (Python 3.12, training)"

  if conda env list | awk '{print $1}' | grep -qx "$VAGEN_ENV"; then
    ok "$VAGEN_ENV env already exists (skipping create — installs go into it)"
  else
    log "Creating conda env $VAGEN_ENV ..."
    conda create -n "$VAGEN_ENV" python=3.12 -y -q
  fi

  conda activate "$VAGEN_ENV"
  PY_VER=$(python -c 'import sys; print("%d.%d"%sys.version_info[:2])')
  if [ "$PY_VER" != "3.12" ]; then
    err "$VAGEN_ENV has Python $PY_VER but training stack needs 3.12. Set VAGEN_ENV to a different env."
  fi
  ok "Active env: $VAGEN_ENV (Python $PY_VER)"

  EXISTING_VLLM=$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true)
  if [ -n "$EXISTING_VLLM" ] && [ "$EXISTING_VLLM" != "0.11.0" ]; then
    if [ "$ALLOW_CLOBBER_VLLM" != "1" ]; then
      err "$VAGEN_ENV already has vllm==$EXISTING_VLLM, but VAGEN/verl require 0.11.0.
       Re-running would downgrade vllm in this env and may break anything else
       that depends on it. Either:
         - use a fresh env:  VAGEN_ENV=vagen bash setup.sh   (recommended)
         - force-downgrade:  ALLOW_CLOBBER_VLLM=1 bash setup.sh"
    fi
    warn "Clobbering vllm $EXISTING_VLLM → 0.11.0 in $VAGEN_ENV (ALLOW_CLOBBER_VLLM=1)"
  fi

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

  # Source-build of flash-attn takes 30-60 min. Prefer Dao-AILab's prebuilt
  # wheel (must match python / torch / cuda / cxx11abi). The URL below is for
  # py3.12 + torch2.8 + cu12 + cxx11abi=FALSE (stock pip torch).
  FLASH_ATTN_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
  log "Installing flash-attn (prebuilt wheel) ..."
  if pip install -q "$FLASH_ATTN_WHL"; then
    ok "flash-attn installed from prebuilt wheel"
  else
    warn "Prebuilt wheel failed (404 or arch mismatch). Falling back to source build (~30-60 min)."
    warn "If you Ctrl-C, training still works but with reduced throughput."
    pip install flash-attn==2.8.1 --no-build-isolation || \
      warn "flash-attn build failed — try later with matching CUDA toolkit; training degrades but works without it"
  fi

  log "Installing verl (editable from ./verl submodule) ..."
  pip install -q -e "$VAGEN_DIR/verl"

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
    fire \
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

  conda activate "$VAGEN_ENV"
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
    conda activate "$VAGEN_ENV"
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
Persistence reminder:
  REPO_ROOT=$REPO_ROOT
  HF_HOME=$HF_HOME
  Conda envs live at \$CONDA_BASE/envs (NOT on /workspace by default —
  re-run setup.sh on a fresh instance to recreate them, or move
  miniconda itself onto /workspace).

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
       conda activate $VAGEN_ENV
       export HF_HOME=$HF_HOME
       bash examples/train/webarena/train_smoke_qwen25_05b.sh
  4. Full training (SETUP.md §9 / RUN_INSTRUCTIONS.md):
       bash examples/train/webarena/train_webarena_grpo_qwen25_3b.sh
EOF
