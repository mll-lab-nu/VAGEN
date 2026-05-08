#!/usr/bin/env bash
# One-shot launcher for WebArena GRPO training on vast.ai single-node.
#
# Does:
#   1. Preflight (conda init, GPU check, dirs)
#   2. Verify SSH tunnel is up (ports 7770/7780/9999/8023/8888 reachable)
#   3. Optional: bootstrap auth cookies (.wa_auth/) if missing
#   4. Start 2 webarena env servers (train @ 8002, val @ 8003)
#   5. Wait for both /health
#   6. Launch training (smoke or full)
#
# Usage:
#   bash run_train.sh                     # full training (default)
#   bash run_train.sh --smoke             # 2-step smoke with Qwen 0.5B
#   bash run_train.sh --no-wandb          # disable wandb logger
#   bash run_train.sh --skip-servers      # assume servers already up
#   bash run_train.sh --restart-servers   # kill existing + restart
#
# Env vars (override defaults):
#   VAGEN_DIR     repo root           [default: /workspace/VAGEN-WEBAGENT]
#   VAGEN_ENV     training conda env  [default: vagen]
#   HF_HOME       HF cache            [default: /workspace/hf_cache]
#   WANDB_API_KEY wandb key (else use --no-wandb)

set -euo pipefail

# -------------------------------------------------------------- defaults
VAGEN_DIR="${VAGEN_DIR:-/workspace/VAGEN-WEBAGENT}"
VAGEN_ENV="${VAGEN_ENV:-vagen}"
WEBARENA_ENV="${WEBARENA_ENV:-webarena}"
HF_HOME="${HF_HOME:-/workspace/hf_cache}"

MODE="full"          # full | smoke
USE_WANDB=1
SKIP_SERVERS=0
RESTART_SERVERS=0
SKIP_TUNNEL_CHECK=0
SKIP_AUTH=0

for arg in "$@"; do
  case "$arg" in
    --smoke) MODE="smoke" ;;
    --no-wandb) USE_WANDB=0 ;;
    --skip-servers) SKIP_SERVERS=1 ;;
    --restart-servers) RESTART_SERVERS=1 ;;
    --skip-tunnel-check) SKIP_TUNNEL_CHECK=1 ;;
    --skip-auth) SKIP_AUTH=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown flag: $arg"; exit 2 ;;
  esac
done

# -------------------------------------------------------------- helpers
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; }
err()  { echo -e "${RED}✗${NC}  $*" >&2; exit 1; }
step() { echo; echo -e "${BLUE}═══ $* ═══${NC}"; }
have() { command -v "$1" >/dev/null 2>&1; }

# -------------------------------------------------------------- conda bootstrap
step "Conda bootstrap"
if ! have conda && [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ]; then
  export PATH="$(dirname "$CONDA_EXE"):$PATH"
fi
if ! have conda && [ -f "$HOME/.bashrc" ]; then
  set +u; source "$HOME/.bashrc" >/dev/null 2>&1 || true; set -u
fi
if ! have conda; then
  for p in /opt/conda /opt/miniconda3 "$HOME/miniconda3" /root/miniconda3; do
    [ -x "$p/bin/conda" ] && export PATH="$p/bin:$PATH" && break
  done
fi
have conda || err "conda not found"
CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
ok "conda at $(command -v conda)"

# -------------------------------------------------------------- preflight
step "Preflight"
[ -d "$VAGEN_DIR" ] || err "VAGEN_DIR=$VAGEN_DIR not found"
cd "$VAGEN_DIR"

if have nvidia-smi; then
  GPU_COUNT=$(nvidia-smi -L | wc -l)
  ok "GPUs: $GPU_COUNT"
  nvidia-smi -L | head -4
else
  err "nvidia-smi not available — no GPU?"
fi

mkdir -p examples/evaluate/webarena/logs

if [ "$USE_WANDB" = 1 ] && [ -z "${WANDB_API_KEY:-}" ]; then
  warn "WANDB_API_KEY not set — pass --no-wandb or export WANDB_API_KEY"
  warn "Continuing in 5s (Ctrl-C to abort) ..."
  sleep 5
fi

# -------------------------------------------------------------- tunnel check
if [ "$SKIP_TUNNEL_CHECK" = 0 ]; then
  step "SSH tunnel check"
  bad=0
  for p in 7770 7780 9999 8023 8888; do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:$p/" || echo "000")
    if [ "$code" = "000" ]; then
      warn "port $p: NO RESPONSE — tunnel may be down"; bad=1
    else
      ok "port $p: HTTP $code"
    fi
  done
  [ "$bad" = 1 ] && err "Tunnel incomplete. Re-run your ssh -L tunnel and try again, or pass --skip-tunnel-check."
fi

# -------------------------------------------------------------- auth cookies
if [ "$SKIP_AUTH" = 0 ] && [ ! -d .wa_auth ] || [ -z "$(ls -A .wa_auth 2>/dev/null)" ]; then
  step "Bootstrap webarena auth cookies (.wa_auth/)"
  conda activate "$WEBARENA_ENV"
  source vagen/envs/webarena/setup_vars.sh
  mkdir -p .wa_auth
  PYTHONPATH=. python vagen/envs/webarena/browser_env/auto_login.py \
    --auth_folder "$(pwd)/.wa_auth" \
    --site_list shopping shopping_admin reddit gitlab
  ok ".wa_auth populated"
  conda deactivate
fi

# -------------------------------------------------------------- env servers
servers_up() {
  curl -s --fail --max-time 3 http://localhost:8002/health >/dev/null \
    && curl -s --fail --max-time 3 http://localhost:8003/health >/dev/null
}

start_env_servers() {
  log "Killing leaked sessions + chromium ..."
  pkill -9 -f "vagen.envs.webarena.serve" 2>/dev/null || true
  pkill -9 -u "$USER" -f chromium 2>/dev/null || true
  sleep 3

  conda activate "$WEBARENA_ENV"
  source vagen/envs/webarena/setup_vars.sh

  log "Starting train server on :8002 (n_browsers=8 × 8 contexts) ..."
  nohup env PYTHONPATH=. python -m vagen.envs.webarena.serve \
    --task_config_file=vagen/envs/webarena/config_files/normalized_train.json \
    --n_browsers=8 --max_contexts_per_browser=8 \
    --port=8002 --auth_cache_dir=./.wa_auth \
    > examples/evaluate/webarena/logs/webarena_train_server.log 2>&1 &

  log "Starting val server on :8003 (n_browsers=4 × 8 contexts) ..."
  nohup env PYTHONPATH=. python -m vagen.envs.webarena.serve \
    --task_config_file=vagen/envs/webarena/config_files/normalized_test.json \
    --n_browsers=4 --max_contexts_per_browser=8 \
    --port=8003 --auth_cache_dir=./.wa_auth \
    > examples/evaluate/webarena/logs/webarena_val_server.log 2>&1 &

  log "Waiting for both servers /health (timeout 90s) ..."
  for _ in $(seq 1 30); do
    servers_up && { ok "Both env servers ready"; conda deactivate; return 0; }
    sleep 3; printf '.'
  done
  echo
  err "Env servers did not come up — check examples/evaluate/webarena/logs/"
}

if [ "$SKIP_SERVERS" = 0 ]; then
  if [ "$RESTART_SERVERS" = 1 ] || ! servers_up; then
    step "Webarena env servers"
    start_env_servers
  else
    step "Webarena env servers"
    ok "Both servers already up — reusing (pass --restart-servers to force restart)"
  fi
fi

# -------------------------------------------------------------- training
step "Launching training (mode=$MODE)"
conda activate "$VAGEN_ENV"
export HF_HOME
export PYTHONUNBUFFERED=1

if [ "$MODE" = "smoke" ]; then
  TRAIN_SCRIPT="examples/train/webarena/train_smoke_qwen25_05b.sh"
else
  TRAIN_SCRIPT="examples/train/webarena/train_webarena_grpo_qwen25_3b.sh"
fi
[ -f "$TRAIN_SCRIPT" ] || err "Training script not found: $TRAIN_SCRIPT"

EXTRA_OVERRIDES=()
if [ "$USE_WANDB" = 0 ]; then
  EXTRA_OVERRIDES+=("trainer.logger=['console']")
fi

LOGFILE="/tmp/vagen_train_$(date +%Y%m%d_%H%M%S).log"
log "Log: $LOGFILE"
log "Run: bash $TRAIN_SCRIPT ${EXTRA_OVERRIDES[*]:-}"
echo

if [ ${#EXTRA_OVERRIDES[@]} -gt 0 ]; then
  bash "$TRAIN_SCRIPT" "${EXTRA_OVERRIDES[@]}" 2>&1 | tee "$LOGFILE"
else
  bash "$TRAIN_SCRIPT" 2>&1 | tee "$LOGFILE"
fi
