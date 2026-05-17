#!/usr/bin/env bash
# Evaluate weizhepei SFT baseline on WebArena via vagen.evaluate.run_eval.
# Designed for vast.ai single-node (no SLURM).
#
# Usage:
#   bash run_eval_vastai.sh sglang     # serve with sglang, then eval
#   bash run_eval_vastai.sh vllm       # serve with vllm,   then eval
#
# Optional env vars:
#   MODEL_PATH    HF id or local path  [default: weizhepei/Qwen2.5-3B-WebArena-Lite-SFT-epoch-5]
#   N_ENVS        episodes to run      [default: 8]
#   MAX_TURNS     max turns per task   [default: 5]
#   SGLANG_PORT   sglang serve port    [default: 30000]
#   VLLM_PORT     vllm serve port      [default: 8001]
#   WEBARENA_PORT webarena env port    [default: 8002]
#   GPU_MEM       backend mem frac     [default: 0.85]
#   REPO_ROOT     repo path            [default: /workspace/VAGEN-WEBAGENT]
#   SKIP_WEBARENA assume env server up [default: 0]

set -euo pipefail

BACKEND="${1:-}"
if [ "$BACKEND" != "sglang" ] && [ "$BACKEND" != "vllm" ]; then
  echo "Usage: $0 <sglang|vllm>"; exit 2
fi

# -------------------------------------------------------------- defaults
REPO_ROOT="${REPO_ROOT:-/workspace/VAGEN-WEBAGENT}"
MODEL_PATH="${MODEL_PATH:-weizhepei/Qwen2.5-3B-WebArena-Lite-SFT-epoch-5}"
N_ENVS="${N_ENVS:-8}"
MAX_TURNS="${MAX_TURNS:-5}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
VLLM_PORT="${VLLM_PORT:-8001}"
WEBARENA_PORT="${WEBARENA_PORT:-8002}"
GPU_MEM="${GPU_MEM:-0.85}"
SKIP_WEBARENA="${SKIP_WEBARENA:-0}"

LOG_DIR="${REPO_ROOT}/examples/evaluate/webarena/logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
SERVE_LOG="${LOG_DIR}/${BACKEND}_serve_${STAMP}.log"
WA_LOG="${LOG_DIR}/webarena_eval_${STAMP}.log"
EVAL_LOG="${LOG_DIR}/eval_${BACKEND}_${STAMP}.log"

# -------------------------------------------------------------- helpers
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; }
err()  { echo -e "${RED}✗${NC}  $*" >&2; exit 1; }

# -------------------------------------------------------------- conda bootstrap
if ! command -v conda >/dev/null 2>&1; then
  if [ -n "${CONDA_EXE:-}" ]; then export PATH="$(dirname "$CONDA_EXE"):$PATH"; fi
  if [ -f "$HOME/.bashrc" ]; then set +u; source "$HOME/.bashrc" >/dev/null 2>&1 || true; set -u; fi
fi
command -v conda >/dev/null 2>&1 || err "conda not found"
CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# -------------------------------------------------------------- preflight
[ -d "$REPO_ROOT" ] || err "REPO_ROOT=$REPO_ROOT not found"
cd "$REPO_ROOT"
nvidia-smi -L | head -2

# -------------------------------------------------------------- start backend (sglang or vllm)
conda activate vagen
[ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/bin" ] && export PATH="$CONDA_PREFIX/bin:$PATH"
VAGEN_PY="$CONDA_PREFIX/bin/python"

start_sglang() {
  log "Starting sglang serve (port $SGLANG_PORT) ..."
  # Blackwell needs flashinfer backend (FA3 not supported on SM>=100)
  nohup env CUDA_VISIBLE_DEVICES=0 "$VAGEN_PY" -m sglang.launch_server \
    --host 0.0.0.0 --port "$SGLANG_PORT" \
    --model-path "$MODEL_PATH" \
    --dp-size 1 --tp 1 \
    --trust-remote-code \
    --mem-fraction-static "$GPU_MEM" \
    --attention-backend flashinfer \
    --log-level warning \
    > "$SERVE_LOG" 2>&1 &
  SERVE_PID=$!
  SERVE_URL="http://127.0.0.1:${SGLANG_PORT}/v1/models"
}

start_vllm() {
  log "Starting vllm serve (port $VLLM_PORT) ..."
  nohup env CUDA_VISIBLE_DEVICES=0 "$VAGEN_PY" -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port "$VLLM_PORT" \
    --model "$MODEL_PATH" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEM" \
    --enforce-eager \
    --max-model-len 32768 \
    > "$SERVE_LOG" 2>&1 &
  SERVE_PID=$!
  SERVE_URL="http://127.0.0.1:${VLLM_PORT}/v1/models"
}

if [ "$BACKEND" = "sglang" ]; then
  start_sglang
  BACKEND_BASE_URL="http://127.0.0.1:${SGLANG_PORT}/v1"
else
  start_vllm
  BACKEND_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
fi
conda deactivate

# -------------------------------------------------------------- start webarena env server
if [ "$SKIP_WEBARENA" = 0 ]; then
  if curl -s --fail --max-time 3 "http://localhost:${WEBARENA_PORT}/health" >/dev/null; then
    ok "webarena server already up on :${WEBARENA_PORT}"
    WEBARENA_PID=""
  else
    log "Starting webarena env server on :${WEBARENA_PORT} ..."
    conda activate webarena
    [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/bin" ] && export PATH="$CONDA_PREFIX/bin:$PATH"
    WEBARENA_PY="$CONDA_PREFIX/bin/python"
    source vagen/envs/webarena/setup_vars.sh
    nohup env PYTHONPATH=. "$WEBARENA_PY" -m vagen.envs.webarena.serve \
      --task_config_file=vagen/envs/webarena/config_files/normalized_test.json \
      --n_browsers=2 --max_contexts_per_browser=4 \
      --port="${WEBARENA_PORT}" --auth_cache_dir=./.wa_auth \
      > "$WA_LOG" 2>&1 &
    WEBARENA_PID=$!
    conda deactivate
  fi
fi

cleanup() {
  log "cleanup"
  [ -n "${SERVE_PID:-}" ] && kill "$SERVE_PID" 2>/dev/null || true
  [ -n "${WEBARENA_PID:-}" ] && kill "$WEBARENA_PID" 2>/dev/null || true
}
trap cleanup EXIT

# -------------------------------------------------------------- wait for both ready
wait_url() {
  local url="$1" name="$2" max="${3:-300}" pid="${4:-}"
  local elapsed=0
  while ! curl -s --fail "$url" >/dev/null 2>&1; do
    if [ -n "$pid" ] && ! kill -0 "$pid" >/dev/null 2>&1; then
      err "$name died — last 60 lines of its log:\n$(tail -60 "$SERVE_LOG" "$WA_LOG" 2>/dev/null)"
    fi
    [ $elapsed -ge $max ] && err "$name not ready in ${max}s"
    sleep 5; elapsed=$((elapsed + 5))
    [ $((elapsed % 30)) -eq 0 ] && log "still waiting for $name (${elapsed}s)"
  done
  ok "$name ready after ${elapsed}s"
}

wait_url "$SERVE_URL" "${BACKEND}_serve" 600 "$SERVE_PID"
[ "$SKIP_WEBARENA" = 0 ] && wait_url "http://localhost:${WEBARENA_PORT}/health" "webarena_serve" 120 "${WEBARENA_PID:-}"

# -------------------------------------------------------------- run eval
log "Launching vagen.evaluate.run_eval (backend=$BACKEND) ..."
conda activate vagen
[ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/bin" ] && export PATH="$CONDA_PREFIX/bin:$PATH"

PYTHONPATH=. python -m vagen.evaluate.run_eval \
  --config "${REPO_ROOT}/examples/evaluate/webarena/config.yaml" \
  run.backend="$BACKEND" \
  "backends.${BACKEND}.base_url=${BACKEND_BASE_URL}" \
  "backends.${BACKEND}.model=${MODEL_PATH}" \
  "envs.0.n_envs=${N_ENVS}" \
  "envs.0.max_turns=${MAX_TURNS}" \
  "envs.0.config.base_urls=http://localhost:${WEBARENA_PORT}" \
  "experiment.dump_dir=${REPO_ROOT}/exps/eval_webarena_${BACKEND}_${STAMP}" \
  2>&1 | tee "$EVAL_LOG"

ok "Eval done. Logs:"
echo "  serve:    $SERVE_LOG"
echo "  webarena: $WA_LOG"
echo "  eval:     $EVAL_LOG"
echo "  dump:     ${REPO_ROOT}/exps/eval_webarena_${BACKEND}_${STAMP}/"
