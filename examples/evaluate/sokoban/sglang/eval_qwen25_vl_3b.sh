#!/usr/bin/env bash
# Evaluate Sokoban against a local SGLang server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/../config.yaml}"
V="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
# shellcheck source=../../common.sh
source "${SCRIPT_DIR}/../../common.sh"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
PORT="${PORT:-30000}"
DP_SIZE="${DP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SEED="${SEED:-42}"

fileroot="${VAGEN_EVAL_ROOT:-${V}/eval_runs}"
MODEL_NAME="${MODEL_NAME:-$(vagen_model_name "${MODEL_PATH}")}"
DUMP_DIR="${DUMP_DIR:-${fileroot}/rollouts/eval_sokoban/${MODEL_NAME}}"
LOG_DIR="${fileroot}/logs"
mkdir -p "$LOG_DIR" "$DUMP_DIR"
SERVER_LOG="${LOG_DIR}/sglang_server_$$.log"

python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model-path "$MODEL_PATH" \
  --dp-size "$DP_SIZE" \
  --tp "$TP_SIZE" \
  --mem-fraction-static "$MEM_FRACTION" \
  --context-length "$MAX_MODEL_LEN" \
  --random-seed "$SEED" \
  --enable-deterministic-inference \
  --trust-remote-code \
  --log-level warning \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" >/dev/null 2>&1 || true
  wait "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

source "${SCRIPT_DIR}/../../wait_for_server.sh"
wait_for_server

python -m vagen.evaluation --config "$CONFIG" \
  run.backend=sglang \
  backends.sglang.base_url="http://127.0.0.1:${PORT}/v1" \
  backends.sglang.model="$MODEL_PATH" \
  fileroot="$fileroot" \
  experiment.dump_dir="$DUMP_DIR" \
  "$@" \
  2>&1 | tee "${LOG_DIR}/eval_$$.log"
