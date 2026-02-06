#!/usr/bin/env bash
set -euo pipefail

# ---------- Defaults / Paths ----------
fileroot="${fileroot:-"$HOME/projects/vagen"}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-"$SCRIPT_DIR/config.yaml"}"
PORT="${PORT:-30000}"
LOG_DIR="${LOG_DIR:-"$SCRIPT_DIR/logs"}"
mkdir -p "$LOG_DIR"

# ---------- Model / Server Config ----------
MODEL_NAME="${MODEL_NAME:-"qwen_25_vl_3b"}"
MODEL_PATH="${QWEN25_VL_7B_PATH:-"Qwen/Qwen2.5-VL-3B-Instruct"}"
DP_SIZE="${QWEN25_VL_7B_DP:-1}"
TP_SIZE="${QWEN25_VL_7B_TP:-1}"
MEM_FRACTION="${QWEN25_VL_7B_MEM:-0.80}"

DUMP_DIR="${DUMP_DIR:-"$fileroot/rollouts/${MODEL_NAME}"}"
mkdir -p "$DUMP_DIR"

SERVER_LOG="${LOG_DIR}/${MODEL_NAME}_server.log"
EVAL_LOG="${LOG_DIR}/${MODEL_NAME}_eval.log"

# ---------- Launch Server ----------
python3 -m sglang.launch_server \
  --host 0.0.0.0 \
  --log-level warning \
  --port "${PORT}" \
  --model-path "${MODEL_PATH}" \
  --dp-size "${DP_SIZE}" \
  --tp "${TP_SIZE}" \
  --trust-remote-code \
  --mem-fraction-static "${MEM_FRACTION}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

# ---------- Cleanup ----------
cleanup() {
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
  wait "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ---------- Wait for server to be ready ----------
# wait_for_server.sh must be in the same directory as this script
source "${SCRIPT_DIR}/wait_for_server.sh"
wait_for_server

# ---------- Run Eval ----------
python -m vagen.evaluate.run_eval --config "${CONFIG}" \
  run.backend=sglang \
  backends.sglang.base_url="http://127.0.0.1:${PORT}/v1" \
  backends.sglang.model="${MODEL_PATH}" \
  experiment.dump_dir="${DUMP_DIR}" \
  fileroot="${fileroot}" \
  "$@" \
  2>&1 | tee "${EVAL_LOG}"
