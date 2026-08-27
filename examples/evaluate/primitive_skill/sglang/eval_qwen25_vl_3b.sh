#!/usr/bin/env bash
set -euo pipefail

# Before running, start the primitive_skill server in another terminal:
#   python -m vagen.envs.primitive_skill.serve --port 8000

# ---------- Defaults / Paths ----------
# VAGEN_EVAL_ROOT first, matching the vllm launcher and the eval configs; a
# developer's home directory is not a default.
fileroot="${VAGEN_EVAL_ROOT:-${fileroot:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/eval_runs}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-"$SCRIPT_DIR/../config.yaml"}"
# shellcheck source=../../common.sh
source "${SCRIPT_DIR}/../../common.sh"
PORT="${PORT:-30000}"
LOG_DIR="${LOG_DIR:-"$SCRIPT_DIR/logs"}"
mkdir -p "$LOG_DIR"

# ---------- Model / Server Config ----------
MODEL_PATH="${QWEN25_VL_3B_PATH:-"Qwen/Qwen2.5-VL-3B-Instruct"}"
# ★ Derived from the checkpoint path, not a constant. Hardcoded, every
# checkpoint of a run landed in one dump directory and overwrote the last
# summary.json -- see vagen_model_name in examples/evaluate/common.sh.
MODEL_NAME="${MODEL_NAME:-$(vagen_model_name "${MODEL_PATH}")}"
DP_SIZE="${QWEN25_VL_3B_DP:-4}"
TP_SIZE="${QWEN25_VL_3B_TP:-1}"

# ★ Clamp to what is actually visible. These defaulted to a full node's worth of data
# parallelism, and on a smaller machine sglang failed at engine init with a message that
# never mentions dp-size -- the node size is the one setting here that is not a property
# of the model.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  _gpus=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
  _gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
fi
[ "${_gpus:-0}" -ge 1 ] 2>/dev/null || _gpus=1
if [ $((DP_SIZE * TP_SIZE)) -gt "$_gpus" ]; then
  echo "dp-size=$DP_SIZE x tp=$TP_SIZE needs $((DP_SIZE * TP_SIZE)) GPUs, but $_gpus are visible." >&2
  echo "Lower it, e.g. DP=1:  QWEN25_VL_3B_DP=1 bash $0" >&2
  exit 1
fi

MEM_FRACTION="${QWEN25_VL_3B_MEM:-0.80}"

DUMP_DIR="${DUMP_DIR:-"$fileroot/rollouts/eval_primitive_skill/${MODEL_NAME}"}"
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
source "${SCRIPT_DIR}/../../wait_for_server.sh"
wait_for_server

# ---------- Run Eval ----------
python -m vagen.evaluation --config "${CONFIG}" \
  run.backend=sglang \
  backends.sglang.base_url="http://127.0.0.1:${PORT}/v1" \
  backends.sglang.model="${MODEL_PATH}" \
  experiment.dump_dir="${DUMP_DIR}" \
  fileroot="${fileroot}" \
  "$@" \
  2>&1 | tee "${EVAL_LOG}"
