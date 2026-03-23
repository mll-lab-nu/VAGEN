  #!/usr/bin/env bash
set -euo pipefail

# Before running, start the SVG server in another terminal:
#   python -m vagen.envs.svg.serve --port 8002
#
# This script uses two conda envs:
#   - "sglang" env (CUDA 12.8) for the sglang inference server
#   - "vagen" env for the eval runner (which talks to both sglang and SVG server)

# ---------- Defaults / Paths ----------
fileroot="${fileroot:-"$HOME/projects/vagen"}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-"$SCRIPT_DIR/../config.yaml"}"
PORT="${PORT:-30000}"
LOG_DIR="${LOG_DIR:-"$SCRIPT_DIR/logs"}"
mkdir -p "$LOG_DIR"

# ---------- Conda env paths ----------
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/march/miniconda3/envs/sglang/bin/python3}"
VAGEN_PYTHON="${VAGEN_PYTHON:-/home/march/miniconda3/envs/vagen/bin/python}"

# ---------- Model / Server Config ----------
MODEL_NAME="${MODEL_NAME:-"qwen_25_vl_3b"}"
MODEL_PATH="${QWEN25_VL_3B_PATH:-"Qwen/Qwen2.5-VL-3B-Instruct"}"
DP_SIZE="${QWEN25_VL_3B_DP:-1}"
TP_SIZE="${QWEN25_VL_3B_TP:-1}"
MEM_FRACTION="${QWEN25_VL_3B_MEM:-0.85}"

DUMP_DIR="${DUMP_DIR:-"$fileroot/rollouts/eval_svg_${MODEL_NAME}"}"
mkdir -p "$DUMP_DIR"

SERVER_LOG="${LOG_DIR}/${MODEL_NAME}_server.log"
EVAL_LOG="${LOG_DIR}/${MODEL_NAME}_eval.log"

# ---------- Ensure sglang env's nvcc is found (for flashinfer JIT) ----------
SGLANG_ENV_DIR="$(dirname "$(dirname "${SGLANG_PYTHON}")")"
export PATH="${SGLANG_ENV_DIR}/bin:${PATH}"
export CUDACXX="${SGLANG_ENV_DIR}/bin/nvcc"
export CUDA_HOME="${SGLANG_ENV_DIR}"

# ---------- Launch Server ----------
${SGLANG_PYTHON} -m sglang.launch_server \
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
source "${SCRIPT_DIR}/wait_for_server.sh"
wait_for_server

# ---------- Run Eval ----------
${VAGEN_PYTHON} -m vagen.evaluate.run_eval --config "${CONFIG}" \
  run.backend=sglang \
  backends.sglang.base_url="http://127.0.0.1:${PORT}/v1" \
  backends.sglang.model="${MODEL_PATH}" \
  experiment.dump_dir="${DUMP_DIR}" \
  fileroot="${fileroot}" \
  "$@" \
  2>&1 | tee "${EVAL_LOG}"
