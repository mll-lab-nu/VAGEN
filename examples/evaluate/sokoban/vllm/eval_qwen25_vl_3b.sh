#!/bin/bash
# Evaluate on sokoban against a local vLLM server.
#
# The repo calls vLLM "the verified default" and makes the two engines mutually exclusive
# extras (setup.py), but every eval launcher shipped was an sglang one -- so following the
# install instructions and then the evaluation instructions left you with an engine no
# example could drive. vLLM's OpenAI-compatible server is what `run.backend=openai` already
# talks to; only the base_url changes.
#
#   MODEL_PATH=/path/to/model bash examples/evaluate/sokoban/vllm/eval_qwen25_vl_3b.sh
#
# Any extra argument is forwarded to run_eval as a hydra override, so a different context
# policy is one flag:
#
#   ... eval_qwen25_vl_3b.sh 'envs.0.harness=no_concat'
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/../config.yaml}"
V="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
PORT="${PORT:-8311}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.55}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_IMAGES="${MAX_IMAGES:-8}"

fileroot="${VAGEN_EVAL_ROOT:-${V}/eval_runs}"
LOG_DIR="${fileroot}/logs"; mkdir -p "$LOG_DIR"
SERVER_LOG="${LOG_DIR}/vllm_server_$$.log"

# ★ vLLM builds custom kernels through ninja even under --enforce-eager, and it is not a
# dependency of either engine extra. Without it the server dies at engine startup with a
# bare `FileNotFoundError: 'ninja'` several frames below anything that mentions vLLM.
command -v ninja >/dev/null 2>&1 || { echo "ninja not found: pip install ninja" >&2; exit 1; }

echo "[server] starting vLLM on :${PORT}, log -> ${SERVER_LOG}"
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${MEM_FRACTION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enforce-eager \
  --limit-mm-per-prompt "{\"image\":${MAX_IMAGES}}" \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
# The server outlives this script otherwise, holding the GPU until someone notices.
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 90); do
  curl -sf -m 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
  kill -0 "${SERVER_PID}" 2>/dev/null || { echo "server died, see ${SERVER_LOG}" >&2; exit 1; }
  sleep 10
done
curl -sf -m 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null || {
  echo "server did not come up in 15 minutes, see ${SERVER_LOG}" >&2; exit 1; }
echo "[server] up"

python -m vagen.evaluate.run_eval --config "${CONFIG}" \
  run.backend=openai \
  backends.openai.base_url="http://127.0.0.1:${PORT}/v1" \
  backends.openai.model="${MODEL_PATH}" \
  backends.openai.api_key=EMPTY \
  fileroot="${fileroot}" \
  "$@" \
  2>&1 | tee "${LOG_DIR}/eval_$$.log"
