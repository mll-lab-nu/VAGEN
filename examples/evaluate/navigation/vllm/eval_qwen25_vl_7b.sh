#!/bin/bash
# Evaluate on Navigation against a local vLLM server.
#
# setup.py calls vLLM "the verified default" and the two engines are mutually exclusive
# extras, but navigation shipped only an sglang launcher -- so following the install instructions and then the
# evaluation instructions left you with an engine no example here could drive.
#
#   MODEL_PATH=/path/to/model bash examples/evaluate/navigation/vllm/eval_qwen25_vl_7b.sh
#
# Any extra argument is forwarded to run_eval as a hydra override:
#
#   ... eval_qwen25_vl_7b.sh 'envs.0.harness=no_concat'
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/../config.yaml}"
V="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
# shellcheck source=../../common.sh
source "${SCRIPT_DIR}/../../common.sh"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
# Not 8000: vagen/envs/navigation/serve.py defaults to 8000, and this eval needs it.
PORT="${PORT:-8311}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.55}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_IMAGES="${MAX_IMAGES:-16}"

fileroot="${VAGEN_EVAL_ROOT:-${V}/eval_runs}"
# The env name belongs in the path: rollouts are keyed on (env, seed, tag, model) and
# resume skips on a match, so two envs sharing a directory skip each other.
MODEL_NAME="${MODEL_NAME:-$(vagen_model_name "${MODEL_PATH}")}"
DUMP_DIR="${DUMP_DIR:-${fileroot}/rollouts/eval_navigation/${MODEL_NAME}}"
LOG_DIR="${fileroot}/logs"; mkdir -p "$LOG_DIR" "$DUMP_DIR"

# navigation's environment runs in its own HTTP server, and the eval config points at
# localhost:8000 for it. Without one the failure is a connection error raised inside the
# first env reset, several frames from anything that names a server.
vagen_require_env_server "http://localhost:8000" \
  "python -m vagen.envs.navigation.serve --port 8000"

vagen_serve_vllm "$MODEL_PATH" "$PORT" "$TP_SIZE" "$MEM_FRACTION" \
                 "$MAX_MODEL_LEN" "$MAX_IMAGES" "${LOG_DIR}/vllm_server_$$.log"

python -m vagen.evaluation --config "${CONFIG}" \
  run.backend=openai \
  backends.openai.base_url="http://127.0.0.1:${PORT}/v1" \
  backends.openai.model="${MODEL_PATH}" \
  backends.openai.api_key=EMPTY \
  fileroot="${fileroot}" \
  experiment.dump_dir="${DUMP_DIR}" \
  "$@" \
  2>&1 | tee "${LOG_DIR}/eval_$$.log"
