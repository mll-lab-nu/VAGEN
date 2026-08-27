#!/bin/bash
# Evaluate on SpatialGym against a local vLLM server.
#
# setup.py calls vLLM "the verified default" and the two engines are mutually exclusive
# extras, but spatial_gym shipped no launcher at all -- so following the install instructions and then the
# evaluation instructions left you with an engine no example here could drive.
#
#   MODEL_PATH=/path/to/model bash examples/evaluate/spatial_gym/vllm/eval_qwen25_vl_3b.sh
#
# Any extra argument is forwarded to run_eval as a hydra override:
#
#   ... eval_qwen25_vl_3b.sh 'envs.0.harness=no_concat'
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/../config_1room.yaml}"
V="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
# shellcheck source=../../common.sh
source "${SCRIPT_DIR}/../../common.sh"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
# Not 8000: keeps the policy server clear of the environment servers other envs use.
PORT="${PORT:-8311}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.55}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_IMAGES="${MAX_IMAGES:-16}"

fileroot="${VAGEN_EVAL_ROOT:-${V}/eval_runs}"
# The env name belongs in the path: rollouts are keyed on (env, seed, tag, model) and
# resume skips on a match, so two envs sharing a directory skip each other.
MODEL_NAME="${MODEL_NAME:-$(vagen_model_name "${MODEL_PATH}")}"
DUMP_DIR="${DUMP_DIR:-${fileroot}/rollouts/eval_spatial_gym/${MODEL_NAME}}"
LOG_DIR="${fileroot}/logs"; mkdir -p "$LOG_DIR" "$DUMP_DIR"

# ★ The room dataset is not tracked in git (vagen/envs/spatial_gym/room_data is gitignored),
# so a fresh clone dies inside the first env reset with a bare file-not-found.
if [ ! -d "$V/vagen/envs/spatial_gym/room_data/1-room" ]; then
  echo "spatial_gym needs its room dataset, which is not tracked in git:" >&2
  echo "  hf download yw12356/spatial_gym_dataset --repo-type dataset \\" >&2
  echo "     --local-dir $V/vagen/envs/spatial_gym/room_data" >&2
  exit 1
fi

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
