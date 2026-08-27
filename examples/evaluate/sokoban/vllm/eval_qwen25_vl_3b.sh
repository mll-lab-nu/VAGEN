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
# shellcheck source=../../common.sh
source "${SCRIPT_DIR}/../../common.sh"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
PORT="${PORT:-8311}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.55}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_IMAGES="${MAX_IMAGES:-8}"

fileroot="${VAGEN_EVAL_ROOT:-${V}/eval_runs}"
# ★ The model name is in the dump directory, and it has to be. Rollouts are keyed on
# (env, seed, tag_id, model) and resume skips on a match, so evaluating checkpoint B into
# the directory checkpoint A used would skip jobs and blend summaries. See
# vagen_model_name for why `basename` is not enough: every verl checkpoint's basename is
# the literal string `huggingface`.
MODEL_NAME="${MODEL_NAME:-$(vagen_model_name "${MODEL_PATH}")}"
DUMP_DIR="${DUMP_DIR:-${fileroot}/rollouts/eval_sokoban/${MODEL_NAME}}"
LOG_DIR="${fileroot}/logs"; mkdir -p "$LOG_DIR" "$DUMP_DIR"

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
