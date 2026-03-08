#!/usr/bin/env bash
set -euo pipefail

# ---------- Defaults / Paths ----------
fileroot="${fileroot:-"$HOME/projects/vagen"}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-"$SCRIPT_DIR/config.yaml"}"
LOG_DIR="${LOG_DIR:-"$SCRIPT_DIR/logs"}"
mkdir -p "$LOG_DIR"

# ---------- Model Config ----------
MODEL_NAME="${MODEL_NAME:-"gpt-4.1"}"
OPENAI_API_KEY="${OPENAI_API_KEY:?'Please set OPENAI_API_KEY'}"

DUMP_DIR="${DUMP_DIR:-"/work/nvme/bgig/ryu4/rollouts/${MODEL_NAME}"}"
mkdir -p "$DUMP_DIR"

EVAL_LOG="${LOG_DIR}/${MODEL_NAME}_eval.log"

# ---------- Run Eval ----------
# WebArena config files use relative paths like ./.auth/; cd so they resolve
cd /work/nvme/bgig/ryu4/webarena

python -m vagen.evaluate.run_eval --config "${CONFIG}" \
  run.backend=openai \
  backends.openai.api_key="${OPENAI_API_KEY}" \
  backends.openai.model="${MODEL_NAME}" \
  experiment.dump_dir="${DUMP_DIR}" \
  fileroot="${fileroot}" \
  "$@" \
  2>&1 | tee "${EVAL_LOG}"
