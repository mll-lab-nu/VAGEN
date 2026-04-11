#!/usr/bin/env bash
set -euo pipefail

# Before running, start the primitive_skill server in another terminal:
#   python -m vagen.envs.primitive_skill.serve

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config.yaml}"
shift 2>/dev/null || true

LOG_FILE="run.log"

python -m vagen.evaluate.run_eval --config "$CONFIG" "$@" \
  2>&1 | tee "${LOG_FILE}"
