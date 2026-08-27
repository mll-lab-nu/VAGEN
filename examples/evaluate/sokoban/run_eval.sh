#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config.yaml}"
shift 2>/dev/null || true

# Beside the results, not in whatever directory you happened to be in.
LOG_DIR="${VAGEN_EVAL_ROOT:-$SCRIPT_DIR/eval_runs}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$$.log"

python -m vagen.evaluation --config "$CONFIG" "$@" \
  2>&1 | tee "${LOG_FILE}"
