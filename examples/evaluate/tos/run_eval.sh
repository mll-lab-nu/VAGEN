#!/bin/bash
# Run TOS evaluation with VAGEN.
# Usage:
#   ./run_eval.sh [config.yaml] [overrides...]
#   ./run_eval.sh summary [summary args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIRST_ARG="${1:-}"

if [ "$FIRST_ARG" = "summary" ]; then
  shift
  python "$SCRIPT_DIR/summarize.py" --rollout_dir "$SCRIPT_DIR/rollouts" "$@"
  exit 0
fi

CONFIG="${1:-$SCRIPT_DIR/config.yaml}"
shift 2>/dev/null || true

cd "$SCRIPT_DIR/../../.."
python -m vagen.evaluate.run_eval --config "$CONFIG" "$@"
