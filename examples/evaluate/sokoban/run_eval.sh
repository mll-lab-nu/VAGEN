#!/bin/bash
# Run evaluation with VAGEN environments
# Usage: ./run_eval.sh [config.yaml] [overrides...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config.yaml}"
shift 2>/dev/null

cd "$SCRIPT_DIR/../.."
python -m vagen.evaluate.run_eval --config "$CONFIG" "$@"
