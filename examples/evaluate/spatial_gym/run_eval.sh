#!/bin/bash
# Run SpatialGym evaluation with VAGEN.
# Usage:
#   ./run_eval.sh [config.yaml] [overrides...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="${1:-$SCRIPT_DIR/config.yaml}"
shift 2>/dev/null || true

cd "$SCRIPT_DIR/../../.."
python -m vagen.evaluate.run_eval --config "$CONFIG" "$@"
