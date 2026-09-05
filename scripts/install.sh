#!/usr/bin/env bash
# Install VAGEN with one rollout backend.
#
#   bash scripts/install.sh               # SGLang (default, fully pinned)
#   BACKEND=vllm bash scripts/install.sh  # vLLM
#   SKIP_ENGINE=1 bash scripts/install.sh # keep an existing engine
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
export BACKEND=${BACKEND:-sglang}
export SKIP_ENGINE=${SKIP_ENGINE:-0}
PIP=("$PYTHON_BIN" -m pip install --no-cache-dir)

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

case "$BACKEND" in
    sglang|vllm) ;;
    *) die "BACKEND must be sglang or vllm (got: $BACKEND)" ;;
esac
[[ "$SKIP_ENGINE" == 0 || "$SKIP_ENGINE" == 1 ]] \
    || die "SKIP_ENGINE must be 0 or 1"
command -v "$PYTHON_BIN" >/dev/null || die "$PYTHON_BIN is not on PATH"
"$PYTHON_BIN" - <<'PY' || die "VAGEN requires Python 3.12"
import sys
raise SystemExit(sys.version_info[:2] != (3, 12))
PY

# SGLang needs an ordered install for its compiled packages; keep that logic in
# one place. vLLM can be resolved normally from setup.py.
if [[ "$BACKEND" == sglang && "$SKIP_ENGINE" == 0 ]]; then
    exec bash "$ROOT/scripts/install_sglang.sh"
fi

step "Prepare the verl submodule"
if [[ ! -f "$ROOT/verl/verl/trainer/config/ppo_trainer.yaml" ]]; then
    git -C "$ROOT" submodule update --init -- verl || die "could not fetch verl"
fi
[[ -f "$ROOT/verl/verl/trainer/config/ppo_trainer.yaml" ]] \
    || die "verl is empty; run: git submodule update --init -- verl"

if [[ "$SKIP_ENGINE" == 1 ]]; then
    step "Install VAGEN without changing the rollout engine"
    "${PIP[@]}" -e "$ROOT"
else
    step "Install VAGEN with $BACKEND"
    "${PIP[@]}" -e "${ROOT}[${BACKEND}]" \
        || die "$BACKEND could not be resolved; use a fresh environment (vLLM and SGLang cannot share one)"
fi

step "Install verl"
"${PIP[@]}" --no-deps -e "$ROOT/verl"
"${PIP[@]}" accelerate codetiming datasets dill hydra-core numpy pandas peft \
    pyarrow pybind11 pylatexenc ray tensorboard tensordict torchdata wandb

step "Verify the environment"
BACKEND="$BACKEND" SKIP_ENGINE="$SKIP_ENGINE" "$PYTHON_BIN" - <<'PY'
import importlib
import importlib.util
import os
import sys

backend = os.environ["BACKEND"]
skipped = os.environ["SKIP_ENGINE"] == "1"
problems = []

for name in ["torch", "transformers", "trl", "verl", "vagen"] + ([] if skipped else [backend]):
    try:
        module = importlib.import_module(name)
        print(f"  {name:<14} {getattr(module, '__version__', 'ok')}")
    except Exception as exc:
        problems.append(f"{name}: {exc}")

if not importlib.util.find_spec("flash_attn") and not importlib.util.find_spec("kernels"):
    problems.append("flash attention is unavailable; install `kernels` or flash-attn")

if not skipped:
    try:
        if backend == "vllm":
            importlib.import_module("vllm.entrypoints.openai.parser")
        else:
            from sglang.srt.managers.io_struct import ContinueGenerationReqInput  # noqa: F401
    except Exception as exc:
        problems.append(f"{backend}/verl API mismatch: {exc}")

if problems:
    print("\nProblems:")
    print("\n".join(f"  - {problem}" for problem in problems))
    sys.exit(1)
print("\nInstall verified.")
PY

step "Done"
echo "Try: bash examples/train/sokoban/train_default_gae_qwen3vl4b.sh"
