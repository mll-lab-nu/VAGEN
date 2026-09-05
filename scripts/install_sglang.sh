#!/usr/bin/env bash
# Install the verified CUDA 13 stack:
#   Torch 2.11.0 / SGLang 0.5.13 / Transformers 5.8.1
#
# Usage (inside a fresh Python 3.12 environment):
#   bash scripts/install_sglang.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOCK="$ROOT/requirements/locks/sglang-cu130.txt"
POST_LOCK="$ROOT/requirements/locks/sglang-cu130-post.txt"
PYTHON_BIN=${PYTHON_BIN:-python3}
PIP=("$PYTHON_BIN" -m pip install --no-cache-dir --timeout 180 --retries 10)

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

command -v "$PYTHON_BIN" >/dev/null || die "$PYTHON_BIN is not on PATH"
"$PYTHON_BIN" - <<'PY' || die "this stack requires Python 3.12"
import sys
raise SystemExit(sys.version_info[:2] != (3, 12))
PY
[[ -f "$LOCK" && -f "$POST_LOCK" ]] || die "SGLang lock files are missing"

step "Prepare the verl submodule"
if [[ ! -f "$ROOT/verl/verl/trainer/config/ppo_trainer.yaml" ]]; then
    git -C "$ROOT" submodule update --init -- verl || die "could not fetch verl"
fi
[[ -f "$ROOT/verl/verl/trainer/config/ppo_trainer.yaml" ]] \
    || die "verl is empty; run: git submodule update --init -- verl"

step "Install the pinned runtime"
"${PIP[@]}" -r "$LOCK" || die "failed to install $LOCK"

# causal-conv1d may fall back from its GitHub wheel to a local build. Point that
# build at the system CUDA toolkit and reuse this environment's Torch.
if [[ -z "${CUDA_HOME:-}" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    CUDA_VERSION=$("$PYTHON_BIN" -c 'import torch; print(torch.version.cuda or "")')
    for candidate in "/usr/local/cuda-$CUDA_VERSION" /usr/local/cuda; do
        if [[ -x "$candidate/bin/nvcc" ]]; then
            CUDA_HOME=$candidate
            break
        fi
    done
fi
if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
    export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
fi
export MAX_JOBS=${MAX_JOBS:-16}

step "Install compiled and compatibility packages"
"${PIP[@]}" --no-build-isolation --no-deps -r "$POST_LOCK" \
    || die "failed to install $POST_LOCK (set CUDA_HOME to a toolkit with nvcc if causal-conv1d built from source)"

step "Install VAGEN and verl"
"${PIP[@]}" --no-deps -e "$ROOT"
"${PIP[@]}" --no-deps -e "$ROOT/verl"
"${PIP[@]}" codetiming dill pybind11 pylatexenc fire ninja cachetools \
    gym-sokoban gymnasium "uvicorn<0.41"

step "Verify imports and CUDA support"
"$PYTHON_BIN" - <<'PY'
import ctypes
import importlib
import sys
from packaging.version import Version

problems = []

def check(name, expected=None):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "ok")
        print(f"  {name:<24} {version}")
        if expected and Version(version).base_version != expected:
            problems.append(f"{name} is {version}, expected {expected}")
    except Exception as exc:
        problems.append(f"{name}: {exc}")

for name, version in {
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "sglang": "0.5.13",
    "flashinfer": "0.6.12",
    "transformers": "5.8.1",
    "trl": "0.9.6",
    "fla": "0.5.2",
    "causal_conv1d": "1.7.0",
}.items():
    check(name, version)

try:
    import torch
    print(f"  {'CUDA device':<24} {torch.cuda.get_device_name(0)}")
    if not torch.cuda.is_available():
        problems.append("Torch cannot see a CUDA device")
except Exception as exc:
    problems.append(f"CUDA check: {exc}")

try:
    ctypes.CDLL("libcudart.so.13")
except OSError as exc:
    problems.append(f"libcudart.so.13 is not loadable: {exc}")

try:
    from transformers.utils.import_utils import (
        is_causal_conv1d_available,
        is_flash_linear_attention_available,
    )
    if not is_flash_linear_attention_available():
        problems.append("Transformers cannot use flash-linear-attention")
    if not is_causal_conv1d_available():
        problems.append("Transformers cannot use causal-conv1d")
except Exception as exc:
    problems.append(f"Qwen3.5 fast-path check: {exc}")

try:
    from sglang.srt.managers.io_struct import ContinueGenerationReqInput  # noqa: F401
except Exception:
    problems.append("SGLang lacks ContinueGenerationReqInput (requires >=0.5.6)")

for name in ("verl", "vagen", "tensorboard", "tyro"):
    check(name)

if problems:
    print("\nProblems:")
    print("\n".join(f"  - {problem}" for problem in problems))
    sys.exit(1)
print("\nInstall verified.")
PY

step "Done"
echo "The VAGEN SGLang environment is ready."
