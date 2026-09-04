#!/usr/bin/env bash
# Install the pinned SGLang candidate stack for VAGEN + verl.
#
#   conda create -p /path/to/env python=3.12 -y && conda activate /path/to/env
#   bash scripts/install_sglang.sh
#
# This is also the default SGLang path selected by scripts/install.sh. It installs
# torch 2.11.0 / SGLang 0.5.13 / transformers 5.8.1. That Transformers version
# is pinned inside verl's supported range: >=5.5.3, !=5.6.0, <5.11.
#
# Versions live in requirements/locks/, not here, so there is one place to change
# them and this file stays a procedure.
set -euo pipefail

V=$(cd "$(dirname "$0")/.." && pwd)
LOCK="$V/requirements/locks/sglang-cu130.txt"
POST="$V/requirements/locks/sglang-cu130-post.txt"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }

command -v python3 >/dev/null || die "no python3 on PATH"
python3 - <<'PY' || die "this stack is built for Python 3.12"
import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)
PY
[ -f "$LOCK" ] || die "lock file not found: $LOCK"
[ -f "$POST" ] || die "override file not found: $POST"

PIP="python3 -m pip install --no-cache-dir --timeout 180 --retries 10"

# ---------------------------------------------------------------- 1. verl source
say "verl submodule"
# Probes for a file, not the directory: an uninitialised submodule leaves `verl/`
# present and empty, which passes every existence check and fails every import.
if [ ! -f "$V/verl/verl/trainer/config/ppo_trainer.yaml" ]; then
    git -C "$V" submodule update --init --recursive \
        || die "could not fetch the verl submodule; check the network"
fi
[ -f "$V/verl/verl/trainer/config/ppo_trainer.yaml" ] || die "verl/ is still empty"

# ------------------------------------------------------------------- 2. the lock
say "pinned stack -- this is the long one"
$PIP -r "$LOCK" || die "the pinned stack did not resolve; see $LOCK"

# trl's numpy upper bound is stale relative to verl, so install only the package
# after the resolved runtime stack is in place.
say "second pass: critic value-head package"
$PIP --no-deps -r "$POST" || die "the second pass failed; see $POST"

# ------------------------------------------------------------ 3. vagen and verl
say "vagen and verl (--no-deps, so the pins above stand)"
$PIP --no-deps -e "$V"
$PIP --no-deps -e "$V/verl"
# verl's own plain dependencies, minus anything the lock already fixes.
# Deliberately no peft here: the lock pins it, and repeating it unpinned invites
# pip to re-resolve torch underneath. Everything below is pure Python.
$PIP codetiming dill pybind11 pylatexenc fire ninja cachetools \
     gym-sokoban gymnasium "uvicorn<0.41"

# ---------------------------------------------------------------- verification
say "checking the install"
python3 - <<'PY'
import ctypes, importlib, importlib.util, sys

problems = []


def report(name, value):
    print(f"  {name:<22} {value}")


# Versions the rest of this only makes sense against.
for mod in ["torch", "torchvision", "sglang", "flashinfer", "transformers", "trl", "accelerate", "ray"]:
    try:
        report(mod, getattr(importlib.import_module(mod), "__version__", "ok"))
    except Exception as exc:
        problems.append(f"{mod}: {exc}")

# Torch <-> CUDA. A wheel built for a different CUDA major than the driver imports
# fine and dies on the first kernel.
try:
    import torch

    report("torch cuda", f"{torch.version.cuda} / device {'yes' if torch.cuda.is_available() else 'NO'}")
    if not torch.cuda.is_available():
        problems.append("torch cannot see a GPU; check the wheel's CUDA build against the driver")
except Exception as exc:
    problems.append(f"torch cuda check failed: {exc}")

# libcudart.so.13 must be loadable, not merely present -- SGLang's compiled
# extensions link it.
try:
    ctypes.CDLL("libcudart.so.13")
    report("libcudart.so.13", "loadable")
except OSError as exc:
    problems.append(f"libcudart.so.13 is not loadable: {exc}")

# flashinfer-python and flashinfer-cubin must match exactly; a mismatch raises
# from flashinfer/jit/env.py at import, naming both versions.
try:
    importlib.import_module("flashinfer")
except Exception as exc:
    problems.append(f"flashinfer import failed (python/cubin mismatch?): {exc}")

try:
    import torch
    cudnn = torch.backends.cudnn.version()
    report("cudnn", cudnn)
except Exception as exc:
    problems.append(f"cudnn check failed: {exc}")

# The import whose absence is how a too-old SGLang shows up: as an ImportError
# from deep inside verl that names neither SGLang nor a version.
try:
    from sglang.srt.managers.io_struct import ContinueGenerationReqInput  # noqa: F401

    report("sglang symbol", "ContinueGenerationReqInput ok")
except Exception:
    problems.append("sglang is too old for this verl: no ContinueGenerationReqInput (needs >= 0.5.6)")

# verl is a checkout on PYTHONPATH, not an installed distribution; importing it is
# the only way to know the submodule is really there.
for mod in ["verl", "vagen"]:
    try:
        m = importlib.import_module(mod)
        report(mod, m.__file__)
    except Exception as exc:
        problems.append(f"{mod} does not import: {exc}")

if problems:
    print("\n\033[1;31mproblems:\033[0m")
    for problem in problems:
        print("  -", problem)
    sys.exit(1)
print("\n\033[1;32mok\033[0m")
PY

say "done"
echo "This environment is ready for the pinned VAGEN SGLang 0.5.13 candidate stack."
