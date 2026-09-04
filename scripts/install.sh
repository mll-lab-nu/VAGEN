#!/bin/bash
# One-command install for VAGEN.
#
#   bash scripts/install.sh                  vLLM   (default, and the verified path)
#   BACKEND=sglang bash scripts/install.sh   SGLang, current stack (torch 2.11 / sglang 0.5.15)
#   BACKEND=sglang STACK=aws-a100 \
#       bash scripts/install.sh              SGLang, the version set verified end to end
#
# The two SGLang stacks are alternatives, not an upgrade path. The default one is
# what the Qwen3.5 path needs (transformers 5.x); aws-a100 is the set a full run
# was observed to complete on (transformers 4.57.1, which has no Qwen3.5). Pick by
# the model you are training. Versions and reasoning live in
# requirements/locks/sglang-a100-cu128.txt, not here -- one place to change.
#
# Assumes you are already in the conda env you want to install into. Safe to re-run.
#
# Pick one engine per environment. They are mutually exclusive, and not by preference:
# every (vllm, sglang) pair pins a different flashinfer patch version, so pip refuses
# them together. verl models them the same way, as separate extras. flashinfer itself
# needs no attention here -- each engine pulls the version it wants.
#
# The pins live in setup.py's extras_require, so there is one place that says which
# versions go together. This script does not call verl's
# scripts/install_vllm_sglang_mcore.sh, which was the previous approach and had three
# problems:
#   - it installs sglang 0.5.2, while verl requires 0.5.8+. verl's sglang rollout imports
#     `ContinueGenerationReqInput`, absent before 0.5.6, so `rollout.name=sglang` died
#     with an ImportError naming neither sglang nor a version.
#   - it sets no `set -e` and ends in an unconditional success echo, so any pip failure
#     inside it still exited 0 and flowed into the next step.
#   - it wgets a hardcoded cp312 flash-attn wheel, which silently no-ops if the download
#     fails and rules out every other Python version.
set -euo pipefail

V=$(cd "$(dirname "$0")/.." && pwd)
cd "$V"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }

export BACKEND=${BACKEND:-vllm}
export STACK=${STACK:-default}
export SKIP_ENGINE=${SKIP_ENGINE:-0}   # 1 = engine already installed; exported for the check below

LOCKFILE=""
POSTFILE=""
case "$STACK" in
    default)   ;;
    aws-a100)  LOCKFILE="$V/requirements/locks/sglang-a100-cu128.txt"
               POSTFILE="$V/requirements/locks/sglang-a100-cu128-post.txt"
               [ "$BACKEND" = sglang ] || die "STACK=aws-a100 is an SGLang stack; pass BACKEND=sglang"
               [ -f "$LOCKFILE" ] || die "lock file not found: $LOCKFILE" ;;
    *)         die "STACK must be 'default' or 'aws-a100', got '$STACK'" ;;
esac

case "$BACKEND" in
    vllm)   ;;
    sglang) warn "the sglang extra is not the verified path; the example scripts are tested on vLLM" ;;
    *)      die "BACKEND must be 'vllm' or 'sglang', got '$BACKEND'" ;;
esac

command -v python3 >/dev/null || die "no python3 on PATH"
python3 - <<'PY' || die "VAGEN needs Python 3.12"
import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)
PY

# ---------------------------------------------------------------------- 1. verl source
say "verl submodule"
# Probes for a file, not the directory: an uninitialised submodule leaves `verl/` present
# and empty, which passes every existence check and fails every import.
if [ ! -f "$V/verl/verl/trainer/config/ppo_trainer.yaml" ]; then
    git submodule update --init --recursive \
        || die "could not fetch the verl submodule; check the network, or clone with --recursive"
fi
[ -f "$V/verl/verl/trainer/config/ppo_trainer.yaml" ] || die "verl/ is still empty after submodule update"

# ------------------------------------------------------------ 2. vagen + engine extra
if [ "$SKIP_ENGINE" = "1" ]; then
    say "vagen (no engine; SKIP_ENGINE=1)"
    python3 -m pip install --no-cache-dir -e "$V"
else
    if [ -n "$LOCKFILE" ]; then
        say "pinned stack '$STACK' -- this is the long one"
        # The lock first, then vagen with --no-deps so setup.py's ranges cannot
        # re-resolve on top of it and move torch or transformers.
        python3 -m pip install --no-cache-dir -r "$LOCKFILE" \
            || die "the pinned stack did not resolve; see $LOCKFILE"
        # Deliberate overrides, second pass. These contradict a pin something in
        # the lock declares, so pip reports ResolutionImpossible if they share a
        # resolution and merely warns when applied after -- the warning is the
        # expected outcome. Currently: cuDNN, which torch pins below the floor
        # SGLang enforces.
        if [ -f "$POSTFILE" ]; then
            python3 -m pip install --no-cache-dir -r "$POSTFILE" \
                || die "the post-install overrides did not apply; see $POSTFILE"
        fi
        python3 -m pip install --no-cache-dir --no-deps -e "$V"
    else
        say "vagen[$BACKEND] -- this is the long one"
        # One pip call so the resolver sees torch, the engine and transformers together and
        # reports a conflict, rather than resolving in sequence and letting the last pin
        # silently downgrade torch under an already-built engine.
        python3 -m pip install --no-cache-dir -e "$V[$BACKEND]" \
            || die "the $BACKEND extra did not resolve. The two engines pin different flashinfer versions and cannot share an environment -- if the other one is already installed here, use a fresh env."
    fi
fi

# ------------------------------------------------------------------------ 3. verl
say "verl (--no-deps) and its own requirements"
# --no-deps: verl pins engine versions of its own that would re-resolve and undo step 2.
python3 -m pip install --no-cache-dir --no-deps -e "$V/verl"
# What --no-deps skipped. Not the engine -- that is step 2 -- just verl's plain deps.
python3 -m pip install --no-cache-dir \
    accelerate codetiming datasets dill hydra-core numpy pandas peft pyarrow \
    pybind11 pylatexenc ray tensordict torchdata wandb

# ---------------------------------------------------------------------- verification
say "checking the install"
python3 - <<'PY'
import importlib, importlib.util, os, sys

backend = os.environ.get("BACKEND", "vllm")
skipped = os.environ.get("SKIP_ENGINE") == "1"

mods = ["torch", "transformers", "trl", "verl", "vagen"]
if not skipped:
    mods.insert(1, backend)

problems = []
for mod in mods:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<14} {getattr(m, '__version__', 'ok')}")
    except Exception as exc:
        problems.append(f"{mod}: {exc}")

# The floors that otherwise fail late and blame something else.
try:
    from packaging.version import parse
    import transformers
    if parse(transformers.__version__) < parse("5.2.0"):
        problems.append(f"transformers {transformers.__version__} < 5.2.0; AutoConfig raises "
                        f"KeyError('qwen3_5') for Qwen3.5")
    try:
        import torchao
        if parse(torchao.__version__) < parse("0.16.0"):
            problems.append(f"torchao {torchao.__version__} < 0.16.0; peft raises on it, breaking LoRA "
                            f"(uninstalling torchao also works)")
    except ImportError:
        pass          # absent is fine; peft only objects to a stale one
except Exception as exc:
    problems.append(f"version check failed: {exc}")

# verl hardcodes attn_implementation="flash_attention_2" when it builds the critic's
# value head, so one of these two has to be able to answer for it. The flash-attn package
# is the direct route but has no wheel for torch 2.11; `kernels` lets transformers fetch a
# prebuilt kernels-community/flash-attn2 from the Hub instead. With neither, the run dies
# several minutes in, at model load, saying only that flash-attn is not installed.
if not importlib.util.find_spec("flash_attn") and not importlib.util.find_spec("kernels"):
    problems.append("no flash attention: install `kernels` (pip install -e '.[vllm]' does) "
                    "so transformers can use kernels-community/flash-attn2, or build flash-attn")

if not skipped:
    # The import whose absence is how a too-old engine shows up: as a ModuleNotFoundError
    # or ImportError from deep inside verl, naming neither the engine nor a version.
    if backend == "vllm":
        try:
            importlib.import_module("vllm.entrypoints.openai.parser")
        except Exception:
            problems.append("vllm is too old for this verl: it lacks "
                            "vllm.entrypoints.openai.parser. Install vllm>=0.18.0.")
    else:
        try:
            from sglang.srt.managers.io_struct import ContinueGenerationReqInput  # noqa: F401
        except Exception:
            problems.append("sglang is too old for this verl: it lacks "
                            "ContinueGenerationReqInput, added in 0.5.6. Install sglang>=0.5.6.")

# --- runtime-linkage checks -------------------------------------------------
# Import success is not enough: each of these fails at model load or first
# generation instead, several frames from anything that names the cause.

# Torch <-> CUDA. A wheel built for a different CUDA major than the driver
# imports fine and dies on the first kernel.
try:
    import torch
    print(f"  torch cuda     {torch.version.cuda} (driver reports "
          f"{'available' if torch.cuda.is_available() else 'NO DEVICE'})")
    if not torch.cuda.is_available():
        problems.append("torch cannot see a GPU; check the wheel's CUDA build against the driver")
except Exception as exc:
    problems.append(f"torch cuda check failed: {exc}")

# libcudart.so.12 has to be loadable, not merely present -- sglang's compiled
# extensions link it.
import ctypes.util, ctypes
try:
    ctypes.CDLL("libcudart.so.12")
    print("  libcudart.so.12 loadable")
except OSError as exc:
    problems.append(f"libcudart.so.12 is not loadable: {exc}")

# flashinfer-python and flashinfer-cubin must be the same version; a mismatch
# raises from flashinfer/jit/env.py at import.
try:
    import flashinfer
    print(f"  flashinfer     {flashinfer.__version__}")
except Exception as exc:
    problems.append(f"flashinfer import failed (python/cubin version mismatch?): {exc}")

# cuDNN floor for torch 2.9.x: sglang refuses to start below 9.15
# (pytorch/pytorch#168167). torch itself pins 9.10, so pip warns -- the newer one
# is correct.
try:
    from packaging.version import parse
    import torch
    if parse(torch.__version__.split("+")[0]) < parse("2.10") and torch.backends.cudnn.version() < 91500:
        problems.append(f"cuDNN {torch.backends.cudnn.version()} < 9.15 with torch "
                        f"{torch.__version__}; sglang refuses to start. "
                        f"pip install nvidia-cudnn-cu12==9.16.0.29")
except Exception:
    pass

# verl is a checkout on PYTHONPATH, not an installed package; importing it is the
# only way to know the submodule is really there and really first.
try:
    import verl
    print(f"  verl           {getattr(verl, '__version__', 'ok')} ({verl.__file__})")
except Exception as exc:
    problems.append(f"verl does not import: {exc}")

if problems:
    print("\n\033[1;31mproblems:\033[0m")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("\n\033[1;32mok\033[0m")
PY

say "done"
cat <<'EOF'
Try a run:
  bash examples/train/sokoban/train_default_gae_qwen25vl3b.sh

Some environments need extra setup -- see their READMEs:
  vagen/envs/spatial_gym/README.md      dataset download
  vagen/envs/navigation/README.md       ai2thor
  vagen/envs/primitive_skill/README.md  maniskill
EOF
