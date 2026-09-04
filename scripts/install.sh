#!/bin/bash
# One-command install for VAGEN.
#
#   bash scripts/install.sh                SGLang 0.5.13 (default)
#   BACKEND=vllm bash scripts/install.sh   vLLM
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
#   - it installs sglang 0.5.2, while verl requires a newer API. verl's rollout imports
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
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }

export BACKEND=${BACKEND:-sglang}
export SKIP_ENGINE=${SKIP_ENGINE:-0}   # 1 = engine already installed; exported for the check below

case "$BACKEND" in
    vllm)   ;;
    sglang) ;;
    *)      die "BACKEND must be 'vllm' or 'sglang', got '$BACKEND'" ;;
esac

command -v python3 >/dev/null || die "no python3 on PATH"
python3 - <<'PY' || die "VAGEN needs Python 3.12"
import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)
PY

# The verified SGLang stack needs a lock plus a second installation pass for cuDNN,
# flash-attn and trl. A setuptools extra cannot express that ordering, so keep the
# canonical procedure in one place.
if [ "$BACKEND" = "sglang" ] && [ "$SKIP_ENGINE" != "1" ]; then
    exec bash "$V/scripts/install_sglang.sh"
fi

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
    say "vagen[$BACKEND] -- this is the long one"
    # One pip call so the resolver sees torch, the engine and transformers together and
    # reports a conflict, rather than resolving in sequence and letting the last pin
    # silently downgrade torch under an already-built engine.
    python3 -m pip install --no-cache-dir -e "$V[$BACKEND]" \
        || die "the $BACKEND extra did not resolve. The two engines pin different flashinfer versions and cannot share an environment -- if the other one is already installed here, use a fresh env."
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

backend = os.environ.get("BACKEND", "sglang")
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
    problems.append("no flash attention: install `kernels` (either engine extra does) "
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
