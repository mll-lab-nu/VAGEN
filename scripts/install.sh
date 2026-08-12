#!/bin/bash
# One-command install for VAGEN.
#
#   bash scripts/install.sh
#
# Assumes you are already in the conda env you want to install into, and that the repo
# was cloned. Safe to re-run: every step is idempotent.
#
# What it installs, and why the order matters:
#   1. the verl submodule            everything else imports it
#   2. verl's engine stack           vllm / sglang, via verl's own installer
#   3. verl itself, --no-deps        its pins would fight the stack just installed
#   4. vagen                         picks up transformers / torchao floors from setup.py
#   5. trl                           pinned; newer versions moved what verl imports
set -euo pipefail

V=$(cd "$(dirname "$0")/.." && pwd)
cd "$V"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }

USE_MEGATRON=${USE_MEGATRON:-0}
export SKIP_ENGINE=${SKIP_ENGINE:-0}   # 1 = vllm/sglang already installed; exported so the verification block can see it

command -v python3 >/dev/null || die "no python3 on PATH"
# 3.12 specifically, not ">=3.10": the source parses under 3.10, but verl's installer
# fetches a cp312-only flash-attn wheel, so an older interpreter gets "not a supported
# wheel on this platform" partway through step 2 rather than a clear message here.
python3 - <<'PY' || die "VAGEN needs Python 3.12 (verl's flash-attn wheel is cp312-only)"
import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)
PY

# ---------------------------------------------------------------------- 1. verl source
say "verl submodule"
# Checking for a file, not the directory: an uninitialised submodule leaves an empty
# `verl/` behind, which every "is it there" test passes and every import fails.
if [ ! -f "$V/verl/verl/trainer/config/ppo_trainer.yaml" ]; then
    git submodule update --init --recursive \
        || die "could not fetch the verl submodule; check network or clone with --recursive"
fi
[ -f "$V/verl/verl/trainer/config/ppo_trainer.yaml" ] || die "verl/ is still empty after submodule update"

# ------------------------------------------------------------------- 2. engine stack
if [ "$SKIP_ENGINE" = "1" ]; then
    say "skipping the engine stack (SKIP_ENGINE=1)"
else
    say "vllm / sglang (via verl's installer; this is the long one)"
    # `bash -e`, not `bash`: set -e does not cross into a script invoked this way, and
    # verl's installer sets none of its own -- its last line is an unconditional
    # "Successfully installed all packages" echo, so a failed pip inside it exited 0 and
    # this step could not fail. The flash-attn step is the likely one: it is
    # `wget ... && pip install ...`, which silently does nothing if the download fails.
    ( cd "$V/verl" && USE_MEGATRON=$USE_MEGATRON bash -e scripts/install_vllm_sglang_mcore.sh )
    python3 -c "import vllm, flash_attn" 2>/dev/null \
        || die "the engine stack did not install cleanly (vllm / flash_attn do not import); see the output above"
fi

# ------------------------------------------------------------------------ 3. verl
say "verl (--no-deps)"
# --no-deps on purpose: verl pins versions of the engine stack that step 2 just resolved,
# and letting pip re-resolve them downgrades vllm underneath a working install.
python3 -m pip install --no-deps -e "$V/verl"

# ----------------------------------------------------------------------- 4. vagen
say "vagen"
python3 -m pip install -e "$V"

# ------------------------------------------------------------------------- 5. trl
say "trl"
python3 -m pip install "trl==0.26.2"

# ---------------------------------------------------------------------- verification
say "checking the install"
SKIP_ENGINE=$SKIP_ENGINE python3 - <<'PY'
import importlib, os, sys

problems = []
mods = ["torch", "transformers", "verl", "vagen"]
if os.environ.get("SKIP_ENGINE") != "1":
    mods.insert(1, "vllm")
for mod in mods:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<14} {getattr(m, '__version__', 'ok')}")
    except Exception as exc:
        problems.append(f"{mod}: {exc}")

# The two floors that fail late and blame something else. transformers below 4.57 raises
# KeyError('qwen3_vl') from AutoConfig; peft *raises* on a torchao older than 0.16 rather
# than skipping it, which breaks LoRA even though nothing here quantises.
try:
    from packaging.version import parse
    import transformers
    if parse(transformers.__version__) < parse("4.57.0"):
        problems.append(f"transformers {transformers.__version__} < 4.57.0; Qwen3-VL will not load")
    try:
        import torchao
        if parse(torchao.__version__) < parse("0.16.0"):
            problems.append(f"torchao {torchao.__version__} < 0.16.0; peft raises on it and LoRA breaks "
                            f"(uninstalling torchao also works)")
    except ImportError:
        pass          # absent is fine; peft only objects to a stale one
except Exception as exc:
    problems.append(f"version check failed: {exc}")

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
