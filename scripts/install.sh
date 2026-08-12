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
SKIP_ENGINE=${SKIP_ENGINE:-0}     # set to 1 if vllm/sglang are already installed

command -v python3 >/dev/null || die "no python3 on PATH"
python3 - <<'PY' || die "VAGEN needs Python 3.10 or newer"
import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)
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
    ( cd "$V/verl" && USE_MEGATRON=$USE_MEGATRON bash scripts/install_vllm_sglang_mcore.sh )
fi

# ------------------------------------------------------------------------ 3. verl
say "verl (--no-deps)"
# --no-deps on purpose: verl pins versions of the engine stack that step 2 just resolved,
# and letting pip re-resolve them downgrades vllm underneath a working install.
pip install --no-deps -e "$V/verl"

# ----------------------------------------------------------------------- 4. vagen
say "vagen"
pip install -e "$V"

# ------------------------------------------------------------------------- 5. trl
say "trl"
pip install "trl==0.26.2"

# ---------------------------------------------------------------------- verification
say "checking the install"
python3 - <<'PY'
import importlib, sys

problems = []
for mod in ("torch", "vllm", "transformers", "verl", "vagen"):
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
