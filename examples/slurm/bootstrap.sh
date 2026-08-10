#!/bin/bash
# Prepare a Slurm cluster to run VAGEN: code, python environment, model weights.
# Submit it as a CPU job -- environment installs and multi-gigabyte downloads do not
# belong on a login node.
#
#   CODE_ROOT=... DATA_ROOT=... sbatch --account=$SLURM_ACCOUNT --qos=$CPU_QOS \
#     --nodes=1 --cpus-per-task=16 --mem=64G --time=05:00:00 examples/slurm/bootstrap.sh
#
# Idempotent: every step checks for its own result, so a rerun resumes rather than redoes.
set -uo pipefail
: "${CODE_ROOT:?}" "${DATA_ROOT:?}"
VAGEN_REMOTE=${VAGEN_REMOTE:-https://github.com/JamesKrW/VAGEN.git}
VAGEN_BRANCH=${VAGEN_BRANCH:-vagen-v0.8.0}
VERL_REMOTE=${VERL_REMOTE:-https://github.com/JamesKrW/verl.git}
VERL_BRANCH=${VERL_BRANCH:-release/v0.8.0}
ENV_PREFIX=$DATA_ROOT/conda/verl
export HF_HOME=$DATA_ROOT/hf_home
# A stale pip in ~/.local is on sys.path for every python of the same minor version and
# will shadow what the environment owns.
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR=$DATA_ROOT/tmp/pip-cache
say(){ echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p "$DATA_ROOT"/{conda,hf_home,outputs,tmp}

say "code"
[ -d "$CODE_ROOT/VAGEN/.git" ] || git clone -q "$VAGEN_REMOTE" "$CODE_ROOT/VAGEN"
git -C "$CODE_ROOT/VAGEN" fetch -q origin "$VAGEN_BRANCH" && \
  git -C "$CODE_ROOT/VAGEN" checkout -q -B "$VAGEN_BRANCH" "origin/$VAGEN_BRANCH"
# verl must be a *sibling* of VAGEN, not a subdirectory: the launch scripts resolve it
# as ../verl, and it must be the fork's branch, which carries the VAGEN patches.
[ -d "$CODE_ROOT/verl/.git" ] || git clone -q -b "$VERL_BRANCH" "$VERL_REMOTE" "$CODE_ROOT/verl"
say "VAGEN $(git -C "$CODE_ROOT/VAGEN" rev-parse --short HEAD) | verl $(git -C "$CODE_ROOT/verl" rev-parse --short HEAD)"

say "conda env at $ENV_PREFIX"
# -p, not -n: -n puts the environment under the base install, i.e. in $HOME, where a
# 25 GB environment does not belong. conda-forge with --override-channels because the
# default channels block non-interactively on a terms-of-service prompt. `pip` explicitly,
# because conda-forge's python does not ship one and the fallback is a pip outside the
# environment that installs to the wrong place and fails three steps later.
source "$(conda info --base)/etc/profile.d/conda.sh"
[ -x "$ENV_PREFIX/bin/python" ] || \
  conda create -y -q -p "$ENV_PREFIX" -c conda-forge --override-channels \
    python=3.12 pip setuptools wheel
PY="$ENV_PREFIX/bin/python"

if ! $PY -c "import torch, vllm, flash_attn" 2>/dev/null; then
  say "pip install"
  # --no-deps: a pip freeze is already the full transitive closure, so the resolver has
  # nothing to add and only surfaces conflicts the working environment does not have.
  # --extra-index-url: the torch pin is a +cuXXX local version that only exists there.
  $PY -m pip install --no-deps \
      --extra-index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}" \
      -r "$CODE_ROOT/VAGEN/examples/slurm/requirements-frozen.txt" 2>&1 | tail -12
  # flash-attn ships as a prebuilt wheel keyed to the torch and python versions; building
  # it from source takes about an hour.
  $PY -m pip install --no-deps "${FLASH_ATTN_WHEEL:?set FLASH_ATTN_WHEEL to the release URL matching your torch/python}"
fi
$PY -c "import torch, vllm; print('torch', torch.__version__, '| vllm', vllm.__version__)"

say "models"
for M in Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen3-4B-Instruct-2507; do
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$M', max_workers=8)" \
    && say "$M ok"
done

say "verify"
PYTHONPATH="$CODE_ROOT/verl:$CODE_ROOT/VAGEN" $PY -c "
import vagen.custom_advantage, vagen.custom_loss
from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY, ADV_ESTIMATOR_REGISTRY
need |= {'turn_gspo','turn_ppo'} - set(POLICY_LOSS_REGISTRY)
assert not need, f'missing: {sorted(need)}'
print('BOOTSTRAP OK')"
