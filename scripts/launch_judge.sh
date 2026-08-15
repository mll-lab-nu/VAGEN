#!/bin/bash
# A small instruct model that turns the agent's descriptions into structured items.
#
# It shares every GPU with training rather than taking one for itself: sharded eight
# ways a 4B model is about a gigabyte of weights per device, so reserving a whole card
# to hold it wastes an eighth of the node. It takes a small fixed fraction and starts
# first, leaving the rollout engine to size itself against what remains.
#
# Instruct-only on purpose: a thinking model would spend its budget reasoning about a
# format conversion, on the critical path of every turn of every rollout.
#
# vllm, matching the cluster entrypoint. Two engines for one job is two sets of failures.
set -eo pipefail
# The conda env the judge runs in. Defaults to the one this shell is already using.
ENV=${ENV:-$(python3 -c 'import sys, os; print(os.path.dirname(os.path.dirname(sys.executable)))' 2>/dev/null || echo "$CONDA_PREFIX")}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
PORT=${PORT:-8123}
# ★ Defaults to the number of visible GPUs, not to 8. Hardcoded, this script died on any
# smaller node with an engine-init error that never mentions tensor parallelism -- and the
# node size is the one thing here that is not a property of the model.
# nvidia-smi rather than torch: this runs before `export PATH="$ENV/bin:$PATH"` below, so
# `python3` here is whatever the login shell has, and on a real node that is usually a
# python without torch -- which would report one GPU and silently size TP to 1.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  _gpus=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
  _gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
fi
[ "${_gpus:-0}" -ge 1 ] 2>/dev/null || _gpus=1
TP=${TP:-$_gpus}
if [ "$TP" -gt "$_gpus" ]; then
  echo "TP=$TP but only $_gpus GPU(s) are visible; vLLM would fail at engine init." >&2
  exit 1
fi
MEM=${MEM:-0.10}

# FlashInfer JITs kernels on first use, so CUDA_HOME must name a real toolkit with nvcc.
# The conda environment carries Python and CUDA runtime libraries but not necessarily the
# compiler; pointing CUDA_HOME at it made startup fail as `$ENV/bin/nvcc: No such file`.
if [ -z "${CUDA_HOME:-}" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  _cuda_version=$(
    "$ENV/bin/python" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true
  )
  for d in "/usr/local/cuda-${_cuda_version}" /usr/local/cuda; do
    if [ -x "$d/bin/nvcc" ]; then
      CUDA_HOME=$d
      break
    fi
  done
fi
if [ -z "${CUDA_HOME:-}" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "CUDA toolkit with nvcc not found; set CUDA_HOME=/path/to/cuda." >&2
  exit 1
fi

export CUDA_HOME
export PATH="$ENV/bin:$CUDA_HOME/bin:$PATH"
# Conda supplies the Python-facing runtime; the toolkit supplies nvcc, headers and lib64.
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:$ENV/targets/x86_64-linux/include:$ENV/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$ENV/lib:$ENV/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$ENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$MEM" \
  --max-model-len 4096 \
  # vLLM 0.22 removed --disable-log-requests; quiet is the default and the flag
  # that exists now is its opposite. Left in as a comment because the failure it
  # caused -- `error: unrecognized arguments` -- names the flag, not the version.
  # --enable-log-requests
