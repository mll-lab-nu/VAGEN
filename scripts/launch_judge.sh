#!/usr/bin/env bash
# Launch the small state-reward judge alongside training.
# Override any setting as an environment variable, for example:
#   PORT=8124 TP=4 MEM=0.12 bash scripts/launch_judge.sh
set -euo pipefail

ENV=${ENV:-$(python3 -c 'import os, sys; print(os.path.dirname(os.path.dirname(sys.executable)))')}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
PORT=${PORT:-8123}
MEM=${MEM:-0.10}
BACKEND=${BACKEND:-sglang}
SEED=${SEED:-42}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flashinfer}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ -x "$ENV/bin/python" ]] || die "Python not found at $ENV/bin/python (set ENV=/path/to/env)"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_COUNT=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
else
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | awk '/^GPU / {n++} END {print n+0}')
fi
(( GPU_COUNT > 0 )) || GPU_COUNT=1
TP=${TP:-$GPU_COUNT}
(( TP <= GPU_COUNT )) || die "TP=$TP but only $GPU_COUNT GPU(s) are visible"

# FlashInfer may JIT kernels, so prefer a system toolkit with nvcc over a
# runtime-only CUDA package in the conda environment.
if [ -z "${CUDA_HOME:-}" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    CUDA_VERSION=$("$ENV/bin/python" -c 'import torch; print(torch.version.cuda or "")')
    for candidate in "/usr/local/cuda-$CUDA_VERSION" /usr/local/cuda; do
        if [[ -x "$candidate/bin/nvcc" ]]; then
            CUDA_HOME=$candidate
            break
        fi
    done
fi
if [ -z "${CUDA_HOME:-}" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    die "CUDA toolkit with nvcc not found; set CUDA_HOME to its install directory"
fi

export CUDA_HOME
export PATH="$ENV/bin:$CUDA_HOME/bin:$PATH"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:$ENV/targets/x86_64-linux/include:$ENV/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$ENV/lib:$ENV/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

case "$BACKEND" in
  sglang)
    exec "$ENV/bin/python" -m sglang.launch_server \
      --host 0.0.0.0 --model-path "$MODEL" --port "$PORT" --tp "$TP" \
      --mem-fraction-static "$MEM" --context-length 4096 \
      --attention-backend "$ATTENTION_BACKEND" --random-seed "$SEED" \
      --enable-deterministic-inference --log-level warning
    ;;
  vllm)
    exec "$ENV/bin/python" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --port "$PORT" --tensor-parallel-size "$TP" \
      --gpu-memory-utilization "$MEM" --max-model-len 4096 --seed "$SEED"
    ;;
  *) die "BACKEND must be sglang or vllm (got: $BACKEND)" ;;
esac
