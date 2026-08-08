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
# vllm, matching the MAST entrypoint. Two engines for one job is two sets of failures.
set -eo pipefail
# The conda env the judge runs in. Defaults to the one this shell is already using.
ENV=${ENV:-$(python3 -c 'import sys, os; print(os.path.dirname(os.path.dirname(sys.executable)))' 2>/dev/null || echo "$CONDA_PREFIX")}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
PORT=${PORT:-8123}
TP=${TP:-8}
MEM=${MEM:-0.10}

export PATH="$ENV/bin:$PATH"
export CUDA_HOME="$ENV"
# flashinfer JITs its sampling kernel on first use, and needs to find both the headers
# and the runtime library. conda keeps headers under targets/ and the libraries in lib/;
# without these the build dies as "cannot find -lcudart", several frames below a message
# that only says "Engine core initialization failed".
export CPLUS_INCLUDE_PATH="$ENV/targets/x86_64-linux/include:$ENV/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$ENV/lib:$ENV/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$ENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$MEM" \
  --max-model-len 4096 \
  --disable-log-requests
