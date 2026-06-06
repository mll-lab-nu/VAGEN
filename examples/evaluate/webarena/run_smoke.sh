#!/usr/bin/env bash
# Smoke-test eval for WebArena env, using weizhepei/Qwen2.5-3B-WebArena-Lite-SFT-epoch-5
# as a baseline. 5 tasks, max 5 turns each. ~5-10 min end-to-end.
#
# Topology:
#   sglang server  -> GPU node (via srun --overlap into existing interactive job)
#   WebArena server -> this (login) node, since SSH tunnels live here
#   eval driver     -> this node, talks to both
#
# Required:
#   - SLURM_JOBID env var pointing to an existing GPU allocation, e.g.:
#       SLURM_JOBID=18118847 bash run_smoke.sh
#   - SSH tunnels for WebArena Docker on ports 7770/7780/9999/8023/8888/4399
#   - Conda envs:
#       webarena (Playwright + fastapi)  for WebArena server
#       vagen    (sglang + hydra)        for sglang server and eval driver
set -euo pipefail

: "${SLURM_JOBID:?Set SLURM_JOBID to your GPU job id (e.g. SLURM_JOBID=18118847)}"

REPO_ROOT="${REPO_ROOT:-/work/nvme/bgig/ryu4/VAGEN-WEBAGENT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

MODEL_PATH="${MODEL_PATH:-weizhepei/Qwen2.5-3B-WebArena-Lite-SFT-epoch-5}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
WEBARENA_PORT="${WEBARENA_PORT:-8002}"
GPU_NODE="${GPU_NODE:-$(squeue -j "${SLURM_JOBID}" -h -o %N | tr -d ' ')}"

SGLANG_LOG="${LOG_DIR}/sglang_server.log"
WEBARENA_LOG="${LOG_DIR}/webarena_server.log"
EVAL_LOG="${LOG_DIR}/eval.log"

CONDA_SH=/u/ryu4/miniconda3/etc/profile.d/conda.sh
HF_CACHE=/work/nvme/bgig/ryu4/huggingface_cache

echo "[smoke] GPU node: ${GPU_NODE}, sglang port: ${SGLANG_PORT}, webarena port: ${WEBARENA_PORT}"

# ---------------- Start sglang on the GPU node ----------------
echo "[smoke] Starting sglang on ${GPU_NODE}..."
srun --jobid="${SLURM_JOBID}" --overlap bash -c "
  source ${CONDA_SH}
  conda activate vagen
  export HF_HOME=${HF_CACHE}
  exec python -m sglang.launch_server \
    --host 0.0.0.0 --port ${SGLANG_PORT} \
    --model-path ${MODEL_PATH} \
    --dp-size 1 --tp 1 \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --log-level warning
" >"${SGLANG_LOG}" 2>&1 &
SGLANG_PID=$!

# ---------------- Start WebArena server on login node ---------
echo "[smoke] Starting WebArena server (this node)..."
(
  cd "${REPO_ROOT}"
  source ${CONDA_SH}
  conda activate webarena
  source vagen/envs/webarena/setup_vars.sh
  exec env PYTHONPATH=. python -m vagen.envs.webarena.serve \
    --task_config_file=vagen/envs/webarena/config_files/normalized_test.json \
    --n_browsers=2 --max_contexts_per_browser=2 \
    --port=${WEBARENA_PORT} \
    --auth_cache_dir=./vagen/envs/webarena/.wa_auth
) >"${WEBARENA_LOG}" 2>&1 &
WEBARENA_PID=$!

cleanup() {
  echo "[smoke] cleanup"
  kill "${SGLANG_PID}" "${WEBARENA_PID}" >/dev/null 2>&1 || true
  wait "${SGLANG_PID}" "${WEBARENA_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ---------------- Wait for both to be ready -------------------
wait_url() {
  local url="$1" name="$2" max="${3:-300}" pid="$4"
  local elapsed=0
  while ! curl -s --fail "$url" >/dev/null 2>&1; do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "[smoke] $name died — last 80 lines:"
      tail -n 80 "${LOG_DIR}/${name}.log" || true
      return 1
    fi
    if (( elapsed >= max )); then
      echo "[smoke] $name not ready in ${max}s"
      tail -n 80 "${LOG_DIR}/${name}.log" || true
      return 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    if (( elapsed % 30 == 0 )); then
      echo "[smoke] still waiting for $name (${elapsed}s)"
    fi
  done
  echo "[smoke] $name ready after ${elapsed}s"
}

wait_url "http://${GPU_NODE}:${SGLANG_PORT}/v1/models"     sglang_server   600 "${SGLANG_PID}"
wait_url "http://localhost:${WEBARENA_PORT}/health"        webarena_server 120 "${WEBARENA_PID}" \
  || wait_url "http://localhost:${WEBARENA_PORT}/"          webarena_server 30  "${WEBARENA_PID}" || true

# ---------------- Run eval ------------------------------------
echo "[smoke] Running eval..."
cd "${REPO_ROOT}"
source ${CONDA_SH}
conda activate vagen
PYTHONPATH=. python -m vagen.evaluate.run_eval \
  --config "${SCRIPT_DIR}/config.yaml" \
  run.backend=sglang \
  backends.sglang.base_url="http://${GPU_NODE}:${SGLANG_PORT}/v1" \
  backends.sglang.model="${MODEL_PATH}" \
  2>&1 | tee "${EVAL_LOG}"
