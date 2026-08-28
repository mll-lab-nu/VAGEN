#!/usr/bin/env bash
# Shared helpers for the per-environment eval launchers. Sourced, not run.
#
# These were copied into each launcher, and the copies drifted: only one of five derived a
# model name from the checkpoint path, and it derived the wrong one.

# ★ A name that distinguishes two checkpoints of the same run.
#
# verl writes a checkpoint to
#     <exp>/verl_checkpoints/global_step_<N>/actor/huggingface
# so `basename` -- which is what the sokoban launcher used, and what docs/evaluation.md
# tells you to point at -- returns the literal string `huggingface` for every step. The
# other four launchers hardcoded a constant. Either way MODEL_NAME is the dump directory
# and the summary is per-directory, so evaluating step 400 overwrote step 200's summary.json
# with no warning, and the rollouts underneath told a different story than the summary.
#
# Everything after the step directory is fixed layout, so the run's own name is what makes
# this readable: sokoban_default_gae_qwen25vl3b-global_step_400.
vagen_model_name() {
  local path="${1%/}" step exp
  # `best_actor` as well as `global_step_N`: trainer.val_log keeps the best-validating actor
  # under <exp>/verl_checkpoints/best_actor/actor/huggingface, and docs/evaluation.md tells
  # the reader that is where the checkpoint to score lives. Matching only global_step_ sent
  # every run's best actor to the same `huggingface` directory.
  step="$(printf '%s' "$path" | grep -oE 'global_step_[0-9]+|best_actor' | tail -1 || true)"
  if [ -z "$step" ]; then
    # A local directory outside verl's layout: its own name is the informative part.
    # Otherwise a HuggingFace hub id (Qwen/Qwen2.5-VL-3B-Instruct), where `/` would open a
    # directory level in the dump path, so flatten it.
    if [ -d "$path" ]; then basename "$path"; else printf '%s' "${path//\//_}"; fi
    return
  fi
  # The run's own name: the directory holding the step, skipping a `checkpoints` level if
  # that is what it turns out to be. Covers both layouts -- <exp>/verl_checkpoints/<step>/…
  # (what the scripts here configure) and verl's own default
  # checkpoints/<project>/<exp>/<step>/…, where the run name is one level nearer.
  local head
  head="$(printf '%s' "$path" | sed -E "s#/${step}(/.*)?\$##")"
  exp="${head##*/}"
  case "$exp" in
    checkpoints|checkpoint|verl_checkpoints|verl_checkpoint) exp="${head%/*}"; exp="${exp##*/}" ;;
  esac
  # A run directory called `checkpoints` at the filesystem root leaves this empty; the step
  # alone is still unique per step, which is the property that matters.
  [ "$exp" = "$path" ] && exp=""
  printf '%s' "${exp:+${exp}-}${step}"
}

# Start a vLLM OpenAI-compatible server and block until it answers, or exit non-zero
# saying where the log is. Sets SERVER_PID and installs the cleanup trap.
#
# ★ The default port is NOT 8000. navigation's and primitive_skill's environment servers
# (vagen/envs/*/serve.py) default to 8000 themselves, so a policy server there collides
# with the environment the policy is meant to be acting in.
vagen_serve_vllm() {
  local model="$1" port="$2" tp="$3" mem="$4" max_len="$5" max_images="$6" log="$7"
  local -a reasoning_args=()
  if [ -n "${VLLM_REASONING_CONFIG:-}" ]; then
    reasoning_args=(--reasoning-config "$VLLM_REASONING_CONFIG")
  fi

  # ★ vLLM builds custom kernels through ninja even under --enforce-eager. Without it the
  # server dies at engine startup with a bare `FileNotFoundError: 'ninja'`, several frames
  # below anything that mentions vLLM.
  command -v ninja >/dev/null 2>&1 || {
    echo "ninja not found: pip install ninja" >&2; return 1; }

  echo "[server] starting vLLM on :${port}, log -> ${log}"
  python -m vllm.entrypoints.openai.api_server \
    --model "$model" --port "$port" \
    --tensor-parallel-size "$tp" \
    --gpu-memory-utilization "$mem" \
    --max-model-len "$max_len" \
    --enforce-eager \
    --limit-mm-per-prompt "{\"image\":${max_images}}" \
    "${reasoning_args[@]}" \
    > "$log" 2>&1 &
  SERVER_PID=$!

  # The server outlives the launcher otherwise, holding the GPU until someone notices.
  # ★ `wait`, not just `kill`: vLLM runs the model in a separate EngineCore process and
  # takes seconds to release device memory. Returning the instant SIGTERM is delivered
  # means a second invocation profiles its memory fraction while the first still holds it
  # and OOMs at engine init -- which is what a loop over checkpoints does.
  #
  # ★ This REPLACES any EXIT trap the caller already installed -- bash has one per signal.
  # None of the shipped launchers set one before calling this, but a script that does would
  # lose it silently, so refuse rather than discard.
  local existing
  existing="$(trap -p EXIT)"
  if [ -n "$existing" ]; then
    echo "vagen_serve_vllm installs an EXIT trap to reap the server, but one is already " >&2
    echo "set and bash keeps only the last: ${existing}" >&2
    echo "Call this before installing your own, and chain to _vagen_reap_server." >&2
    return 1
  fi
  # shellcheck disable=SC2317
  _vagen_reap_server() {
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  }
  trap _vagen_reap_server EXIT

  local i
  for i in $(seq 1 90); do
    curl -sf -m 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && break
    kill -0 "${SERVER_PID}" 2>/dev/null || { echo "server died, see ${log}" >&2; return 1; }
    sleep 10
  done
  curl -sf -m 3 "http://127.0.0.1:${port}/v1/models" >/dev/null || {
    echo "server did not come up in 15 minutes, see ${log}" >&2; return 1; }
  echo "[server] up"
}

# Environments that talk to their own HTTP server (navigation, primitive_skill) cannot be
# evaluated without it, and the failure without one is a connection error from inside an
# env reset, several frames from anything naming the server.
vagen_require_env_server() {
  local url="$1" how="$2"
  curl -sf -m 3 "${url}" >/dev/null 2>&1 || \
  curl -sf -m 3 "${url}/health" >/dev/null 2>&1 || {
    echo "no environment server answering at ${url}." >&2
    echo "start one first:  ${how}" >&2
    return 1
  }
}
