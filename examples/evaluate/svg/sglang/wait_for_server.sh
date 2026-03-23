
#!/usr/bin/env bash
set -euo pipefail

# This file defines: wait_for_server
#
# Required env vars:
#   PORT, SERVER_PID, SERVER_LOG, MODEL_PATH
#
# Optional env vars:
#   SERVER_MAX_WAIT (default 1800)
#   SERVER_READY_INTERVAL (default 5)
#   STRICT_READY (default 0)
#   SERVER_LOG_TAIL_EVERY (default 60)

wait_for_server() {
  local base_url="http://127.0.0.1:${PORT}/v1"
  local ready_url="${base_url}/models"

  local max_wait="${SERVER_MAX_WAIT:-1800}"
  local interval="${SERVER_READY_INTERVAL:-5}"
  local strict="${STRICT_READY:-0}"

  local elapsed=0
  echo "[INFO] Waiting for server to be ready at ${base_url}"
  echo "[INFO] max_wait=${max_wait}s interval=${interval}s strict=${strict}"

  while true; do
    # If the server process died, fail fast
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      echo "[ERROR] Server process exited unexpectedly."
      echo "[ERROR] Last 200 lines of server log (${SERVER_LOG}):"
      tail -n 200 "${SERVER_LOG}" || true
      return 1
    fi

    if [[ "${strict}" == "1" ]]; then
      # Strict check: run a tiny inference
      if curl -s --fail "${base_url}/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
          "model": "'"${MODEL_PATH}"'",
          "messages": [{"role":"user","content":"ping"}],
          "max_tokens": 1
        }' >/dev/null 2>&1; then
        echo "[INFO] Server is ready after ${elapsed}s (strict mode)."
        return 0
      fi
    else
      # Lightweight check: OpenAI-compatible models endpoint
      if curl -s --fail "${ready_url}" >/dev/null 2>&1; then
        echo "[INFO] Server is ready after ${elapsed}s."
        return 0
      fi
    fi

    if (( elapsed >= max_wait )); then
      echo "[ERROR] Server did not become ready within ${max_wait}s."
      echo "[ERROR] Last 200 lines of server log (${SERVER_LOG}):"
      tail -n 200 "${SERVER_LOG}" || true
      return 1
    fi

    # Periodically show log tail for visibility
    local log_tail_every="${SERVER_LOG_TAIL_EVERY:-60}"
    if (( elapsed % log_tail_every == 0 )) && (( elapsed != 0 )); then
      echo "[INFO] Still waiting... elapsed=${elapsed}s"
      tail -n 20 "${SERVER_LOG}" || true
    fi

    sleep "${interval}"
    elapsed=$((elapsed + interval))
  done
}
