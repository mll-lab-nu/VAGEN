
#!/usr/bin/env bash
set -euo pipefail

wait_for_server() {
  local base_url="http://127.0.0.1:${PORT}/v1"
  local ready_url="${base_url}/models"
  local max_wait="${SERVER_MAX_WAIT:-1800}"
  local interval="${SERVER_READY_INTERVAL:-5}"
  local elapsed=0

  echo "[INFO] Waiting for server to be ready at ${base_url}"
  while true; do
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      echo "[ERROR] Server process exited unexpectedly."
      tail -n 200 "${SERVER_LOG}" || true
      return 1
    fi
    if curl -s --fail "${ready_url}" >/dev/null 2>&1; then
      echo "[INFO] Server is ready after ${elapsed}s."
      return 0
    fi
    if (( elapsed >= max_wait )); then
      echo "[ERROR] Server did not become ready within ${max_wait}s."
      tail -n 200 "${SERVER_LOG}" || true
      return 1
    fi
    if (( elapsed % 60 == 0 )) && (( elapsed != 0 )); then
      echo "[INFO] Still waiting... elapsed=${elapsed}s"
      tail -n 20 "${SERVER_LOG}" || true
    fi
    sleep "${interval}"
    elapsed=$((elapsed + interval))
  done
}
