#!/bin/bash
echo "=== WebArena Service Connectivity Test ==="

services=(
  "SHOPPING|http://localhost:7770"
  "SHOPPING_ADMIN|http://localhost:7780/admin"
  "GITLAB|http://localhost:8023"
  "REDDIT|http://localhost:9999"
  "WIKIPEDIA|http://localhost:8888"
  "HOMEPAGE|http://localhost:4399"
  "MAP|http://localhost:3000"
)

for svc in "${services[@]}"; do
  name="${svc%%|*}"
  url="${svc##*|}"
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$url")
  if [ "$code" -ge 200 ] 2>/dev/null && [ "$code" -lt 400 ] 2>/dev/null; then
    echo "[OK]   $name ($url) -> HTTP $code"
  else
    echo "[FAIL] $name ($url) -> HTTP $code"
  fi
done
