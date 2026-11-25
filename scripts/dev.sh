#!/usr/bin/env bash
# Start the API (:8000) and the web interface (:5173) together; Ctrl+C stops both.
# Same as `make dev`, for machines without make.
#
#     bash scripts/dev.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

# Read only the operator admin key from .env (other secrets in .env are not exported).
if [[ -f .env ]]; then
  key=$(grep -E '^OVASCOPE_ADMIN_KEY=' .env | head -1 | cut -d= -f2- | sed 's/[[:space:]]*#.*//')
  [[ -n "$key" ]] && export OVASCOPE_ADMIN_KEY="$key"
fi
[[ -d web/node_modules ]] || (cd web && npm ci --no-audit --no-fund --loglevel=error)

trap 'kill 0' INT TERM EXIT
"$PY" -m uvicorn api.main:app --port 8000 &
(cd web && npm run dev) &
echo
echo "API:      http://localhost:8000/docs"
echo "Web app:  http://localhost:5173   (first account created becomes the administrator)"
wait
