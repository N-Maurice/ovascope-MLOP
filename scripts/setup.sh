#!/usr/bin/env bash
# One-command local setup for OvaScope: checks the tools, installs Python and web dependencies,
# creates .env and the database. Safe to run again.
#
#     bash scripts/setup.sh            # uses the active Python (a conda env or venv is recommended)
#     bash scripts/setup.sh --no-python  # only the web interface
set -euo pipefail
cd "$(dirname "$0")/.."

ok()   { printf "  \033[32m✔\033[0m %s\n" "$1"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$1"; }
fail() { printf "  \033[31m✘\033[0m %s\n" "$1"; }

echo "1. Checking tools"
PY=${PYTHON:-python}
if command -v "$PY" >/dev/null 2>&1; then
  ver=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  [[ "$ver" == "3.12" ]] && ok "Python $ver" || warn "Python $ver found; the project is tested with 3.12"
else
  fail "Python not found (install Python 3.12 or Miniconda)"; exit 1
fi
if command -v node >/dev/null 2>&1; then
  major=$(node -v | sed 's/v\([0-9]*\).*/\1/')
  (( major >= 20 )) && ok "Node.js $(node -v)" || { fail "Node.js $(node -v) is too old; install Node 20 or newer"; exit 1; }
  ok "npm $(npm -v)"
else
  fail "Node.js/npm not found: install Node 20+ (e.g. 'sudo apt install nodejs npm' or https://nodejs.org)"; exit 1
fi
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  ok "Docker $(docker --version | cut -d' ' -f3 | tr -d ,) with Compose"
else
  warn "Docker not installed (only needed for containers): sudo bash scripts/install_system_tools.sh"
fi

if [[ "${1:-}" != "--no-python" ]]; then
  echo "2. Python packages (requirements.txt)"
  "$PY" -m pip install -q -r requirements.txt && ok "installed"
fi

echo "3. Web interface packages (web/package-lock.json)"
(cd web && npm ci --no-audit --no-fund --loglevel=error) && ok "installed"

echo "4. Configuration"
if [[ ! -f .env ]]; then
  key=$("$PY" -c 'import secrets; print(secrets.token_urlsafe(24))')
  sed "s/^OVASCOPE_ADMIN_KEY=.*/OVASCOPE_ADMIN_KEY=$key/" .env.example > .env
  ok "created .env with a random OVASCOPE_ADMIN_KEY"
else
  ok ".env already exists (left unchanged)"
fi

echo "5. Database"
"$PY" -c 'from src.config import load_config; from src.database import init_db; init_db(load_config()["paths"]["database"])' \
  && ok "db/ovascope.db ready (tables created or migrated)"

[[ -f models/active_model.json ]] && ok "active model: $("$PY" -c 'import json; print(json.load(open("models/active_model.json"))["version"])')" \
  || warn "no trained model yet: run the notebook or 'python -m src.train' first"

cat <<'EOF'

Done. Start everything with:
  bash scripts/dev.sh          # API on :8000 + web app on http://localhost:5173 (or: make dev)
or with Docker:
  docker compose up -d --build # web app on http://localhost:8080
Then open the web app, choose "Sign in" -> "Create account": the first account becomes the administrator.
EOF
