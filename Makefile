# OvaScope shortcuts.  `make help` lists them.
PYTHON ?= python
# Only the admin key is read from .env (other secrets in that file are not exported to commands).
ADMIN_KEY_ENV = OVASCOPE_ADMIN_KEY="$$(grep -E '^OVASCOPE_ADMIN_KEY=' .env 2>/dev/null | head -1 | cut -d= -f2- | sed 's/[[:space:]]*\#.*//')"

.PHONY: help setup api web dev test test-web build docker-up docker-down docker-logs admin

help:            ## show this list
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

setup:           ## install Python + web dependencies, create .env and the database
	bash scripts/setup.sh

api:             ## run the API on :8000 (reload on code changes)
	$(ADMIN_KEY_ENV) $(PYTHON) -m uvicorn api.main:app --port 8000 --reload --reload-dir api --reload-dir src

web:             ## run the web app dev server on :5173 (needs the API)
	cd web && npm run dev

dev:             ## run API and web app together (Ctrl+C stops both)
	@trap 'kill 0' INT TERM EXIT; \
	$(ADMIN_KEY_ENV) $(PYTHON) -m uvicorn api.main:app --port 8000 & \
	(cd web && npm run dev) & \
	wait

test:            ## Python test suite
	$(PYTHON) -m pytest -q

test-web:        ## type-check, lint and build the web app
	cd web && npm run build && npx eslint .

build:           ## production build of the web app (web/dist)
	cd web && npm run build

docker-up:       ## build and start all containers (web :8080, API gateway :8000, Streamlit :8501)
	docker compose up -d --build

docker-down:     ## stop the containers (data on the host is kept)
	docker compose down

docker-logs:     ## follow the container logs
	docker compose logs -f

admin:           ## create an administrator account from the command line
	@read -p "Username: " u; read -p "Email: " e; $(PYTHON) -m src.users create-admin --username "$$u" --email "$$e"
