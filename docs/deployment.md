# Cloud deployment

**Status: not yet deployed.** This guide is the deployment plan and configuration; no public URL exists yet.
Fill in the URL in the README only after the steps below have been carried out and verified.

## Why a single VM with Docker Compose

| Requirement | What it implies |
|---|---|
| FastAPI + Streamlit + nginx + 1–3 API replicas | several containers → Docker Compose |
| Uploaded images, SQLite database, model versions must survive restarts | persistent disk (bind mounts) |
| Retraining must really run in the cloud | ≥ 2 vCPU and ~3 GB RAM per API container for TensorFlow CPU training |
| Locust comparison of 1/2/3 containers | ability to scale the API service and load-balance it |

Free PaaS tiers (Render free, Hugging Face Spaces free, Streamlit Cloud) run a single container with ~0.5–2 GB
RAM and no persistent disk, so retraining and persistence would not work there. A small VM is the simplest
architecture that supports everything. Recommended size: **4 vCPU, 8–16 GB RAM, 30 GB disk**, Ubuntu 24.04
(e.g. AWS EC2 `t3.xlarge`, Azure `D4s_v5`, GCP `e2-standard-4`, or an equivalent student-credit VM).
No GPU is needed: inference takes tens of milliseconds on CPU and retraining uses a small dataset.

## Architecture

```
Internet ──► :8501  Streamlit UI (ui container) ──► http://gateway ──┐
         └─► :8000  nginx gateway  ──round-robin──► api × N (FastAPI + TensorFlow CPU)
                                                      │
              bind-mounted host folders: models/  db/  data/uploads/  data/bundle/  artifacts/
```

## Steps

```bash
# 1. On your computer: prepare deployable artifacts (models + small data bundle)
python scripts/export_artifacts.py                      # writes data/bundle (~150 MB)

# 2. On the VM: install Docker
curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER   # log out/in afterwards

# 3. Copy the project (code from GitHub, artifacts with rsync/scp)
git clone <your-repo-url> ovascope-MLOP && cd ovascope-MLOP
rsync -av models/ data/bundle/ artifacts/ <vm>:~/ovascope-MLOP/...       # from your computer

# 4. Configure secrets (never commit .env)
cp .env.example .env && nano .env                        # set a long random OVASCOPE_ADMIN_KEY

# 5. Start and check
docker compose up -d --build --scale api=2
curl http://localhost:8000/health
docker compose ps && docker compose logs -f api

# 6. Open ports 8000 and 8501 in the VM firewall / security group.
#    For HTTPS, put Caddy or a cloud load balancer in front (optional for the demo).
```

## Verification checklist (do this before claiming the deployment works)

- [ ] `GET /health` returns `model_loaded: true` from the public address
- [ ] a prediction through the Streamlit UI returns a class
- [ ] upload of labelled images succeeds and appears in *Upload training data → Database records*
- [ ] *Retrain Model* finishes with status `completed` and a new version appears
- [ ] `docker compose restart` → uploads, jobs and the active model are still there
- [ ] `python scripts/evaluate_deployed.py --url http://<vm>:8000` produces `production_evaluation.json`
- [ ] Locust run against the public URL (see `load_testing/`)

## Operational limitations

- SQLite on one host: fine for a demo; several hosts would need PostgreSQL and object storage.
- Retraining runs inside the API container that received the request and competes with inference for CPU.
- `OVASCOPE_ADMIN_KEY` is a single shared secret, not per-user authentication.
