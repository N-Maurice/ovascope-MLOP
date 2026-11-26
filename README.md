# OvaScope MLOps

End-to-end machine-learning system that classifies microscope images of **parasite eggs** into
**11 classes**, with training, evaluation, a prediction API, a web interface, database-backed uploads,
genuine retraining from the saved model, monitoring, Docker packaging and load testing.

> **Scope.** The input is an image **cropped around one egg**. OvaScope does not detect or count eggs in a full
> microscope photo, and it is a student research prototype — **not a clinically validated diagnostic tool**.
> Confidence values are model scores, not guarantees.

## Contents
1. [Overview and objectives](#1-overview-and-objectives) · 2. [Dataset](#2-dataset) · 3. [Architecture](#3-system-architecture) ·
4. [Repository structure](#4-repository-structure) · 5. [Installation](#5-installation) · 6. [Dataset setup](#6-dataset-setup) ·
7. [Notebook](#7-notebook-execution) · 8. [Training](#8-training) · 9. [Results](#9-evaluation-results) ·
10. [Prediction](#10-prediction) · 11. [Retraining](#11-retraining) · 12. [API](#12-api) · 13. [Streamlit](#13-streamlit-application) ·
14. [Docker](#14-docker) · 15. [Cloud](#15-cloud-deployment) · 16. [Locust](#16-load-testing-locust) ·
17. [Tests](#17-tests) · 18. [Video](#18-video-demonstration) · 19. [Limitations](#19-limitations-and-future-work) · 20. [References](#20-references)

## 1. Overview and objectives

Intestinal parasites are diagnosed by finding their eggs in stool samples under a microscope. This is
slow, skilled work. OvaScope explores how well a compact CNN can recognise the egg species once an egg
has been located.

**Machine-learning objectives:** a verified, leakage-free dataset; a reproducible baseline CNN; a
transfer-learning model chosen on validation data; honest evaluation on a protected test set.
**MLOps objectives:** one shared preprocessing path for training and serving; versioned models;
database-backed uploads; retraining that continues from the saved model with a promotion check; monitoring;
containerisation; load testing.

## 2. Dataset

| | |
|---|---|
| Source | [Chula-ParasiteEgg-11, Zenodo record 15866298](https://zenodo.org/records/15866298) (ICIP 2022 challenge) |
| Content | 11,000 training + 2,200 test photos of faecal smears from several microscopes, cameras and smartphones |
| Annotations | COCO JSON, bounding box + class for every egg (11,031 train, 2,228 test); test labels in `test_labels_200.json` |
| Classes | 0 Ascaris lumbricoides · 1 Capillaria philippinensis · 2 Enterobius vermicularis · 3 Fasciolopsis buski · 4 Hookworm egg · 5 Hymenolepis diminuta · 6 Hymenolepis nana · 7 Opisthorchis viverrini* · 8 Paragonimus spp. · 9 Taenia spp. egg · 10 Trichuris trichiura |

\*The official label is spelled *Opisthorchis viverrine*; it is kept unchanged in `class_map.json` (`original_name`)
and only the display name is normalised.

**Preprocessing for classification.** Every box is padded by 10%, made square and cropped, which gives
**13,259 single-egg crops**:
- **train:** 9,379 crops (85% of the official training photos).
- **validation:** 1,652 crops (15%).
- **test:** 2,228 crops, from the official test set.

The train/validation split is stratified by class and **grouped by source photo**. The audit found no
exact or near-duplicate photos across splits. The crop dataset rebuilds **byte-for-byte identically** from
the official archives. Details: [reports/dataset_audit.md](reports/dataset_audit.md) and [data/README.md](data/README.md).

**Limitations of the data.** Photo quality varies strongly (focus, colour, illumination). 119 boxes run past
the photo edge. The test photos are darker and sharper than the training photos on average. Size cues
vary with magnification.

## 3. System architecture

```
 Zenodo archives ─► src/preprocessing.py ─► crops + manifest ─► src/train.py ─► models/versions/vNNN_*/
                                                                    │                (keras, h5, metadata)
                                                                    ▼                        │ activate
                                              notebook/ovascope_ml_pipeline.ipynb      models/active_model.json
                                                                                              │
  Browser ─► React app (web/) ─/api─► nginx ─► FastAPI × N (api/) ─► src/prediction.py ◄──────┘
            (or Streamlit, app/)              │   ├─ egg gate (src/gate.py) refuses non-egg images
                                                   │   ├─ /upload ─► src/database.py (SQLite: image BLOBs)
                                                   │   └─ /retrain ─► src/retrain.py (separate process):
                                                   │        load active model → new + replay images → fit →
                                                   │        evaluate vs active on validation → new version → promote?
                                                   └─ src/monitoring.py (request log, latency, uptime)
```

## 4. Repository structure

| Path | Purpose |
|---|---|
| `configs/config.yaml` | all settings (paths, image size, epochs, learning rates, retraining thresholds) |
| `src/config.py` | loads the config; environment-variable path overrides for Docker/Colab/Kaggle |
| `src/preprocessing.py` | crop generation, verification, image statistics, resize/augment/`tf.data` pipeline |
| `src/model.py` | baseline CNN and MobileNetV2 transfer model |
| `src/train.py` | training recipes, callbacks, versioned saving (`.keras` + load-tested `.h5`) |
| `src/evaluation.py` | metrics, confusion matrix, curves, example grids, latency |
| `src/prediction.py` | `Predictor`: load a version and classify one image |
| `src/database.py` | SQLite schema, upload validation and storage, retraining job records |
| `src/retrain.py` | retraining job: continue from the active model, evaluate, version, promote |
| `src/monitoring.py` | request log, latency statistics, uptime |
| `src/registry.py` | model versions, active pointer, rollback |
| `api/main.py` | FastAPI service |
| `src/gate.py` | egg gate: refuses images that are not parasite eggs |
| `src/results.py` | read-only access to saved evaluation results for the web interface |
| `src/users.py` | accounts: password hashing, sessions, roles; CLI `python -m src.users` |
| `scripts/setup.sh`, `scripts/dev.sh`, `scripts/install_system_tools.sh`, `Makefile` | setup and run helpers |
| `web/` | React + TypeScript web interface (main UI, see `web/README.md`) |
| `app/streamlit_app.py` | lightweight Streamlit interface (6 pages) |
| `notebook/ovascope_ml_pipeline.ipynb` | full documented ML lifecycle with real outputs |
| `scripts/` | `prepare_data.py`, `audit_dataset.py`, `evaluate_final.py`, `evaluate_deployed.py`, `export_artifacts.py` |
| `tests/` | pytest suite (preprocessing, prediction, API, retraining) on a synthetic mini-dataset |
| `load_testing/` | Locust file, experiment runner, analysis, results |
| `models/` | model versions (`versions/`), active model copies, protected test hashes |
| `artifacts/` | metrics, plots, logs, exports |
| `reports/` | dataset audit, model evaluation, load test report, rubric compliance |
| `docs/` | deployment guide, demo checklist |

## 5. Installation

**What you need**

| Tool | Version | Used for | Install |
|---|---|---|---|
| Python | 3.12 | training, API, tests | Miniconda, or `apt install python3.12` |
| Node.js + npm | 20 or newer | web interface (`web/`) | `sudo bash scripts/install_system_tools.sh` |
| Docker + Compose | any recent | containers (optional) | `sudo bash scripts/install_system_tools.sh` |
| make | any | shortcuts (optional) | `sudo bash scripts/install_system_tools.sh` |

`scripts/install_system_tools.sh` installs Docker, Compose, make and (if missing) Node.js from Ubuntu's own
packages and adds you to the `docker` group (log out and in afterwards).

**Quick setup** (after creating the Python environment below):
```bash
bash scripts/setup.sh      # checks tools, installs Python + web packages, creates .env and the database
bash scripts/dev.sh        # API on :8000 + web app on http://localhost:5173   (or: make dev)
```
Open the web app and choose **Sign in → Create account**: the first account becomes the administrator.

Python 3.12 is required (TensorFlow 2.21 has no build for Python 3.13+).

```bash
conda create -n ovascope python=3.12 -y && conda activate ovascope
pip install -r requirements.txt                       # CPU
# NVIDIA GPU: pip install "tensorflow[and-cuda]==2.21.0" instead of tensorflow
```

With pip-installed CUDA libraries, TensorFlow may print *"Cannot dlopen some GPU libraries"* (libcusolver).
Add the NVIDIA library folders to the library path:
```bash
export LD_LIBRARY_PATH=$(ls -d $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')
```

## 6. Dataset setup

Download `Chula-ParasiteEgg-11.zip`, `Chula-ParasiteEgg-11_test.zip` and `test_labels_200.json` from
[Zenodo](https://zenodo.org/records/15866298) into `data/raw/`, then:

```bash
python scripts/prepare_data.py          # verify MD5s → unzip → build crops → verify (≈3 min with 16 cores)
python scripts/audit_dataset.py         # audit: duplicates, leakage, statistics → artifacts/metrics/
```

## 7. Notebook execution

`notebook/ovascope_ml_pipeline.ipynb` covers the whole lifecycle (24 sections). It reuses the `src/` code and
detects its environment automatically (settings cell in Section 3).

| | How |
|---|---|
| **Local** | `conda activate ovascope && jupyter lab notebook/ovascope_ml_pipeline.ipynb`. Data from `data/`, outputs to `models/` and `artifacts/` |
| **Google Colab** | Upload and unzip the project to `/content/ovascope-MLOP` (or set `PROJECT_REPO_URL`). Upload `chula_parasite_egg11_crops.zip` (or set `USE_GOOGLE_DRIVE = True` and keep it on Drive). Choose a GPU runtime, then *Run all*. The results ZIP is offered via `files.download()` |
| **Kaggle** | Attach two input datasets: the project folder and `chula_parasite_egg11_crops.zip`. Enable GPU and internet (internet is needed for ImageNet weights; otherwise attach the MobileNetV2 weights file and pass its path). *Save Version* → download `ovascope_outputs/artifacts/exports/ovascope_training_results.zip` from the **Output** tab |

`QUICK_RUN = True` trains for 1–2 epochs to check the setup (≈5 min on a GPU).
`REBUILD_CROPS = True` rebuilds the crops from the official archives when they are available and proves the
rebuild is identical.

## 8. Training

```bash
python -m src.train --model both --select    # baseline CNN + MobileNetV2, activate the better (validation macro-F1)
python scripts/evaluate_final.py             # validation comparison + one-time test evaluation of the active model
python scripts/export_artifacts.py           # results ZIP + deployment data bundle
```

Optimisation techniques:
- **Adam** optimiser.
- **Dropout** 0.3.
- **Augmentation:** flips, rotation, shift, zoom, ±10% brightness and contrast.
- **Early stopping** with best-weight restoration.
- **ReduceLROnPlateau** and best-checkpoint saving.
- **Transfer learning** from ImageNet.
- **Fine-tuning** of the upper MobileNetV2 layers at lr 1e-5, with BatchNorm frozen.

Seeds are fixed (42); some GPU kernels remain non-deterministic.

## 9. Evaluation results

RESULTS_PLACEHOLDER

## 10. Prediction

```python
from src.config import load_config
from src.prediction import Predictor
predictor = Predictor.from_active(load_config())
predictor.predict_file("data/processed/crops/test/10_trichuris_trichiura/<file>.jpg")
# {'predicted_class_id': 10, 'predicted_class': 'Trichuris trichiura', 'confidence': ..., 'top_k': [...], 'model_version': ...}
```
or `curl -F file=@egg.jpg http://localhost:8000/predict`.

## 11. Retraining

1. **Upload:** labelled images go to the *Upload* page or `POST /upload`. Labels come from one label for all
   files, a per-file list, or a ZIP with one folder per class.
2. **Validation:** each file is checked for extension, size (≤ 10 MB) and that it decodes. Its label must be
   in the class map, and exact duplicates are rejected (SHA-256). Copies of protected test images are also
   rejected.
3. **Storage:** accepted images are stored **as BLOBs in SQLite** (`db/ovascope.db`, table `uploads`), with a
   browsable copy in `data/uploads/`.
4. **Retrain:** *Retrain Model* / `POST /retrain` creates a job record, and a second job cannot start while
   one is running. A separate process then:
   - loads the **active saved model**;
   - trains it for 3 epochs at lr 1e-5 on the new images (×3) plus 100 replay crops per class;
   - evaluates the candidate and the active model on the validation set;
   - saves the candidate as a new version (`vNNN_retrain`);
   - **activates it only if** macro-F1(candidate) ≥ macro-F1(active) − 0.005.
   The previous versions are kept, so you can roll back (`POST /model/rollback`).
5. **Automatic trigger:** when `retraining.trigger_threshold` (default 50) validated new images are waiting,
   retraining is flagged *recommended*. With `auto_start: true` it starts by itself. The threshold is a
   configurable operating choice, not an optimised value.

## 12. API

`uvicorn api.main:app --port 8000` → interactive docs at `http://localhost:8000/docs`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | service status, uptime since process start, model loaded/version, serving container |
| `/predict` | POST | classify one uploaded image (multipart field `file`) |
| `/upload` | POST 🔒 | upload labelled images (`files`, plus `label` or `labels`, or a class-folder ZIP) |
| `/uploads`, `/uploads/{id}/image` | GET 🔒 | stored records and counts; one stored image |
| `/retrain` | POST 🔒 | start a retraining job (HTTP 202 + job id; 409 if not possible) |
| `/retrain/status` | GET | readiness, pending images, trigger flag, latest job |
| `/retrain/status/{job_id}` | GET | job status, progress, metrics, promotion decision |
| `/model/info` | GET | active model metadata |
| `/model/versions` | GET | all versions and their metrics |
| `/model/activate/{version}`, `/model/rollback` | POST 🔒 | switch the active model |
| `/metrics` | GET | request counts, failures, latency, recent activity, retraining events |
| `/classes`, `/classes/{id}/examples/{n}` | GET | class list; validation example crops (test crops are never served) |
| `/results/evaluation` | GET | final test + validation metrics, confusion matrices, model comparison, accuracy by camera |
| `/results/history/{version}` | GET | per-epoch training history of a version |
| `/dataset/summary` | GET | dataset audit: splits, class balance, leakage checks, crop sizes |
| `/gate` | GET | egg-gate threshold and results |
| `/auth/register`, `/auth/login` | POST | create an account; sign in with username **or** email + password → session token |
| `/auth/me`, `/auth/logout`, `/auth/password` | GET, POST, POST | current account; end the session; change password (signed in) |
| `/auth/status`, `/auth/verify` | GET, POST 🔒 | sign-up/first-admin state; check admin credentials |
| `/users`, `/users/{id}` | GET, PATCH, DELETE 🔒 | list accounts; change role (admin/user) or active flag; delete |
| `/mongo/status`, `/mongo/sync` | GET, POST 🔒 | MongoDB Atlas copy: state, last sync, document counts; copy now |

🔒 requires an **administrator**: either `Authorization: Bearer <token>` from `/auth/login` of an account with the
admin role, or the operator header `X-API-Key: <OVASCOPE_ADMIN_KEY>` (scripts, Streamlit, load tests).

**Accounts.** Stored in the SQLite database (`users`, `sessions`); passwords as salted scrypt hashes, sessions as
SHA-256 hashes of random tokens (7 days, or 1 day without "keep me signed in"). The **first account becomes admin**;
later sign-ups are `user` until an admin promotes them (admin console → Users). The last active admin cannot be
demoted, deactivated or deleted. Five failed sign-ins for the same name block that name for five minutes.
Settings: `auth:` in `configs/config.yaml` (`allow_signup`, `session_days`, `min_password_length`).
Command line: `python -m src.users create-admin --username NAME --email MAIL`, `reset-password --login NAME`, `list`.
Uploads and retraining jobs record which account made them (`uploaded_by`, `requested_by`).
Only on a fresh install with **no** accounts and **no** `OVASCOPE_ADMIN_KEY` are admin endpoints open (development).
The model is loaded once and swapped atomically when the active version changes. Retraining never runs inside a request.

**MongoDB Atlas copy (optional).** SQLite stays the main store; when `MONGODB_URI` is set in `.env`, the API also
copies the records to MongoDB (database `ovascope`): `uploads` (with the image bytes), `retrain_jobs`, `request_log`,
`models` and `users` (username, role and dates only: password hashes and email addresses stay local). The copy runs
every 60 s and right after uploads, retraining starts and account changes; it is an idempotent upsert, so the API
keeps working while the cluster is unreachable and catches up afterwards. Settings: `mongodb:` in
`configs/config.yaml`. Atlas only accepts connections from IP addresses listed under **Security → Network Access**.
Admin console → Records shows the state and has a **Sync now** button; from the command line:
`python -m src.mongo_sync status` (connection check + counts) and `python -m src.mongo_sync push` (copy now).
Browse the documents in Atlas (**Browse Collections**) or with the MongoDB extension for VS Code.

## 13. Web interface (React) and Streamlit application

**React app (`web/`, main interface).** Works on laptop and phone screens, light and dark mode.
```bash
OVASCOPE_ADMIN_KEY=choose-a-key uvicorn api.main:app --port 8000     # terminal 1
cd web && npm install && npm run dev                                 # terminal 2 -> http://localhost:5173
```
Pages: Home, Classify (crop one egg, top-3, egg check, reference and look-alike crops), Classes, Dataset,
Performance (metrics, confusion matrix, per-class table, training curves, egg check), Status (live monitoring),
About, and an admin console behind sign-in (upload, database records, retraining with live progress,
model activation and rollback). Details: `web/README.md`.

**Streamlit (lightweight alternative):**
`OVASCOPE_API_URL=http://localhost:8000 streamlit run app/streamlit_app.py`

The app has six pages:
1. **Dashboard:** status, model, uptime, requests, retraining.
2. **Predict:** upload, preview, class, confidence, top-3 and version.
3. **Dataset insights:** class distribution, plus brightness, size and texture/colour, each with an interpretation.
4. **Upload training data.**
5. **Train & retrain:** Retrain button, job status, version comparison, activation and rollback.
6. **Monitoring:** health, latency, failures, recent requests, evaluation results.

## 14. Docker

```bash
python scripts/export_artifacts.py               # builds data/bundle (validation set + replay sample)
cp .env.example .env                             # set OVASCOPE_ADMIN_KEY
docker compose up -d --build                     # web app :8080, gateway :8000, Streamlit :8501, 1 API
docker compose up -d --scale api=3               # 3 API containers behind nginx
curl localhost:8000/health                       # "instance" = container that answered
docker compose logs -f api                       # logs
docker compose down                              # stop (models/, db/, uploads persist on the host)
```
The images contain code and dependencies only. Models, database, uploads and the data bundle are mounted,
so nothing large is baked in and everything survives restarts.

DOCKER_STATUS_PLACEHOLDER

## 15. Cloud deployment

DEPLOY_STATUS_PLACEHOLDER

## 16. Load testing (Locust)

LOCUST_PLACEHOLDER

## 17. Tests

```bash
pytest -q
```
The tests cover:
- preprocessing (shapes, ranges, channels, invalid images, train/serve consistency);
- prediction (loading, valid indices and probabilities, class names, `.h5` equality);
- the API (all endpoints, upload validation, ZIP folders, auth);
- retraining (loads the saved model, creates a new version, keeps the old one, failure safety,
  duplicate-job prevention, trigger).

They run on a synthetic mini-dataset in a temporary folder. TESTS_PLACEHOLDER

## 18. Video demonstration

Pending — link not yet created. Recording plan: [docs/demo_checklist.md](docs/demo_checklist.md).

## 19. Limitations and future work

LIMITATIONS_PLACEHOLDER

## 20. References

- Palasuwan, D., Naruenatthanaset, K., Kobchaisawat, T., Chalidabhongse, T. H., Nunthanasup, N., Boonpeng, K.,
  Anantrasirichai, N. *Parasitic Egg Detection and Classification in Microscopic Images* (Chula-ParasiteEgg-11).
  Zenodo record 15866298, DOI 10.21227/vyh8-4h71, MIT licence. https://zenodo.org/records/15866298 ;
  challenge page: https://icip2022challenge.piclab.ai/dataset/
- Sandler, M. et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018.
- TensorFlow/Keras, FastAPI, Streamlit, Locust documentation.

GitHub repository: pending (no public URL yet).
