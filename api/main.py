"""OvaScope REST API (FastAPI).

Run locally:   uvicorn api.main:app --host 0.0.0.0 --port 8000
Docs:          http://localhost:8000/docs

Public endpoints:     GET /health, POST /predict, GET /model/info, GET /metrics, GET /classes
                      (/predict answers 422 "not_a_parasite_egg" when the egg gate rejects the image)
Protected endpoints:  POST /upload, POST /retrain, POST /model/activate/{version}, POST /model/rollback
                      (send header  X-API-Key: <OVASCOPE_ADMIN_KEY>)
"""
from __future__ import annotations

import io
import os
import socket
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from src import database, monitoring, registry, retrain
from src.config import load_config
from src.gate import GATE_FOLDER, GATE_METADATA_FILE, NotParasiteEggError
from src.preprocessing import load_class_map, load_image_from_bytes

CFG = load_config()
DB = CFG["paths"]["database"]
MODELS_DIR = CFG["paths"]["models_dir"]
MAX_BYTES = CFG["api"]["max_upload_mb"] * 1_000_000
ALLOWED = set(CFG["api"]["allowed_extensions"])
ADMIN_KEY = os.environ.get("OVASCOPE_ADMIN_KEY", "")
INSTANCE = socket.gethostname()   # identifies the container that served a request (load balancing checks)


class ModelHolder:
    """Keeps one Predictor in memory and swaps it when the active version (or the egg gate) changes.

    Loading happens outside the lock; only the reference swap is locked, so
    predictions keep being served by the old model while a new one loads.
    """

    def __init__(self):
        self.predictor = None
        self.error = None
        self._lock = threading.Lock()
        self._pointer_mtime = None

    def refresh(self) -> None:
        pointer = MODELS_DIR / "active_model.json"
        gate_meta = MODELS_DIR / GATE_FOLDER / GATE_METADATA_FILE
        mtime = (pointer.stat().st_mtime if pointer.exists() else None,
                 gate_meta.stat().st_mtime if gate_meta.exists() else None)
        if mtime == self._pointer_mtime and (self.predictor or self.error):
            return
        try:
            from src.prediction import Predictor
            predictor = Predictor.from_active(CFG)
            with self._lock:
                self.predictor, self.error, self._pointer_mtime = predictor, None, mtime
        except Exception as error:   # keep serving the previous model if the new one fails to load
            self.error = f"{error.__class__.__name__}: {error}"
            self._pointer_mtime = mtime

    def get(self):
        self.refresh()
        with self._lock:
            return self.predictor


MODEL = ModelHolder()


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db(DB)
    database.fail_stale_jobs(DB)
    MODEL.refresh()
    yield


app = FastAPI(title="OvaScope API", version="1.0",
              description="Parasite-egg image classification (11 classes, Chula-ParasiteEgg-11). "
                          "Input: a crop centred on one egg. Research prototype - not a diagnostic device.",
              lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Time every prediction request and store it for monitoring."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    latency = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Served-By"] = INSTANCE
    if request.url.path == "/predict":
        info = getattr(request.state, "prediction", {}) or {}
        try:
            monitoring.record_request(DB, "/predict", response.status_code, latency, request_id,
                                      info.get("model_version"), info.get("predicted_class_id"),
                                      info.get("confidence"))
        except Exception:
            pass   # monitoring must never break a prediction
    return response


def require_admin(x_api_key: str | None = Header(default=None)):
    """Basic access control for data-changing endpoints. If OVASCOPE_ADMIN_KEY is unset (local
    development only) the check is skipped; always set it for a public deployment."""
    if ADMIN_KEY and x_api_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


def class_map():
    return load_class_map(CFG["paths"]["processed_dir"] if (CFG["paths"]["processed_dir"] / "class_map.json").exists()
                          else MODELS_DIR / "class_mapping.json")


# ---------------------------------------------------------------------------
# Health, model info, metrics
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    predictor = MODEL.get()
    gate = predictor.gate if predictor else None
    return {"status": "ok", "service_uptime_seconds": round(monitoring.uptime_seconds(), 1),
            "model_loaded": predictor is not None, "model_version": predictor.version if predictor else None,
            "model_error": MODEL.error, "instance": INSTANCE,
            "input_check": gate.info() if gate else {"enabled": False}}


@app.get("/classes")
def classes():
    return class_map()


@app.get("/model/info")
def model_info():
    predictor = MODEL.get()
    if predictor is None:
        raise HTTPException(status_code=503, detail=f"No model loaded: {MODEL.error}")
    return predictor.metadata


@app.get("/model/versions")
def model_versions():
    active = registry.get_active_version(MODELS_DIR)
    return [{**{k: v.get(k) for k in ("version", "kind", "architecture", "parent_version", "created_at",
                                      "validation_metrics", "test_metrics", "model_size_mb")},
             "active": v["version"] == active} for v in registry.list_versions(MODELS_DIR)]


@app.post("/model/activate/{version}", dependencies=[Depends(require_admin)])
def activate(version: str):
    try:
        pointer = registry.activate_version(MODELS_DIR, version)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    MODEL.refresh()
    return pointer


@app.post("/model/rollback", dependencies=[Depends(require_admin)])
def rollback():
    try:
        pointer = registry.rollback(MODELS_DIR)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    MODEL.refresh()
    return pointer


@app.get("/metrics")
def metrics():
    data = monitoring.get_metrics(DB)
    predictor = MODEL.get()
    data["model_version"] = predictor.version if predictor else None
    data["instance"] = INSTANCE
    return data


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@app.post("/predict")
def predict(request: Request, file: UploadFile = File(...)):
    predictor = MODEL.get()
    if predictor is None:
        raise HTTPException(status_code=503, detail="No trained model is available yet")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File larger than {CFG['api']['max_upload_mb']} MB")
    try:
        result = predictor.predict_bytes(data)
    except NotParasiteEggError as rejected:
        # 422: the file is a valid image, but not of a parasite egg - ask for a proper image
        raise HTTPException(status_code=422, detail={"error": "not_a_parasite_egg", "message": str(rejected),
                                                     **rejected.check})
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    request.state.prediction = result
    return result


# ---------------------------------------------------------------------------
# Upload of labelled training images
# ---------------------------------------------------------------------------

def _store(filename: str, data: bytes, label, classes, protected) -> dict:
    checked = database.validate_upload(filename, data, label, classes, ALLOWED, MAX_BYTES, protected)
    if not checked["ok"]:
        return checked
    predictor = MODEL.get()
    if predictor is not None and predictor.gate is not None:   # keep non-egg images out of retraining data
        verdict = predictor.gate.check(load_image_from_bytes(data))
        if not verdict["is_parasite_egg"]:
            return {"ok": False, "filename": checked["filename"],
                    "reason": f"does not look like a parasite egg image (egg score {verdict['egg_score']:.3f} "
                              f"< threshold {verdict['threshold']:.3f})"}
    folder = next(c["folder"] for c in classes if c["class_id"] == checked["class_id"])
    return database.save_upload(DB, checked, data, CFG["paths"]["uploads_dir"], folder)


def _zip_members(data: bytes, classes):
    """Yield (filename, bytes, label) from a ZIP laid out as <class folder>/<image>.

    Every folder name must be a known class folder, display name or id;
    otherwise the whole archive is rejected before anything is stored."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Not a valid ZIP archive")
    members = [m for m in archive.infolist() if not m.is_dir() and not PurePosixPath(m.filename).name.startswith(".")
               and "__MACOSX" not in m.filename]
    bad = sorted({str(PurePosixPath(m.filename).parent) for m in members
                  if len(PurePosixPath(m.filename).parts) < 2
                  or database.resolve_label(PurePosixPath(m.filename).parts[-2], classes) is None})
    if bad:
        raise HTTPException(status_code=400, detail=f"Unrecognised class folder(s) in ZIP: {bad}")
    for m in members:
        if m.file_size > MAX_BYTES:
            yield PurePosixPath(m.filename).name, b"\0" * (MAX_BYTES + 1), PurePosixPath(m.filename).parts[-2]
        else:
            yield PurePosixPath(m.filename).name, archive.read(m), PurePosixPath(m.filename).parts[-2]


@app.post("/upload", dependencies=[Depends(require_admin)])
def upload(files: list[UploadFile] = File(...), label: str | None = Form(default=None),
           labels: str | None = Form(default=None)):
    """Upload labelled images for retraining.

    Labelling options:
      * `label`  - one class (id, name or folder) applied to every file, or
      * `labels` - comma-separated list, one label per file in the same order, or
      * a ZIP file whose folders are class names (e.g. 00_ascaris_lumbricoides/img1.jpg).
    """
    classes = class_map()
    protected = database.load_protected_hashes(MODELS_DIR / "protected_test_hashes.txt")
    per_file = [x.strip() for x in labels.split(",")] if labels else None
    if per_file and len(per_file) != len(files):
        raise HTTPException(status_code=400, detail=f"{len(files)} files but {len(per_file)} labels")
    accepted, rejected = [], []
    for index, item in enumerate(files):
        data = item.file.read(MAX_BYTES * 50 + 1)
        if (item.filename or "").lower().endswith(".zip"):
            entries = _zip_members(data, classes)
        else:
            entries = [(item.filename or "", data, per_file[index] if per_file else label)]
        for filename, content, file_label in entries:
            result = _store(filename, content, file_label, classes, protected)
            (accepted if result["ok"] else rejected).append(result)
    status = retrain.retraining_status(CFG)
    if CFG["retraining"].get("auto_start") and status["retraining_recommended"] and not status["job_running"]:
        try:
            status["auto_started_job"] = retrain.start_retraining(CFG, trigger="automatic")
        except retrain.RetrainError as error:
            status["auto_start_skipped"] = str(error)
    return {"accepted": accepted, "rejected": rejected, "saved_to_database": len(accepted),
            "retraining": status}


@app.get("/uploads")
def uploads(limit: int = 200):
    return {"counts": database.upload_counts(DB), "records": database.list_uploads(DB, limit)}


@app.get("/uploads/{upload_id}/image")
def upload_image(upload_id: int):
    data = database.get_upload_image(DB, upload_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Upload not found")
    return Response(content=data, media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------

@app.post("/retrain", dependencies=[Depends(require_admin)])
def start_retrain():
    try:
        job_id = retrain.start_retraining(CFG, trigger="manual")
    except retrain.RetrainError as error:
        raise HTTPException(status_code=409, detail=str(error))
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "queued",
                                                  "status_url": f"/retrain/status/{job_id}"})


@app.get("/retrain/status")
def retrain_readiness():
    return retrain.retraining_status(CFG)


@app.get("/retrain/status/{job_id}")
def retrain_job_status(job_id: str):
    job = database.get_job(DB, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/retrain/jobs")
def retrain_jobs(limit: int = 20):
    return database.list_jobs(DB, limit)
