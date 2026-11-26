"""OvaScope REST API (FastAPI).

Run locally:   uvicorn api.main:app --host 0.0.0.0 --port 8000
Docs:          http://localhost:8000/docs

Public endpoints:     GET /health, POST /predict, GET /model/info, GET /model/versions, GET /metrics, GET /classes,
                      GET /classes/{id}/examples/{n}, GET /results/evaluation, GET /results/history/{version},
                      GET /dataset/summary, GET /gate, GET /retrain/status, GET /auth/status
                      (/predict answers 422 "not_a_parasite_egg" when the egg gate rejects the image)
Accounts:             POST /auth/register, POST /auth/login (username or email + password), POST /auth/logout,
                      GET /auth/me, POST /auth/password
Admin endpoints:      POST /upload, GET /uploads, GET /uploads/{id}/image, POST /retrain,
                      POST /model/activate/{version}, POST /model/rollback, POST /auth/verify,
                      GET /users, PATCH /users/{id}, DELETE /users/{id}, GET /mongo/status, POST /mongo/sync
                      (send  Authorization: Bearer <token from /auth/login>  of an admin account,
                       or  X-API-Key: <OVASCOPE_ADMIN_KEY>  for scripts)

The React interface in web/ calls these endpoints (see web/README.md).
"""
from __future__ import annotations

import io
import os
import secrets
import socket
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from src import database, monitoring, registry, results, retrain, users
from src.mongo_sync import MongoMirror
from src.config import load_config
from src.gate import GATE_FOLDER, GATE_METADATA_FILE, NotParasiteEggError
from src.preprocessing import load_class_map, load_image_from_bytes

CFG = load_config()
DB = CFG["paths"]["database"]
MODELS_DIR = CFG["paths"]["models_dir"]
MAX_BYTES = CFG["api"]["max_upload_mb"] * 1_000_000
ALLOWED = set(CFG["api"]["allowed_extensions"])
ADMIN_KEY = os.environ.get("OVASCOPE_ADMIN_KEY", "")
AUTH = {"allow_signup": True, "session_days": 7, "min_password_length": 8, **(CFG.get("auth") or {})}
THROTTLE = users.LoginThrottle()
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
MIRROR = MongoMirror(CFG)   # optional copy of the records in MongoDB Atlas (off unless MONGODB_URI is set)
# Successful writes to these paths trigger a MongoDB sync soon after (instead of waiting for the interval).
SYNC_AFTER = ("/upload", "/retrain", "/auth/register", "/users", "/model/activate", "/model/rollback")


def limit_tf_threads() -> None:
    """Match TensorFlow's thread pools to the container's CPU quota (see retrain.tf_thread_count).

    Without this, TensorFlow sizes its pools for every host core; under many concurrent predictions in a
    CPU-limited container the extra threads only compete for the quota, which lowers throughput."""
    threads = retrain.tf_thread_count()
    if not threads:
        return
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(2, threads))
    except RuntimeError:   # TensorFlow already initialised in this process (e.g. tests): keep its settings
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    limit_tf_threads()
    database.init_db(DB)
    database.fail_stale_jobs(DB, stale_minutes=CFG["retraining"].get("stale_job_minutes", 10))
    MODEL.refresh()
    MIRROR.start()
    yield
    MIRROR.stop()


app = FastAPI(title="OvaScope API", version="1.0",
              description="Parasite-egg image classification (11 classes, Chula-ParasiteEgg-11). "
                          "Input: a crop centred on one egg. Research prototype - not a diagnostic device.",
              lifespan=lifespan)

# Browsers block cross-origin calls unless the API allows them. The React dev server (port 5173) and
# the Docker web container use a same-origin proxy, so this list only matters for other front ends.
CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "OVASCOPE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["X-Served-By", "X-Request-ID"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Time every prediction request and store it for monitoring."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    latency = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Served-By"] = INSTANCE
    if request.method != "GET" and response.status_code < 400 and request.url.path.startswith(SYNC_AFTER):
        MIRROR.kick()
    if request.url.path == "/predict":
        info = getattr(request.state, "prediction", {}) or {}
        try:
            monitoring.record_request(DB, "/predict", response.status_code, latency, request_id,
                                      info.get("model_version"), info.get("predicted_class_id"),
                                      info.get("confidence"))
        except Exception:
            pass   # monitoring must never break a prediction
    return response


def _bearer(authorization: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return None


def open_mode() -> bool:
    """Development only: no admin key configured and no accounts created yet -> admin endpoints are open."""
    return not ADMIN_KEY and users.count_users(DB) == 0


def current_user(authorization: str | None = Header(default=None)) -> dict:
    user = users.user_from_token(DB, _bearer(authorization))
    if user is None:
        raise HTTPException(status_code=401, detail="Please sign in (session missing or expired)")
    return user


def require_admin(x_api_key: str | None = Header(default=None), authorization: str | None = Header(default=None)) -> dict:
    """Access control for data-changing endpoints: an admin account's session token, or the
    OVASCOPE_ADMIN_KEY header (for scripts, Streamlit and load tests)."""
    if ADMIN_KEY and x_api_key and secrets.compare_digest(x_api_key, ADMIN_KEY):
        return {"username": "admin-key", "role": "admin"}
    user = users.user_from_token(DB, _bearer(authorization))
    if user is not None:
        if user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Your account does not have administrator rights")
        return user
    if open_mode():
        return {"username": None, "role": "admin"}
    raise HTTPException(status_code=401, detail="Admin sign-in required (Authorization: Bearer token or X-API-Key)")


def class_map():
    return load_class_map(CFG["paths"]["processed_dir"] if (CFG["paths"]["processed_dir"] / "class_map.json").exists()
                          else MODELS_DIR / "class_mapping.json")


# ---------------------------------------------------------------------------
# Health, model info, metrics
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    """The API has no home page of its own: send visitors to the interactive documentation."""
    return RedirectResponse(url="/docs")


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


@app.get("/classes/{class_id}/examples/{index}")
def class_example(class_id: int, index: int = 0):
    """A validation crop of the class (test crops are never served). `index` cycles through them."""
    path = results.example_image(CFG["paths"]["processed_dir"], class_id, index)
    if path is None:
        raise HTTPException(status_code=404, detail="No example images available for this class")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


# ---------------------------------------------------------------------------
# Saved results (read-only; produced by the notebook / training scripts)
# ---------------------------------------------------------------------------

@app.get("/results/evaluation")
def results_evaluation():
    data = results.evaluation(CFG["paths"]["artifacts_dir"])
    if data is None:
        raise HTTPException(status_code=404, detail="No final evaluation found; run training and evaluation first")
    data["class_names"] = [c["display_name"] for c in class_map()]
    return data


@app.get("/results/history/{version}")
def results_history(version: str):
    if "/" in version or ".." in version:
        raise HTTPException(status_code=400, detail="Invalid version name")
    rows = results.training_history(MODELS_DIR, version)
    if rows is None:
        raise HTTPException(status_code=404, detail=f"No training history for '{version}'")
    return rows


@app.get("/dataset/summary")
def dataset_summary():
    data = results.dataset_summary(CFG["paths"]["artifacts_dir"], class_map())
    if data is None:
        raise HTTPException(status_code=404, detail="No dataset audit found; run the data preparation first")
    return data


@app.get("/gate")
def gate():
    return results.gate_report(MODELS_DIR, CFG["paths"]["artifacts_dir"])


# ---------------------------------------------------------------------------
# Admin sign-in check (the key itself is the credential; nothing is stored server-side)
# ---------------------------------------------------------------------------

class Credentials(BaseModel):
    login: str = Field(min_length=1, max_length=254, description="username or email")
    password: str = Field(min_length=1, max_length=256)
    remember: bool = True


class Registration(BaseModel):
    username: str = Field(max_length=64)
    email: str = Field(max_length=254)
    password: str = Field(max_length=256)


class PasswordChange(BaseModel):
    current_password: str = Field(max_length=256)
    new_password: str = Field(max_length=256)


class UserUpdate(BaseModel):
    role: str | None = None
    active: bool | None = None


def _account_error(error: users.AccountError):
    raise HTTPException(status_code=error.status, detail=str(error))


def _session_response(user: dict, remember: bool = True) -> dict:
    session = users.create_session(DB, user["id"], AUTH["session_days"] if remember else 1)
    return {"token": session["token"], "expires_at": session["expires_at"], "user": user}


@app.get("/auth/status")
def auth_status():
    accounts = users.count_users(DB)
    admins = users.count_users(DB, role="admin", active_only=True)
    return {"admin_key_required": bool(ADMIN_KEY), "accounts_exist": accounts > 0,
            "signup_open": bool(AUTH["allow_signup"]) or admins == 0,
            "first_account_becomes_admin": admins == 0, "open_mode": open_mode(),
            "min_password_length": AUTH["min_password_length"]}


@app.post("/auth/register", status_code=201)
def register(body: Registration):
    if not AUTH["allow_signup"] and users.count_users(DB, role="admin", active_only=True) > 0:
        raise HTTPException(status_code=403, detail="Sign-up is disabled; ask an administrator for an account")
    try:
        user = users.create_user(DB, body.username, body.email, body.password,
                                 min_password_length=AUTH["min_password_length"])
    except users.AccountError as error:
        _account_error(error)
    return _session_response(user)


@app.post("/auth/login")
def login(body: Credentials):
    try:
        THROTTLE.check(body.login)
        user = users.authenticate(DB, body.login, body.password)
    except users.AccountError as error:
        if error.status == 401:
            THROTTLE.failed(body.login)
        _account_error(error)
    THROTTLE.succeeded(body.login)
    return _session_response(user, body.remember)


@app.post("/auth/logout", status_code=204)
def logout(authorization: str | None = Header(default=None)):
    token = _bearer(authorization)
    if token:
        users.end_session(DB, token)
    return Response(status_code=204)


@app.get("/auth/me")
def me(user: dict = Depends(current_user)):
    return user


@app.post("/auth/password", status_code=204)
def password(body: PasswordChange, user: dict = Depends(current_user)):
    try:
        users.change_password(DB, user["id"], body.current_password, body.new_password, AUTH["min_password_length"])
    except users.AccountError as error:
        _account_error(error)
    return Response(status_code=204)


@app.post("/auth/verify")
def auth_verify(admin: dict = Depends(require_admin)):
    return {"ok": True, "admin_key_required": bool(ADMIN_KEY), "username": admin.get("username")}


@app.get("/users", dependencies=[Depends(require_admin)])
def list_accounts():
    return users.list_users(DB)


@app.patch("/users/{user_id}", dependencies=[Depends(require_admin)])
def update_account(user_id: int, body: UserUpdate):
    try:
        return users.update_user(DB, user_id, role=body.role, active=body.active)
    except users.AccountError as error:
        _account_error(error)


@app.delete("/users/{user_id}", status_code=204, dependencies=[Depends(require_admin)])
def delete_account(user_id: int):
    try:
        users.delete_user(DB, user_id)
    except users.AccountError as error:
        _account_error(error)
    return Response(status_code=204)


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

def _store(filename: str, data: bytes, label, classes, protected, uploaded_by: str | None = None) -> dict:
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
    return database.save_upload(DB, checked, data, CFG["paths"]["uploads_dir"], folder, uploaded_by=uploaded_by)


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


@app.post("/upload")
def upload(files: list[UploadFile] = File(...), label: str | None = Form(default=None),
           labels: str | None = Form(default=None), admin: dict = Depends(require_admin)):
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
            result = _store(filename, content, file_label, classes, protected, admin.get("username"))
            (accepted if result["ok"] else rejected).append(result)
    status = retrain.retraining_status(CFG)
    if CFG["retraining"].get("auto_start") and status["retraining_recommended"] and not status["job_running"]:
        try:
            status["auto_started_job"] = retrain.start_retraining(CFG, trigger="automatic")
        except retrain.RetrainError as error:
            status["auto_start_skipped"] = str(error)
    return {"accepted": accepted, "rejected": rejected, "saved_to_database": len(accepted),
            "retraining": status}


@app.get("/uploads", dependencies=[Depends(require_admin)])
def uploads(limit: int = 200):
    return {"counts": database.upload_counts(DB), "records": database.list_uploads(DB, limit)}


@app.get("/uploads/{upload_id}/image", dependencies=[Depends(require_admin)])
def upload_image(upload_id: int):
    data = database.get_upload_image(DB, upload_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Upload not found")
    kind = {b"\xff\xd8": "image/jpeg", b"\x89P": "image/png", b"BM": "image/bmp"}.get(data[:2], "application/octet-stream")
    return Response(content=data, media_type=kind)


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------

@app.post("/retrain")
def start_retrain(admin: dict = Depends(require_admin)):
    try:
        job_id = retrain.start_retraining(CFG, trigger="manual", requested_by=admin.get("username"))
    except retrain.RetrainError as error:
        raise HTTPException(status_code=409, detail=str(error))
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "queued",
                                                  "status_url": f"/retrain/status/{job_id}"})


STALE_MINUTES = CFG["retraining"].get("stale_job_minutes", 10)
WORKER_DIED = "the retraining worker stopped unexpectedly (e.g. out of memory); see artifacts/logs/retrain_<job>.log"


@app.get("/retrain/status")
def retrain_readiness():
    database.fail_stale_jobs(DB, reason=WORKER_DIED, stale_minutes=STALE_MINUTES)
    return retrain.retraining_status(CFG)


@app.get("/retrain/status/{job_id}")
def retrain_job_status(job_id: str):
    database.fail_stale_jobs(DB, reason=WORKER_DIED, stale_minutes=STALE_MINUTES)
    job = database.get_job(DB, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/retrain/jobs")
def retrain_jobs(limit: int = 20):
    return database.list_jobs(DB, limit)


# ---------------------------------------------------------------------------
# MongoDB Atlas copy (src/mongo_sync.py)
# ---------------------------------------------------------------------------

@app.get("/mongo/status", dependencies=[Depends(require_admin)])
def mongo_status(counts: bool = False):
    """Whether the MongoDB copy is on, when it last synced, and (counts=true) documents per collection."""
    return MIRROR.status(remote_counts=counts)


@app.post("/mongo/sync", dependencies=[Depends(require_admin)])
def mongo_sync():
    """Copy every record to MongoDB now."""
    if not MIRROR.enabled:
        raise HTTPException(status_code=409, detail="MongoDB copy is off: set MONGODB_URI in .env and restart the API")
    try:
        written = MIRROR.sync()
    except Exception:
        raise HTTPException(status_code=502, detail=f"MongoDB sync failed: {MIRROR.last_error}")
    return {"written": written, **MIRROR.status(remote_counts=True)}
