"""SQLite storage for uploaded training images, retraining jobs and request logs.

SQLite was chosen because it is a single file, needs no server, and is easy to
inspect for a demonstration (e.g. `sqlite3 db/ovascope.db "select * from uploads"`).

Tables
    uploads         labelled images uploaded for retraining (image bytes stored as BLOB)
    retrain_jobs    one row per retraining run, with status, metrics and decision
    request_log     one row per API prediction request (operational monitoring)
    users           accounts for the web interface (password stored as a scrypt hash, see src/users.py)
    sessions        sign-in sessions (only the SHA-256 of each token is stored)
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from src.preprocessing import load_image_from_bytes
from src.registry import utc_now

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    filename           TEXT NOT NULL,
    image_data         BLOB NOT NULL,
    class_id           INTEGER NOT NULL,
    class_label        TEXT NOT NULL,
    file_hash          TEXT NOT NULL UNIQUE,     -- SHA-256 of the bytes: blocks exact duplicates
    width              INTEGER,
    height             INTEGER,
    upload_timestamp   TEXT NOT NULL,
    validation_status  TEXT NOT NULL,            -- 'validated'
    training_status    TEXT NOT NULL DEFAULT 'pending',   -- 'pending' | 'used'
    used_in_job        TEXT
);
CREATE TABLE IF NOT EXISTS retrain_jobs (
    id                   TEXT PRIMARY KEY,
    status               TEXT NOT NULL,          -- queued | running | completed | failed
    progress             TEXT,
    created_at           TEXT NOT NULL,
    started_at           TEXT,
    finished_at          TEXT,
    trigger              TEXT,                   -- manual | automatic
    n_new_images         INTEGER,
    base_version         TEXT,
    candidate_version    TEXT,
    train_metrics        TEXT,                   -- JSON
    candidate_metrics    TEXT,                   -- JSON (promotion set)
    active_metrics       TEXT,                   -- JSON (promotion set)
    promotion_decision   TEXT,
    promoted             INTEGER DEFAULT 0,
    error                TEXT
);
CREATE TABLE IF NOT EXISTS request_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id         TEXT,
    timestamp          TEXT NOT NULL,
    endpoint           TEXT NOT NULL,
    status_code        INTEGER NOT NULL,
    latency_ms         REAL NOT NULL,
    model_version      TEXT,
    predicted_class_id INTEGER,
    confidence         REAL
);
CREATE TABLE IF NOT EXISTS users (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    username       TEXT NOT NULL UNIQUE COLLATE NOCASE,
    email          TEXT NOT NULL UNIQUE,
    password_hash  TEXT NOT NULL,
    role           TEXT NOT NULL DEFAULT 'user',      -- admin | user
    active         INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL,
    last_login     TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash  TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);
"""
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_request_log_endpoint ON request_log(endpoint, id);
CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(training_status);
"""

# Columns added after the first release. init_db adds them to existing databases (no data is lost).
MIGRATIONS = {
    "uploads": {"uploaded_by": "TEXT"},
    "retrain_jobs": {"requested_by": "TEXT", "worker": "TEXT"},   # worker = "<host>:<pid>" running the job
}


@contextmanager
def connect(db_path: Path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")   # readers do not block the writer
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db(db_path: Path) -> None:
    with connect(db_path) as db:
        db.executescript(SCHEMA)
        for table, columns in MIGRATIONS.items():
            existing = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
            for column, kind in columns.items():
                if column not in existing:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
        db.executescript(INDEXES)


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------

def load_protected_hashes(path: Path) -> set[str]:
    """SHA-256 hashes of every held-out test crop and test photo (see scripts/export_artifacts.py)."""
    path = Path(path)
    return set(path.read_text().split()) if path.exists() else set()


def validate_upload(filename: str, data: bytes, label, class_map: list[dict], allowed_extensions,
                    max_bytes: int, protected_hashes: set[str]) -> dict:
    """Check one uploaded file. Returns {"ok": True, ...} or {"ok": False, "reason": ...}.

    Checks: safe file name, allowed extension, size limit, decodable image,
    label belongs to the verified class map, not a copy of a protected test image.
    """
    safe_name = Path(str(filename).replace("\\", "/")).name   # drop any directory components
    if not safe_name or safe_name.startswith("."):
        return {"ok": False, "filename": str(filename), "reason": "invalid file name"}
    if Path(safe_name).suffix.lower() not in allowed_extensions:
        return {"ok": False, "filename": safe_name, "reason": f"unsupported file type (allowed: {', '.join(allowed_extensions)})"}
    if len(data) > max_bytes:
        return {"ok": False, "filename": safe_name, "reason": f"file larger than {max_bytes // 1_000_000} MB"}
    class_entry = resolve_label(label, class_map)
    if class_entry is None:
        return {"ok": False, "filename": safe_name, "reason": f"unknown class label '{label}'"}
    try:
        image = load_image_from_bytes(data)
    except ValueError as error:
        return {"ok": False, "filename": safe_name, "reason": str(error)}
    if image.width < 16 or image.height < 16:
        return {"ok": False, "filename": safe_name, "reason": "image smaller than 16x16 pixels"}
    file_hash = hashlib.sha256(data).hexdigest()
    if file_hash in protected_hashes:
        return {"ok": False, "filename": safe_name, "reason": "identical to a protected test-set image"}
    return {"ok": True, "filename": safe_name, "class_id": class_entry["class_id"],
            "class_label": class_entry["display_name"], "file_hash": file_hash,
            "width": image.width, "height": image.height}


def resolve_label(label, class_map: list[dict]):
    """Accept a class id (int or digit string), display name, official name, or folder name."""
    if label is None:
        return None
    text = str(label).strip()
    for entry in class_map:
        if text.isdigit() and int(text) == entry["class_id"]:
            return entry
        if text.lower() in (entry["display_name"].lower(), entry["original_name"].lower(), entry["folder"].lower()):
            return entry
    return None


def save_upload(db_path: Path, checked: dict, data: bytes, uploads_dir: Path | None = None, class_folder: str = "",
                uploaded_by: str | None = None) -> dict:
    """Insert a validated upload. Duplicate bytes (same SHA-256) are rejected by the UNIQUE constraint."""
    try:
        with connect(db_path) as db:
            cursor = db.execute(
                "INSERT INTO uploads (filename, image_data, class_id, class_label, file_hash, width, height,"
                " upload_timestamp, validation_status, uploaded_by) VALUES (?,?,?,?,?,?,?,?, 'validated', ?)",
                (checked["filename"], sqlite3.Binary(data), checked["class_id"], checked["class_label"],
                 checked["file_hash"], checked["width"], checked["height"], utc_now(), uploaded_by))
            upload_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        return {"ok": False, "filename": checked["filename"], "reason": "duplicate of an earlier upload"}
    if uploads_dir:   # optional human-browsable copy; the database remains the source of truth
        folder = Path(uploads_dir) / (class_folder or str(checked["class_id"]))
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{upload_id:06d}_{checked['file_hash'][:8]}{Path(checked['filename']).suffix.lower()}").write_bytes(data)
    return {"ok": True, "id": upload_id, "filename": checked["filename"], "class_id": checked["class_id"],
            "class_label": checked["class_label"]}


def list_uploads(db_path: Path, limit: int = 200) -> list[dict]:
    with connect(db_path) as db:
        rows = db.execute("SELECT id, filename, class_id, class_label, file_hash, width, height, upload_timestamp,"
                          " validation_status, training_status, used_in_job, uploaded_by, length(image_data) AS bytes"
                          " FROM uploads ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_upload_image(db_path: Path, upload_id: int) -> bytes | None:
    with connect(db_path) as db:
        row = db.execute("SELECT image_data FROM uploads WHERE id=?", (upload_id,)).fetchone()
    return bytes(row["image_data"]) if row else None


def upload_counts(db_path: Path) -> dict:
    with connect(db_path) as db:
        rows = db.execute("SELECT training_status, COUNT(*) AS n FROM uploads WHERE validation_status='validated'"
                          " GROUP BY training_status").fetchall()
        per_class = db.execute("SELECT class_label, COUNT(*) AS n FROM uploads WHERE training_status='pending'"
                               " GROUP BY class_label ORDER BY class_label").fetchall()
    counts = {r["training_status"]: r["n"] for r in rows}
    return {"pending": counts.get("pending", 0), "used": counts.get("used", 0),
            "total": sum(counts.values()), "pending_per_class": {r["class_label"]: r["n"] for r in per_class}}


def pending_uploads(db_path: Path) -> list[dict]:
    with connect(db_path) as db:
        rows = db.execute("SELECT id, filename, image_data, class_id FROM uploads"
                          " WHERE validation_status='validated' AND training_status='pending' ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def mark_uploads_used(db_path: Path, upload_ids: list[int], job_id: str) -> None:
    with connect(db_path) as db:
        db.executemany("UPDATE uploads SET training_status='used', used_in_job=? WHERE id=?",
                       [(job_id, i) for i in upload_ids])


# ---------------------------------------------------------------------------
# Retraining jobs
# ---------------------------------------------------------------------------

JOB_JSON_FIELDS = ("train_metrics", "candidate_metrics", "active_metrics")


def create_job_if_idle(db_path: Path, trigger: str = "manual", requested_by: str | None = None) -> str | None:
    """Atomically create a queued job unless one is already queued/running. Returns the job id or None."""
    job_id = uuid.uuid4().hex[:12]
    connection = sqlite3.connect(db_path, timeout=30)
    try:
        connection.execute("BEGIN IMMEDIATE")   # write lock: two clicks cannot both pass the check
        busy = connection.execute("SELECT id FROM retrain_jobs WHERE status IN ('queued','running')").fetchone()
        if busy:
            connection.rollback()
            return None
        connection.execute("INSERT INTO retrain_jobs (id, status, progress, created_at, trigger, requested_by, worker)"
                           " VALUES (?,?,?,?,?,?,?)",
                           (job_id, "queued", "waiting to start", utc_now(), trigger, requested_by,
                            f"{socket.gethostname()}:{os.getpid()}"))
        connection.commit()
        return job_id
    finally:
        connection.close()


def update_job(db_path: Path, job_id: str, **fields) -> None:
    for key in JOB_JSON_FIELDS:
        if key in fields and not isinstance(fields[key], (str, type(None))):
            fields[key] = json.dumps(fields[key])
    assignments = ", ".join(f"{k}=?" for k in fields)
    with connect(db_path) as db:
        db.execute(f"UPDATE retrain_jobs SET {assignments} WHERE id=?", (*fields.values(), job_id))


def get_job(db_path: Path, job_id: str) -> dict | None:
    with connect(db_path) as db:
        row = db.execute("SELECT * FROM retrain_jobs WHERE id=?", (job_id,)).fetchone()
    return _job_dict(row) if row else None


def list_jobs(db_path: Path, limit: int = 20) -> list[dict]:
    with connect(db_path) as db:
        rows = db.execute("SELECT * FROM retrain_jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_job_dict(r) for r in rows]


def _job_dict(row) -> dict:
    job = dict(row)
    for key in JOB_JSON_FIELDS:
        if job.get(key):
            job[key] = json.loads(job[key])
    job["promoted"] = bool(job.get("promoted"))
    return job


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def fail_stale_jobs(db_path: Path) -> int:
    """At API start-up, mark queued/running jobs whose worker process no longer exists as failed.

    Each job records the host and process id of its worker. Jobs started by another container (another
    host name) are left alone, so starting or scaling API containers never cancels a running job.
    Jobs without a recorded worker (created by older versions) are treated as stale."""
    host = socket.gethostname()
    with connect(db_path) as db:
        rows = db.execute("SELECT id, worker FROM retrain_jobs WHERE status IN ('queued','running')").fetchall()
        stale = []
        for row in rows:
            worker_host, _, pid = (row["worker"] or "").rpartition(":")
            if not row["worker"] or (worker_host == host and not (pid.isdigit() and _process_alive(int(pid)))):
                stale.append(row["id"])
        for job_id in stale:
            db.execute("UPDATE retrain_jobs SET status='failed', error='interrupted (service restarted)', finished_at=?"
                       " WHERE id=?", (utc_now(), job_id))
    return len(stale)
