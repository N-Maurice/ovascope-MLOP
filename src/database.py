"""SQLite storage for uploaded training images, retraining jobs and request logs.

SQLite was chosen because it is a single file, needs no server, and is easy to
inspect for a demonstration (e.g. `sqlite3 db/ovascope.db "select * from uploads"`).

Tables
    uploads         labelled images uploaded for retraining (image bytes stored as BLOB)
    retrain_jobs    one row per retraining run, with status, metrics and decision
    request_log     one row per API prediction request (operational monitoring)
"""
from __future__ import annotations

import hashlib
import json
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
"""


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


def save_upload(db_path: Path, checked: dict, data: bytes, uploads_dir: Path | None = None, class_folder: str = "") -> dict:
    """Insert a validated upload. Duplicate bytes (same SHA-256) are rejected by the UNIQUE constraint."""
    try:
        with connect(db_path) as db:
            cursor = db.execute(
                "INSERT INTO uploads (filename, image_data, class_id, class_label, file_hash, width, height,"
                " upload_timestamp, validation_status) VALUES (?,?,?,?,?,?,?,?, 'validated')",
                (checked["filename"], sqlite3.Binary(data), checked["class_id"], checked["class_label"],
                 checked["file_hash"], checked["width"], checked["height"], utc_now()))
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
                          " validation_status, training_status, used_in_job, length(image_data) AS bytes"
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


def create_job_if_idle(db_path: Path, trigger: str = "manual") -> str | None:
    """Atomically create a queued job unless one is already queued/running. Returns the job id or None."""
    job_id = uuid.uuid4().hex[:12]
    connection = sqlite3.connect(db_path, timeout=30)
    try:
        connection.execute("BEGIN IMMEDIATE")   # write lock: two clicks cannot both pass the check
        busy = connection.execute("SELECT id FROM retrain_jobs WHERE status IN ('queued','running')").fetchone()
        if busy:
            connection.rollback()
            return None
        connection.execute("INSERT INTO retrain_jobs (id, status, progress, created_at, trigger) VALUES (?,?,?,?,?)",
                           (job_id, "queued", "waiting to start", utc_now(), trigger))
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


def fail_stale_jobs(db_path: Path) -> int:
    """At API start-up, a job still marked running/queued belongs to a dead process: mark it failed."""
    with connect(db_path) as db:
        cursor = db.execute("UPDATE retrain_jobs SET status='failed', error='interrupted (service restarted)',"
                            " finished_at=? WHERE status IN ('queued','running')", (utc_now(),))
    return cursor.rowcount
