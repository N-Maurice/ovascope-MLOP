"""Optional copy of the OvaScope records in MongoDB (e.g. a MongoDB Atlas cluster).

SQLite (db/ovascope.db) stays the main store: it needs no network, and the API keeps working when
the cluster is unreachable. When MONGODB_URI is set, a background thread in the API copies the
records to MongoDB every `mongodb.sync_interval_seconds`, and right after uploads, retraining starts
and account changes. Each sync is an idempotent upsert, so it can be repeated or run by hand.

Collections (database `mongodb.database`, default "ovascope"):
    uploads        one document per uploaded image (_id = SQLite id); the image bytes too if store_images
    retrain_jobs   one document per job (_id = job id), metrics stored as nested documents
    request_log    one document per API request (_id = SQLite id), only new rows are sent
    models         one document per model version (_id = version name), with an `active` flag
    users          username, role, active, dates. Password hashes and email addresses are NOT copied.

Connection settings come from the environment, or from the project's .env file when the variable is
not set at all (setting it to an empty value turns the copy off):
    MONGODB_URI                 mongodb+srv://user:password@cluster.xxxxx.mongodb.net/
    MONGODB_USERNAME/PASSWORD   only needed when the URI has no user:password part

Command line:
    python -m src.mongo_sync status     # connection check + document counts in MongoDB
    python -m src.mongo_sync push       # copy everything now
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from datetime import datetime, timezone

from src import database, registry
from src.config import find_project_root, load_config

log = logging.getLogger("ovascope.mongo")

DEFAULTS = {"enabled": True, "database": "ovascope", "sync_interval_seconds": 60, "store_images": True}
USER_FIELDS = "id, username, role, active, created_at, last_login"   # never password_hash or email
UPLOAD_FIELDS = ("id, filename, class_id, class_label, file_hash, width, height, upload_timestamp, validation_status,"
                 " training_status, used_in_job, uploaded_by, length(image_data) AS bytes")


def setting(name: str) -> str:
    """Value of an environment variable, else of the same key in the project's .env (not exported)."""
    if name in os.environ:
        return os.environ[name].strip()
    env_file = find_project_root() / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == name:
                value = value.strip()
                if value[:1] in "\"'" and value[:1] in value[1:]:          # quoted value
                    return value[1:value.index(value[0], 1)]
                return value.split(" #")[0].strip()                         # unquoted, drop a trailing comment
    return ""


def describe_error(error: Exception) -> str:
    """Short error text, with a hint for the usual Atlas set-up problems."""
    detail = str(error).split(" (configured timeouts")[0].split(", Timeout:")[0]   # first server's error is enough
    text = f"{error.__class__.__name__}: {detail[:300]}"
    if "SSL handshake failed" in text or "TLSV1_ALERT" in text:
        text += (" - Atlas refused the connection. Add this computer's (or server's) IP address in Atlas under"
                 " Security > Network Access.")
    elif "bad auth" in text or "Authentication failed" in text:
        text += " - check the user name and password in MONGODB_URI (Atlas > Database Access)."
    return text


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MongoMirror:
    """Copies SQLite records to MongoDB. Thread-safe; failures are recorded, never raised to API callers."""

    def __init__(self, cfg: dict, client=None):
        self.options = {**DEFAULTS, **(cfg.get("mongodb") or {})}
        self.db_path = cfg["paths"]["database"]
        self.models_dir = cfg["paths"]["models_dir"]
        self.uri = setting("MONGODB_URI")
        self._client = client                 # tests pass a mongomock client
        self._lock = threading.Lock()         # one sync at a time
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self.last_sync: str | None = None
        self.last_error: str | None = None
        self.last_result: dict = {}

    # ------------------------------------------------------------------ connection
    @property
    def enabled(self) -> bool:
        return bool(self.options["enabled"]) and (bool(self.uri) or self._client is not None)

    def client(self):
        if self._client is None:
            from pymongo import MongoClient
            extra = {}
            if "@" not in self.uri and setting("MONGODB_USERNAME"):
                extra = {"username": setting("MONGODB_USERNAME"), "password": setting("MONGODB_PASSWORD")}
            self._client = MongoClient(self.uri, serverSelectionTimeoutMS=8000, connectTimeoutMS=8000,
                                       appname="ovascope", tz_aware=True, **extra)
        return self._client

    def db(self):
        return self.client()[self.options["database"]]

    # ------------------------------------------------------------------ sync
    def sync(self) -> dict:
        """Copy every collection now. Returns {collection: documents written}; raises on connection errors."""
        if not self.enabled:
            raise RuntimeError("MongoDB copy is off: set MONGODB_URI (and mongodb.enabled: true in the config)")
        with self._lock:
            try:
                db = self.db()
                result = {"uploads": self._sync_uploads(db), "retrain_jobs": self._sync_jobs(db),
                          "request_log": self._sync_requests(db), "models": self._sync_models(db),
                          "users": self._sync_users(db)}
            except Exception as error:
                self.last_error = describe_error(error)
                raise
            self.last_sync, self.last_error, self.last_result = registry.utc_now(), None, result
            return result

    def _upsert(self, collection, docs: list[dict]) -> int:
        from pymongo import ReplaceOne
        if not docs:
            return 0
        collection.bulk_write([ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in docs], ordered=False)
        return len(docs)

    def _sync_uploads(self, db) -> int:
        from bson.binary import Binary
        from pymongo import UpdateOne
        coll = db["uploads"]
        with database.connect(self.db_path) as conn:
            rows = [dict(r) for r in conn.execute(f"SELECT {UPLOAD_FIELDS} FROM uploads ORDER BY id")]
        stamp = _now()
        ops = [UpdateOne({"_id": r["id"]}, {"$set": {**{k: v for k, v in r.items() if k != "id"}, "synced_at": stamp}},
                         upsert=True) for r in rows]
        if ops:
            coll.bulk_write(ops, ordered=False)
        if self.options["store_images"]:                       # image bytes are sent once per upload
            have = {d["_id"] for d in coll.find({"has_image": True}, {"_id": 1})}
            missing = [r["id"] for r in rows if r["id"] not in have]
            for start in range(0, len(missing), 50):
                chunk = missing[start:start + 50]
                with database.connect(self.db_path) as conn:
                    images = conn.execute(f"SELECT id, image_data FROM uploads WHERE id IN ({','.join('?' * len(chunk))})",
                                          chunk).fetchall()
                coll.bulk_write([UpdateOne({"_id": i["id"]}, {"$set": {"image": Binary(bytes(i["image_data"])),
                                                                       "has_image": True}}) for i in images],
                                ordered=False)
        return len(rows)

    def _sync_jobs(self, db) -> int:
        jobs = database.list_jobs(self.db_path, limit=1_000_000)
        return self._upsert(db["retrain_jobs"], [{"_id": j.pop("id"), **j, "synced_at": _now()} for j in jobs])

    def _sync_requests(self, db) -> int:
        coll = db["request_log"]
        last = coll.find_one({}, {"_id": 1}, sort=[("_id", -1)])
        after = last["_id"] if last else 0
        sent = 0
        while True:                                           # only rows newer than the last copied one
            with database.connect(self.db_path) as conn:
                rows = [dict(r) for r in conn.execute("SELECT * FROM request_log WHERE id > ? ORDER BY id LIMIT 5000",
                                                      (after,))]
            if not rows:
                return sent
            after = rows[-1]["id"]
            sent += self._upsert(coll, [{"_id": r.pop("id"), **r} for r in rows])

    def _sync_models(self, db) -> int:
        active = registry.get_active_version(self.models_dir)
        docs = []
        for meta in registry.list_versions(self.models_dir):
            meta = json.loads(json.dumps(meta, default=str))   # plain JSON types only
            name = meta.get("version")
            if name:
                docs.append({**meta, "_id": name, "active": name == active, "synced_at": _now()})
        return self._upsert(db["models"], docs)

    def _sync_users(self, db) -> int:
        with database.connect(self.db_path) as conn:
            rows = [dict(r) for r in conn.execute(f"SELECT {USER_FIELDS} FROM users")]
        coll = db["users"]
        docs = [{**r, "_id": r["id"], "active": bool(r["active"]), "synced_at": _now()} for r in rows]
        for doc in docs:
            del doc["id"]
        written = self._upsert(coll, docs)
        coll.delete_many({"_id": {"$nin": [d["_id"] for d in docs]}})   # accounts deleted here are removed there
        return written

    # ------------------------------------------------------------------ background thread
    def start(self) -> None:
        if not self.enabled or self._thread:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="mongo-sync", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None

    def kick(self) -> None:
        """Ask for a sync soon (after an upload, a retraining start or an account change)."""
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.sync()
            except Exception:
                log.warning("MongoDB sync failed: %s", self.last_error)
            self._wake.wait(self.options["sync_interval_seconds"])
            self._wake.clear()

    # ------------------------------------------------------------------ status
    def status(self, remote_counts: bool = False) -> dict:
        info = {"enabled": self.enabled, "configured": bool(self.uri), "database": self.options["database"],
                "store_images": bool(self.options["store_images"]),
                "sync_interval_seconds": self.options["sync_interval_seconds"],
                "last_sync": self.last_sync, "last_error": self.last_error, "last_result": self.last_result,
                "cluster": self.uri.split("@")[-1].split("/")[0].split("?")[0] if self.uri else None}
        if remote_counts and self.enabled:
            try:
                db = self.db()
                info["documents"] = {name: db[name].estimated_document_count()
                                     for name in ("uploads", "retrain_jobs", "request_log", "models", "users")}
            except Exception as error:
                info["last_error"] = describe_error(error)
        return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy the OvaScope records to MongoDB.")
    parser.add_argument("command", choices=["status", "push"])
    args = parser.parse_args()
    mirror = MongoMirror(load_config())
    if not mirror.enabled:
        raise SystemExit("MongoDB copy is off: set MONGODB_URI in .env (and mongodb.enabled: true in the config).")
    if args.command == "push":
        print("written:", mirror.sync())
    status = mirror.status(remote_counts=True)
    print(f"cluster: {status['cluster']}   database: {status['database']}")
    if status["last_error"]:
        raise SystemExit(f"error: {status['last_error']}")
    print("documents in MongoDB:", status.get("documents"))


if __name__ == "__main__":
    main()
