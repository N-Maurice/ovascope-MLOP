"""Optional MongoDB copy of the records (src/mongo_sync.py), tested against an in-memory mongomock client."""
import mongomock
import pytest

from src import database, monitoring, users
from src.mongo_sync import MongoMirror, setting


@pytest.fixture(autouse=True)
def mongomock_accepts_new_bulk_arguments(monkeypatch):
    """pymongo >= 4.11 passes `sort` to bulk updates; mongomock 4.3 does not know it yet."""
    builder = mongomock.collection.BulkOperationBuilder
    for name in ("add_update", "add_replace"):
        original = getattr(builder, name)
        monkeypatch.setattr(builder, name, lambda self, *a, _f=original, sort=None, **k: _f(self, *a, **k))


@pytest.fixture
def mirror(project, tmp_path):
    cfg = {**project["cfg"], "paths": {**project["cfg"]["paths"], "database": tmp_path / "m.db"}}
    database.init_db(cfg["paths"]["database"])
    return MongoMirror(cfg, client=mongomock.MongoClient())


def add_upload(db_path, jpeg_bytes, seed, label=3):
    data = jpeg_bytes(label, seed=seed)
    checked = {"filename": f"egg{seed}.jpg", "class_id": label, "class_label": "Fasciolopsis buski",
               "file_hash": f"hash{seed}", "width": 64, "height": 64}
    return database.save_upload(db_path, checked, data, uploaded_by="alice")["id"], data


def test_copy_off_without_uri(project):
    mirror = MongoMirror(project["cfg"])            # conftest sets MONGODB_URI to ""
    assert not mirror.enabled and mirror.status()["enabled"] is False
    with pytest.raises(RuntimeError):
        mirror.sync()


def test_sync_copies_every_collection(mirror, jpeg_bytes):
    db_path = mirror.db_path
    upload_id, data = add_upload(db_path, jpeg_bytes, 1)
    job_id = database.create_job_if_idle(db_path, "manual", requested_by="alice")
    database.update_job(db_path, job_id, status="completed", candidate_metrics={"macro_f1": 0.93})
    monitoring.record_request(db_path, "/predict", 200, 12.5, "req1", "v001", 3, 0.9)
    users.create_user(db_path, "alice", "alice@example.org", "secret-pass-1")

    written = mirror.sync()
    assert written["uploads"] == 1 and written["retrain_jobs"] == 1 and written["request_log"] == 1
    assert written["users"] == 1 and written["models"] >= 1
    mongo = mirror.db()
    doc = mongo["uploads"].find_one({"_id": upload_id})
    assert doc["uploaded_by"] == "alice" and doc["training_status"] == "pending" and bytes(doc["image"]) == data
    assert mongo["retrain_jobs"].find_one({"_id": job_id})["candidate_metrics"]["macro_f1"] == 0.93
    user = mongo["users"].find_one({"username": "alice"})
    assert "password_hash" not in user and "email" not in user           # secrets and emails stay local
    assert mongo["models"].count_documents({"active": True}) == 1
    assert mirror.status()["last_sync"] and mirror.status()["last_error"] is None


def test_sync_is_incremental_and_follows_changes(mirror, jpeg_bytes):
    db_path = mirror.db_path
    upload_id, _ = add_upload(db_path, jpeg_bytes, 2)
    monitoring.record_request(db_path, "/predict", 200, 10.0, "a", "v001", 1, 0.8)
    mirror.sync()

    database.mark_uploads_used(db_path, [upload_id], "job42")
    monitoring.record_request(db_path, "/predict", 200, 11.0, "b", "v001", 2, 0.7)
    written = mirror.sync()
    mongo = mirror.db()
    assert written["request_log"] == 1                                    # only the new request is sent
    assert mongo["request_log"].count_documents({}) == 2
    assert mongo["uploads"].find_one({"_id": upload_id})["used_in_job"] == "job42"
    assert mongo["uploads"].count_documents({}) == 1                      # upserts never duplicate


def test_deleted_accounts_are_removed(mirror):
    db_path = mirror.db_path
    users.create_user(db_path, "admin1", "a1@example.org", "secret-pass-1")
    second = users.create_user(db_path, "user2", "u2@example.org", "secret-pass-1")
    mirror.sync()
    users.delete_user(db_path, second["id"])
    mirror.sync()
    assert [u["username"] for u in mirror.db()["users"].find()] == ["admin1"]


def test_connection_error_is_recorded(mirror):
    class Broken:
        def __getitem__(self, name):
            raise ConnectionError("cluster unreachable")
    mirror._client = Broken()
    with pytest.raises(ConnectionError):
        mirror.sync()
    assert "cluster unreachable" in mirror.status()["last_error"]


def test_setting_reads_env_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text('MONGODB_URI="mongodb+srv://u:p@c.example.net/"\nOTHER=x  # note\n')
    monkeypatch.setenv("OVASCOPE_ROOT", str(tmp_path))
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    assert setting("MONGODB_URI") == "mongodb+srv://u:p@c.example.net/"
    assert setting("OTHER") == "x"
    monkeypatch.setenv("MONGODB_URI", "")                                 # an empty variable turns it off
    assert setting("MONGODB_URI") == ""


def test_api_endpoints(project, monkeypatch):
    from fastapi.testclient import TestClient
    import api.main as main
    with TestClient(main.app) as c:
        assert c.get("/mongo/status").json()["enabled"] is False
        assert c.post("/mongo/sync").status_code == 409
        monkeypatch.setattr(main, "MIRROR", MongoMirror(main.CFG, client=mongomock.MongoClient()))
        body = c.post("/mongo/sync").json()
        assert body["documents"]["models"] >= 1 and body["last_error"] is None
