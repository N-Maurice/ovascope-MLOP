"""Accounts: registration, sign-in with username or email, roles, sessions and admin access."""
import pytest
from fastapi.testclient import TestClient

from src import database, users


@pytest.fixture
def client(project):
    import api.main as main
    db = project["cfg"]["paths"]["database"]
    with TestClient(main.app) as test_client:
        main.THROTTLE.failures.clear()
        yield test_client, main
    with database.connect(db) as conn:   # leave no accounts behind: other modules expect open dev mode
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM users")


def register(c, username, email, password="secret-pass-1"):
    return c.post("/auth/register", json={"username": username, "email": email, "password": password})


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_password_hashing():
    stored = users.hash_password("correct horse 1")
    assert stored.startswith("scrypt$") and "correct horse" not in stored
    assert users.verify_password("correct horse 1", stored)
    assert not users.verify_password("wrong", stored)
    assert users.hash_password("same") != users.hash_password("same")   # random salt


def test_first_account_is_admin_then_users(client):
    c, _ = client
    assert c.get("/auth/status").json()["first_account_becomes_admin"] is True
    first = register(c, "alice", "Alice@Example.org")
    assert first.status_code == 201 and first.json()["user"]["role"] == "admin"
    assert first.json()["user"]["email"] == "alice@example.org"
    second = register(c, "bob", "bob@example.org")
    assert second.json()["user"]["role"] == "user"
    status = c.get("/auth/status").json()
    assert status["first_account_becomes_admin"] is False and status["open_mode"] is False


def test_register_validation_and_duplicates(client):
    c, _ = client
    assert register(c, "al", "a@example.org").status_code == 400                 # username too short
    assert register(c, "alice", "not-an-email").status_code == 400
    assert register(c, "alice", "a@example.org", "short1").status_code == 400    # too short
    assert register(c, "alice", "a@example.org", "onlyletters").status_code == 400
    assert register(c, "alice", "a@example.org").status_code == 201
    assert register(c, "ALICE", "other@example.org").status_code == 409         # case-insensitive username
    assert register(c, "alice2", "A@EXAMPLE.ORG").status_code == 409             # same email


def test_login_with_username_or_email(client):
    c, _ = client
    register(c, "alice", "alice@example.org")
    by_name = c.post("/auth/login", json={"login": "Alice", "password": "secret-pass-1"})
    by_mail = c.post("/auth/login", json={"login": "alice@example.org", "password": "secret-pass-1"})
    assert by_name.status_code == 200 and by_mail.status_code == 200
    assert c.get("/auth/me", headers=bearer(by_mail.json()["token"])).json()["username"] == "alice"
    wrong = c.post("/auth/login", json={"login": "alice", "password": "nope"})
    unknown = c.post("/auth/login", json={"login": "nobody", "password": "nope"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"]   # no hint whether the account exists


def test_login_throttle(client):
    c, _ = client
    register(c, "alice", "alice@example.org")
    for _ in range(5):
        c.post("/auth/login", json={"login": "alice", "password": "bad"})
    assert c.post("/auth/login", json={"login": "alice", "password": "secret-pass-1"}).status_code == 429


def test_admin_endpoints_by_role(client):
    c, _ = client
    admin = register(c, "alice", "alice@example.org").json()["token"]
    user = register(c, "bob", "bob@example.org").json()["token"]
    assert c.get("/uploads").status_code == 401                         # accounts exist -> no open mode
    assert c.get("/uploads", headers=bearer(user)).status_code == 403   # signed in, but not admin
    assert c.get("/uploads", headers=bearer(admin)).status_code == 200
    assert c.post("/auth/verify", headers=bearer(admin)).json()["username"] == "alice"
    assert c.post("/retrain", headers=bearer(user)).status_code == 403


def test_admin_key_still_works(client, monkeypatch):
    c, main = client
    monkeypatch.setattr(main, "ADMIN_KEY", "secret")
    assert c.get("/uploads", headers={"X-API-Key": "secret"}).status_code == 200
    assert c.get("/uploads", headers={"X-API-Key": "wrong"}).status_code == 401


def test_logout_and_password_change(client):
    c, _ = client
    token = register(c, "alice", "alice@example.org").json()["token"]
    other = c.post("/auth/login", json={"login": "alice", "password": "secret-pass-1"}).json()["token"]
    assert c.post("/auth/password", headers=bearer(token),
                  json={"current_password": "wrong", "new_password": "new-pass-22"}).status_code == 401
    assert c.post("/auth/password", headers=bearer(token),
                  json={"current_password": "secret-pass-1", "new_password": "new-pass-22"}).status_code == 204
    assert c.get("/auth/me", headers=bearer(other)).status_code == 401    # all sessions ended
    fresh = c.post("/auth/login", json={"login": "alice", "password": "new-pass-22"}).json()["token"]
    assert c.post("/auth/logout", headers=bearer(fresh)).status_code == 204
    assert c.get("/auth/me", headers=bearer(fresh)).status_code == 401


def test_user_management_and_last_admin_guard(client):
    c, _ = client
    admin_body = register(c, "alice", "alice@example.org").json()
    admin, alice_id = admin_body["token"], admin_body["user"]["id"]
    bob = register(c, "bob", "bob@example.org").json()
    listed = c.get("/users", headers=bearer(admin)).json()
    assert [u["username"] for u in listed] == ["alice", "bob"] and "password_hash" not in listed[0]
    # cannot remove the only admin
    assert c.patch(f"/users/{alice_id}", headers=bearer(admin), json={"role": "user"}).status_code == 409
    assert c.delete(f"/users/{alice_id}", headers=bearer(admin)).status_code == 409
    # promote bob, who can then use admin endpoints
    assert c.patch(f"/users/{bob['user']['id']}", headers=bearer(admin), json={"role": "admin"}).json()["role"] == "admin"
    assert c.get("/uploads", headers=bearer(bob["token"])).status_code == 200
    # deactivating ends bob's sessions and blocks sign-in
    c.patch(f"/users/{bob['user']['id']}", headers=bearer(admin), json={"active": False})
    assert c.get("/auth/me", headers=bearer(bob["token"])).status_code == 401
    assert c.post("/auth/login", json={"login": "bob", "password": "secret-pass-1"}).status_code == 403
    assert c.delete(f"/users/{bob['user']['id']}", headers=bearer(admin)).status_code == 204


def test_uploads_and_jobs_record_who_did_it(client, jpeg_bytes):
    c, _ = client
    admin = register(c, "alice", "alice@example.org").json()["token"]
    response = c.post("/upload", headers=bearer(admin), data={"label": "3"},
                      files=[("files", ("x.jpg", jpeg_bytes(3, seed=4242)))]).json()
    assert response["saved_to_database"] == 1
    record = c.get("/uploads", headers=bearer(admin)).json()["records"][0]
    assert record["uploaded_by"] == "alice"


def test_migration_adds_columns_to_old_database(tmp_path):
    import sqlite3
    old = tmp_path / "old.db"
    with sqlite3.connect(old) as conn:   # schema of the first release, without the new columns/tables
        conn.execute("CREATE TABLE uploads (id INTEGER PRIMARY KEY, filename TEXT, training_status TEXT)")
        conn.execute("CREATE TABLE retrain_jobs (id TEXT PRIMARY KEY, status TEXT)")
        conn.execute("INSERT INTO uploads (filename) VALUES ('kept.jpg')")
    database.init_db(old)
    with database.connect(old) as conn:
        assert "uploaded_by" in {r["name"] for r in conn.execute("PRAGMA table_info(uploads)")}
        assert {"requested_by", "worker", "heartbeat_at"} <= {r["name"] for r in conn.execute("PRAGMA table_info(retrain_jobs)")}
        assert conn.execute("SELECT filename FROM uploads").fetchone()[0] == "kept.jpg"
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0


def test_stale_job_detection_keeps_other_hosts_jobs(tmp_path):
    import socket
    db = tmp_path / "jobs.db"
    database.init_db(db)
    with database.connect(db) as conn:
        for job_id, worker in [("dead", f"{socket.gethostname()}:999999"), ("other", "another-container:1"),
                               ("legacy", None)]:
            conn.execute("INSERT INTO retrain_jobs (id, status, created_at, worker, heartbeat_at)"
                         " VALUES (?, 'running', '2999-01-01', ?, '2999-01-01')", (job_id, worker))
    assert database.fail_stale_jobs(db) == 2
    assert database.get_job(db, "other")["status"] == "running"
    assert database.get_job(db, "dead")["status"] == "failed"


def test_dead_worker_detected_while_api_runs(tmp_path):
    """A worker that was killed (left as a zombie child) is reported as failed on the next status check."""
    import socket
    import subprocess
    import sys
    db = tmp_path / "jobs.db"
    database.init_db(db)
    worker = subprocess.Popen([sys.executable, "-c", "pass"])
    while worker.poll() is None:
        pass
    # poll() reaped it; start another child and let it exit without being waited for -> zombie
    zombie = subprocess.Popen([sys.executable, "-c", "pass"])
    import time
    time.sleep(1.0)
    with database.connect(db) as conn:
        conn.execute("INSERT INTO retrain_jobs (id, status, created_at, worker, heartbeat_at)"
                     " VALUES ('z', 'running', '2999-01-01', ?, '2999-01-01')", (f"{socket.gethostname()}:{zombie.pid}",))
        conn.execute("INSERT INTO retrain_jobs (id, status, created_at, worker, heartbeat_at)"
                     " VALUES ('alive', 'running', '2999-01-01', ?, '2999-01-01')",
                     (f"{socket.gethostname()}:{__import__('os').getpid()}",))
    assert database.fail_stale_jobs(db, reason="worker died") == 1
    assert database.get_job(db, "z")["error"] == "worker died"
    assert database.get_job(db, "alive")["status"] == "running"


def test_silent_job_on_vanished_host_is_failed(tmp_path):
    """A job from a replaced container (unknown host) is failed once its heartbeat stops."""
    from datetime import datetime, timedelta, timezone
    recent = datetime.now(timezone.utc).isoformat(timespec="seconds")
    old = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat(timespec="seconds")
    db = tmp_path / "jobs.db"
    database.init_db(db)
    with database.connect(db) as conn:
        conn.execute("INSERT INTO retrain_jobs (id, status, created_at, worker, heartbeat_at) VALUES"
                     " ('silent', 'running', ?, 'replaced-container:7', ?),"
                     " ('beating', 'running', ?, 'other-container:7', ?)", (old, old, old, recent))
    assert database.fail_stale_jobs(db, stale_minutes=10) == 1
    assert database.get_job(db, "silent")["status"] == "failed"
    assert database.get_job(db, "beating")["status"] == "running"
