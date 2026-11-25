"""Endpoints used by the React interface (web/): saved results, dataset summary, examples, sign-in."""
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(project):
    import api.main as main
    with TestClient(main.app) as test_client:
        yield test_client, main


def test_class_examples_are_validation_crops(client):
    c, _ = client
    response = c.get("/classes/3/examples/0")
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    assert c.get("/classes/3/examples/7").status_code == 200   # index cycles
    assert c.get("/classes/99/examples/0").status_code == 404


def test_evaluation_missing_then_present(client, project):
    c, _ = client
    metrics = project["cfg"]["paths"]["artifacts_dir"] / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    test_file = metrics / "final_test_metrics.json"
    test_file.unlink(missing_ok=True)
    assert c.get("/results/evaluation").status_code == 404
    test_file.write_text(json.dumps({"model_version": project["version"], "accuracy": 0.5,
                                     "confusion_matrix": [[1]], "per_class": {}}))
    (metrics / "model_comparison.csv").write_text("model,version,Accuracy\nm,v,0.5\n")
    (metrics / "accuracy_by_resolution.csv").write_text(
        "resolution,val_crops,val_accuracy,test_crops,test_accuracy\n10x10,4.0,0.5,,\n")
    body = c.get("/results/evaluation").json()
    assert body["model_version"] == project["version"] and body["test"]["accuracy"] == 0.5
    assert body["comparison"][0]["Accuracy"] == 0.5 and len(body["class_names"]) == 11
    assert body["by_resolution"][0]["test_crops"] is None   # empty CSV cell -> null, not NaN


def test_training_history(client, project):
    c, _ = client
    rows = c.get(f"/results/history/{project['version']}").json()
    assert rows[0]["epoch"] == 1 and rows[0]["phase"] == "test"
    assert c.get("/results/history/v999_missing").status_code == 404
    assert c.get("/results/history/..%2F..").status_code in (400, 404)


def test_dataset_summary_and_gate_without_files(client):
    c, _ = client
    assert c.get("/dataset/summary").status_code in (200, 404)
    assert c.get("/gate").json() == {"enabled": False}


def test_auth_status_and_verify(client, monkeypatch):
    c, main = client
    assert c.get("/auth/status").json()["admin_key_required"] is False
    monkeypatch.setattr(main, "ADMIN_KEY", "secret")
    assert c.get("/auth/status").json()["admin_key_required"] is True
    assert c.post("/auth/verify", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/auth/verify", headers={"X-API-Key": "secret"}).json()["ok"] is True


def test_uploads_listing_requires_admin_key(client, monkeypatch):
    c, main = client
    monkeypatch.setattr(main, "ADMIN_KEY", "secret")
    assert c.get("/uploads").status_code == 401
    assert c.get("/uploads", headers={"X-API-Key": "secret"}).status_code == 200


def test_cors_allows_dev_server(client):
    c, _ = client
    response = c.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
