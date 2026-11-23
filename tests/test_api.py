import io
import zipfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(project):
    import api.main as main   # imported after the project fixture set the environment
    with TestClient(main.app) as test_client:
        yield test_client, main


def test_health(client, project):
    c, _ = client
    body = c.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] and body["model_version"] == project["version"]


def test_predict_valid_image(client, jpeg_bytes):
    c, _ = client
    response = c.post("/predict", files={"file": ("egg.jpg", jpeg_bytes(2), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert {"predicted_class_id", "predicted_class", "confidence", "model_version"} <= body.keys()
    assert response.headers["X-Served-By"]


def test_predict_rejects_corrupt_file(client):
    c, _ = client
    response = c.post("/predict", files={"file": ("bad.jpg", b"garbage", "image/jpeg")})
    assert response.status_code == 400


def test_metrics_count_requests(client, jpeg_bytes):
    c, _ = client
    before = c.get("/metrics").json()["prediction_requests_total"]
    c.post("/predict", files={"file": ("egg.jpg", jpeg_bytes(4), "image/jpeg")})
    after = c.get("/metrics").json()
    assert after["prediction_requests_total"] == before + 1
    assert after["uptime_seconds"] > 0


def test_upload_valid_invalid_and_duplicate(client, jpeg_bytes):
    c, _ = client
    good, other = jpeg_bytes(1, seed=101), jpeg_bytes(2, seed=102)
    response = c.post("/upload", data={"labels": "1,Enterobius vermicularis,1"},
                      files=[("files", ("a.jpg", good)), ("files", ("b.jpg", other)), ("files", ("c.jpg", b"junk"))])
    body = response.json()
    assert response.status_code == 200
    assert body["saved_to_database"] == 2
    assert len(body["rejected"]) == 1
    again = c.post("/upload", data={"label": "1"}, files=[("files", ("a_copy.jpg", good))]).json()
    assert again["saved_to_database"] == 0 and "duplicate" in again["rejected"][0]["reason"]
    unknown = c.post("/upload", data={"label": "Not a parasite"},
                     files=[("files", ("x.jpg", jpeg_bytes(3, seed=103)))]).json()
    assert "unknown class label" in unknown["rejected"][0]["reason"]
    records = c.get("/uploads").json()
    assert records["counts"]["total"] >= 2
    assert c.get(f"/uploads/{records['records'][0]['id']}/image").status_code == 200


def test_upload_rejects_protected_test_image(client, project):
    from src.preprocessing import read_manifest
    c, _ = client
    processed = project["cfg"]["paths"]["processed_dir"]
    test_row = next(r for r in read_manifest(processed) if r["split"] == "test")
    data = (processed / test_row["crop_path"]).read_bytes()
    body = c.post("/upload", data={"label": str(test_row["class_id"])}, files=[("files", ("t.jpg", data))]).json()
    assert body["saved_to_database"] == 0 and "protected" in body["rejected"][0]["reason"]


def test_upload_zip_checks_folder_names(client, jpeg_bytes):
    c, _ = client
    good = io.BytesIO()
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("04_hookworm_egg/one.jpg", jpeg_bytes(4, seed=201))
        z.writestr("Hymenolepis nana/two.jpg", jpeg_bytes(6, seed=202))
    body = c.post("/upload", files=[("files", ("batch.zip", good.getvalue()))]).json()
    assert body["saved_to_database"] == 2
    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("not_a_class/one.jpg", jpeg_bytes(4, seed=203))
    response = c.post("/upload", files=[("files", ("bad.zip", bad.getvalue()))])
    assert response.status_code == 400


def test_unsafe_filename_is_sanitised(project, jpeg_bytes):
    from src.database import validate_upload
    from src.preprocessing import load_class_map
    checked = validate_upload("../../etc/passwd.jpg", jpeg_bytes(0, seed=7), "0",
                              load_class_map(project["cfg"]["paths"]["processed_dir"]), {".jpg"}, 10**7, set())
    assert checked["ok"] and checked["filename"] == "passwd.jpg"


def test_admin_key_required_when_configured(client, monkeypatch):
    c, main = client
    monkeypatch.setattr(main, "ADMIN_KEY", "secret")
    assert c.post("/retrain").status_code == 401
    assert c.post("/retrain", headers={"X-API-Key": "wrong"}).status_code == 401


def test_retrain_status_and_unknown_job(client):
    c, _ = client
    status = c.get("/retrain/status").json()
    assert {"pending_new_images", "trigger_threshold", "retraining_recommended"} <= status.keys()
    assert c.get("/retrain/status/does-not-exist").status_code == 404


def test_model_info_and_versions(client, project):
    c, _ = client
    assert c.get("/model/info").json()["version"] == project["version"]
    versions = c.get("/model/versions").json()
    assert any(v["active"] for v in versions)
