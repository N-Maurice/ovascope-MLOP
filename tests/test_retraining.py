"""Retraining: loads the saved model, uses new DB images, makes a new version, keeps the old one."""
import copy

import pytest

from src import database, registry, retrain


@pytest.fixture
def small_cfg(project):
    cfg = copy.deepcopy(project["cfg"])
    cfg["retraining"].update({"epochs": 1, "replay_images_per_class": 2, "min_new_images": 3,
                              "new_image_repeat": 1, "promotion_tolerance": 1.0})   # always promote in tests
    return cfg


def add_uploads(cfg, jpeg_bytes, n, seed0):
    from src.database import save_upload, validate_upload
    from src.preprocessing import load_class_map
    class_map = load_class_map(cfg["paths"]["processed_dir"])
    for k in range(n):
        data = jpeg_bytes(k % 11, seed=seed0 + k)
        checked = validate_upload(f"new_{seed0 + k}.jpg", data, str(k % 11), class_map, {".jpg"}, 10**7, set())
        assert checked["ok"]
        assert save_upload(cfg["paths"]["database"], checked, data)["ok"]


def test_refuses_without_enough_images(small_cfg):
    small_cfg["retraining"]["min_new_images"] = 10_000
    with pytest.raises(retrain.RetrainError, match="at least"):
        retrain.start_retraining(small_cfg, run_in_process=True)


def test_retraining_creates_new_version_from_saved_model(small_cfg, jpeg_bytes, monkeypatch):
    cfg = small_cfg
    models_dir, db_path = cfg["paths"]["models_dir"], cfg["paths"]["database"]
    add_uploads(cfg, jpeg_bytes, 6, seed0=5000)
    before = registry.get_active_version(models_dir)

    loaded = []
    import tensorflow as tf
    original_load = tf.keras.models.load_model
    monkeypatch.setattr(tf.keras.models, "load_model", lambda p, **k: loaded.append(str(p)) or original_load(p, **k))

    job_id = retrain.start_retraining(cfg, run_in_process=True)
    job = database.get_job(db_path, job_id)
    assert job["status"] == "completed", job.get("error")
    assert job["base_version"] == before
    assert any(before in path for path in loaded)                    # the previous saved model was loaded
    assert job["n_new_images"] >= 6
    assert job["candidate_metrics"]["samples"] > 0 and job["active_metrics"]["samples"] > 0
    candidate = job["candidate_version"]
    meta = {v["version"]: v for v in registry.list_versions(models_dir)}
    assert meta[candidate]["parent_version"] == before
    assert registry.version_path(models_dir, before)                  # previous model still available
    assert job["promoted"] and registry.get_active_version(models_dir) == candidate
    assert database.upload_counts(db_path)["pending"] == 0
    registry.rollback(models_dir)
    assert registry.get_active_version(models_dir) == before


def test_failed_retraining_keeps_active_model(small_cfg, jpeg_bytes, monkeypatch):
    cfg = small_cfg
    models_dir, db_path = cfg["paths"]["models_dir"], cfg["paths"]["database"]
    add_uploads(cfg, jpeg_bytes, 4, seed0=7000)
    before = registry.get_active_version(models_dir)
    versions_before = len(registry.list_versions(models_dir))

    def broken(*args, **kwargs):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr("src.preprocessing.make_dataset", broken)
    job_id = retrain.start_retraining(cfg, run_in_process=True)
    job = database.get_job(db_path, job_id)
    assert job["status"] == "failed" and "simulated failure" in job["error"]
    assert registry.get_active_version(models_dir) == before
    assert len(registry.list_versions(models_dir)) == versions_before
    assert database.upload_counts(db_path)["pending"] >= 4            # images remain available


def test_duplicate_jobs_are_prevented(small_cfg):
    db_path = small_cfg["paths"]["database"]
    first = database.create_job_if_idle(db_path)
    assert first is not None
    assert database.create_job_if_idle(db_path) is None
    database.update_job(db_path, first, status="failed", error="test cleanup")


def test_trigger_flag(small_cfg, jpeg_bytes):
    small_cfg["retraining"]["trigger_threshold"] = 1
    add_uploads(small_cfg, jpeg_bytes, 1, seed0=9000)
    assert retrain.retraining_status(small_cfg)["retraining_recommended"]
    small_cfg["retraining"]["trigger_threshold"] = 10_000
    assert not retrain.retraining_status(small_cfg)["retraining_recommended"]
