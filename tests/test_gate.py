"""Egg gate (input check): data helpers, threshold rule, Predictor and API behaviour.

The gate model used here is untrained (no ImageNet download); its decisions are forced through
the saved threshold, which is exactly the mechanism the real gate uses."""
import io
import json
import random

import numpy as np
import pytest
from PIL import Image

from src.gate import (GATE_FOLDER, GATE_METADATA_FILE, GATE_MODEL_FILE, EggGate, NotParasiteEggError,
                      build_gate_model, choose_threshold, gate_report, sample_background_box)


def png_bytes(colour=(200, 30, 30), size=(120, 90)):
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def gate_folder(tmp_path_factory):
    """An untrained gate saved like a real one. The threshold is edited per test."""
    folder = tmp_path_factory.mktemp("models") / GATE_FOLDER
    folder.mkdir()
    build_gate_model(224, weights=None).save(folder / GATE_MODEL_FILE)
    (folder / GATE_METADATA_FILE).write_text(json.dumps({"threshold": 0.5, "input_shape": [224, 224, 3]}))
    return folder


def set_threshold(folder, value):
    (folder / GATE_METADATA_FILE).write_text(json.dumps({"threshold": value, "input_shape": [224, 224, 3]}))


def test_background_box_avoids_eggs():
    rng = random.Random(0)
    eggs = [(100, 100, 50, 50), (300, 200, 80, 60)]
    for _ in range(200):
        box = sample_background_box((640, 480), eggs, 90, rng)
        assert box is not None and box[2] - box[0] == 90 and box[3] - box[1] == 90
        assert 0 <= box[0] and box[2] <= 640 and 0 <= box[1] and box[3] <= 480
        for x, y, w, h in eggs:   # no overlap with the padded egg box
            assert box[2] <= x - 0.25 * w or box[0] >= x + 1.25 * w or box[3] <= y - 0.25 * h or box[1] >= y + 1.25 * h


def test_background_box_none_when_no_space():
    assert sample_background_box((100, 100), [(0, 0, 100, 100)], 50, random.Random(0)) is None


def test_threshold_keeps_target_share_of_eggs():
    scores = np.linspace(0.01, 1.0, 1000)
    threshold = choose_threshold(scores, target_egg_recall=0.995, max_threshold=0.5)
    assert (scores >= threshold).mean() >= 0.995
    assert choose_threshold(np.full(100, 0.99), 0.995, max_threshold=0.5) == 0.5   # capped


def test_gate_report_rates():
    report = gate_report([0.9, 0.8, 0.1, 0.2, 0.7], [1, 1, 0, 0, 0], ["egg", "egg", "cifar100", "cifar100", "imagenette"],
                         threshold=0.5)
    assert report["egg_acceptance_rate"] == 1.0
    assert report["non_egg_rejection_rate"] == pytest.approx(2 / 3)
    assert report["per_source"]["imagenette"]["rate"] == 0.0
    assert report["confusion"]["non_eggs_accepted"] == 1


def test_gate_threshold_controls_decision(gate_folder):
    image = Image.new("RGB", (64, 64), (120, 120, 120))
    set_threshold(gate_folder, 1.01)   # nothing can reach it -> rejected
    assert EggGate(gate_folder).check(image)["is_parasite_egg"] is False
    set_threshold(gate_folder, 0.0)    # everything passes
    check = EggGate(gate_folder).check(image)
    assert check["is_parasite_egg"] is True and 0.0 <= check["egg_score"] <= 1.0


def test_missing_gate_means_no_check(tmp_path):
    assert EggGate.from_models_dir(tmp_path) is None


def test_predictor_rejects_and_accepts_with_gate(project, gate_folder, jpeg_bytes):
    from src import registry
    from src.prediction import Predictor
    version_dir = registry.version_path(project["cfg"]["paths"]["models_dir"], project["version"])
    set_threshold(gate_folder, 1.01)
    predictor = Predictor(version_dir, gate=EggGate(gate_folder))
    with pytest.raises(NotParasiteEggError) as rejected:
        predictor.predict_bytes(png_bytes())
    assert rejected.value.check["is_parasite_egg"] is False
    assert "predicted_class" in predictor.predict_bytes(png_bytes(), check_input=False)   # explicit bypass
    set_threshold(gate_folder, 0.0)
    result = Predictor(version_dir, gate=EggGate(gate_folder)).predict_bytes(jpeg_bytes(1))
    assert result["input_check"]["enabled"] and result["input_check"]["is_parasite_egg"]


def test_predictor_without_gate_reports_disabled(project, jpeg_bytes):
    from src.prediction import Predictor
    assert Predictor.from_active(project["cfg"]).predict_bytes(jpeg_bytes(2))["input_check"] == {"enabled": False}


def test_api_returns_422_and_upload_rejects_non_egg(project, gate_folder):
    from fastapi.testclient import TestClient
    import api.main as main
    with TestClient(main.app) as client:
        predictor = main.MODEL.get()
        set_threshold(gate_folder, 1.01)
        predictor.gate = EggGate(gate_folder)
        try:
            response = client.post("/predict", files={"file": ("car.png", png_bytes(), "image/png")})
            assert response.status_code == 422
            detail = response.json()["detail"]
            assert detail["error"] == "not_a_parasite_egg" and "parasite egg" in detail["message"]
            assert client.get("/health").json()["input_check"]["enabled"] is True
            upload = client.post("/upload", files=[("files", ("tree.png", png_bytes((20, 160, 20)), "image/png"))],
                                 data={"label": "0"}).json()
            assert upload["saved_to_database"] == 0
            assert "does not look like a parasite egg" in upload["rejected"][0]["reason"]
        finally:
            predictor.gate = None
        assert client.get("/health").json()["input_check"] == {"enabled": False}
