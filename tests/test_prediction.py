import numpy as np
import pytest
import tensorflow as tf

from src import registry
from src.prediction import Predictor
from src.preprocessing import load_class_map


@pytest.fixture(scope="module")
def predictor(project):
    return Predictor.from_active(project["cfg"])


def test_saved_model_loads(predictor, project):
    assert predictor.version == project["version"]
    assert predictor.model.output_shape[-1] == 11


def test_prediction_fields_are_valid(predictor, jpeg_bytes):
    result = predictor.predict_bytes(jpeg_bytes(3))
    assert 0 <= result["predicted_class_id"] < 11
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_k"]) == 3
    assert result["top_k"][0]["class_id"] == result["predicted_class_id"]
    assert result["model_version"] == predictor.version


def test_class_name_matches_saved_mapping(predictor, jpeg_bytes, project):
    class_map = load_class_map(project["cfg"]["paths"]["processed_dir"])
    result = predictor.predict_bytes(jpeg_bytes(5))
    assert result["predicted_class"] == class_map[result["predicted_class_id"]]["display_name"]


def test_probabilities_sum_to_one(predictor, jpeg_bytes):
    from src.preprocessing import load_image_from_bytes
    probabilities = predictor.predict_probabilities(load_image_from_bytes(jpeg_bytes(1)))
    assert probabilities.shape == (11,)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-4)
    assert (probabilities >= 0).all()


def test_invalid_input_raises(predictor):
    with pytest.raises(ValueError):
        predictor.predict_bytes(b"definitely not an image")


def test_h5_export_matches_keras(project):
    folder = registry.version_path(project["cfg"]["paths"]["models_dir"], project["version"])
    x = np.random.default_rng(0).uniform(0, 255, (2, 224, 224, 3)).astype("float32")
    a = tf.keras.models.load_model(folder / "model.keras", compile=False).predict(x, verbose=0)
    b = tf.keras.models.load_model(folder / "model.h5", compile=False).predict(x, verbose=0)
    assert np.abs(a - b).max() < 1e-4


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model found"):
        Predictor(tmp_path)


def test_reactivating_same_version_keeps_rollback_target(tmp_path, project):
    import json
    import shutil
    models = tmp_path / "models"
    for name in ("vA", "vB"):
        shutil.copytree(registry.version_path(project["cfg"]["paths"]["models_dir"], project["version"]),
                        models / "versions" / name)
    registry.activate_version(models, "vA")
    registry.activate_version(models, "vB")
    registry.activate_version(models, "vB")          # e.g. refresh after adding test metrics
    assert json.loads((models / "active_model.json").read_text())["previous"] == "vA"
    assert registry.rollback(models)["version"] == "vA"
