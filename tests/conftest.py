"""Shared test fixtures.

Tests run in an isolated temporary "project" (own processed data, models, DB),
built from a tiny synthetic COCO dataset, so they never touch real artifacts and
do not need the 13 GB dataset. Environment variables redirect src.config paths.
"""
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["MONGODB_URI"] = ""   # tests never write to a real MongoDB cluster from .env (they use mongomock)

CLASS_NAMES = ["Ascaris lumbricoides", "Capillaria philippinensis", "Enterobius vermicularis", "Fasciolopsis buski",
               "Hookworm egg", "Hymenolepis diminuta", "Hymenolepis nana", "Opisthorchis viverrine",
               "Paragonimus spp", "Taenia spp. egg", "Trichuris trichiura"]


def synthetic_photo(class_id: int, seed: int, size=(320, 240)):
    """A fake 'microscope photo' with one ellipse whose shape/colour depends on the class."""
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", size, tuple(int(v) for v in rng.integers(170, 230, 3)))
    draw = ImageDraw.Draw(image)
    w, h = 30 + 4 * class_id, 60 - 2 * class_id
    x, y = int(rng.integers(40, size[0] - 80)), int(rng.integers(40, size[1] - 80))
    draw.ellipse([x, y, x + w, y + h], fill=(20 * class_id % 255, 100, 200 - 10 * class_id), outline=(0, 0, 0))
    return image, [float(x), float(y), float(w), float(h)]


def make_coco(folder: Path, n_per_class: int, seed_offset: int, prefix_with_class: bool):
    folder.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for class_id in range(11):
        for k in range(n_per_class):
            image_id = len(images) + 1
            photo, box = synthetic_photo(class_id, seed_offset + image_id)
            name = f"{CLASS_NAMES[class_id]}_{k:04d}.jpg" if prefix_with_class else f"{image_id:04d}.jpg"
            photo.save(folder / name, quality=95)
            images.append({"id": image_id, "file_name": name, "width": photo.width, "height": photo.height})
            annotations.append({"id": image_id, "image_id": image_id, "category_id": class_id, "bbox": box,
                                "area": box[2] * box[3]})
    categories = [{"id": i, "name": n, "supercategory": None} for i, n in enumerate(CLASS_NAMES)]
    return {"images": images, "annotations": annotations, "categories": categories}


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    """Temporary project: synthetic crops + one saved, activated (untrained) baseline model."""
    base = tmp_path_factory.mktemp("ovascope")
    raw = base / "raw"
    train = make_coco(raw / "train", n_per_class=6, seed_offset=0, prefix_with_class=True)
    test = make_coco(raw / "test", n_per_class=2, seed_offset=10_000, prefix_with_class=False)
    (raw / "train.json").write_text(json.dumps(train))
    (raw / "test.json").write_text(json.dumps(test))

    os.environ.update({
        "OVASCOPE_PROCESSED_DIR": str(base / "processed"), "OVASCOPE_MODELS_DIR": str(base / "models"),
        "OVASCOPE_ARTIFACTS_DIR": str(base / "artifacts"), "OVASCOPE_DB": str(base / "db" / "test.db"),
        "OVASCOPE_UPLOADS_DIR": str(base / "uploads"),
    })
    from src.config import load_config
    from src.preprocessing import build_crop_dataset, write_protected_hashes
    build_crop_dataset(raw / "train.json", raw / "train", raw / "test.json", raw / "test", base / "processed",
                       val_fraction=1 / 3, margin=0.1, seed=42, workers=1)
    cfg = load_config()
    write_protected_hashes(cfg["paths"]["processed_dir"], cfg["paths"]["models_dir"] / "protected_test_hashes.txt")

    import pandas as pd
    from src import database, registry
    from src.model import build_baseline_cnn
    from src.train import compute_metrics, save_version
    model = build_baseline_cnn(cfg["training"]["image_size"], 11)
    history = pd.DataFrame({"epoch": [1], "loss": [2.4], "accuracy": [0.1], "val_loss": [2.4],
                            "val_accuracy": [0.1], "learning_rate": [1e-3], "phase": ["test"]})
    dummy = compute_metrics(np.arange(11), np.eye(11), CLASS_NAMES)
    version = save_version(model, "baseline_cnn", cfg, history, dummy, {"training_seconds": 0})
    registry.activate_version(cfg["paths"]["models_dir"], version)
    database.init_db(cfg["paths"]["database"])
    return {"base": base, "cfg": cfg, "version": version}


@pytest.fixture
def jpeg_bytes():
    def make(class_id=0, seed=1):
        image, box = synthetic_photo(class_id, seed)
        buffer = io.BytesIO()
        image.crop((int(box[0]) - 5, int(box[1]) - 5, int(box[0] + box[2]) + 5, int(box[1] + box[3]) + 5)) \
             .save(buffer, "JPEG")
        return buffer.getvalue()
    return make
