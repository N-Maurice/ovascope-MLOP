"""Input check ("egg gate"): is this image a microscope crop of a parasite egg at all?

The 11-class classifier is a *closed-set* model: whatever it receives - a car, a face, a tree -
it must answer with one of the 11 egg classes, often with high confidence. The egg gate is a
separate binary model that runs first and rejects images that do not look like a parasite-egg
crop, so the app can ask the user for a proper image instead of returning a meaningless class.

    python -m src.gate --prepare      # download / build the negative images (once)
    python -m src.gate --train        # train, pick the threshold on validation, evaluate on test

Training data
    egg (label 1)      the single-egg crops of the classification dataset (same train/val/test split)
    not egg (label 0)  * Imagenette (10 ImageNet classes: vehicles, animals, people, buildings, objects)
                       * CIFAR-100 (all 20 superclasses: people, trees, vehicles, animals, food, furniture...)
                       * empty microscope background: patches cut from the dataset's own photos away from
                         every annotated egg (hard negatives: same microscopes and debris, but no egg)
                       * synthetic images: solid colours, gradients and noise
    Splits never mix: background patches come from photos of the matching split, Imagenette /
    CIFAR validation and test images come from their official held-out parts.

Decision rule
    egg_score = P(egg) from the gate. The threshold is chosen on the *validation* set so that at
    least `target_egg_recall` (99.5 %) of real egg crops are accepted; the test set only reports
    how well that threshold works.

Model
    MobileNetV2 (ImageNet weights, frozen) + global average pooling + dropout + 1 sigmoid unit.
    ImageNet features separate microscopy from everyday photos very well, so a small head suffices.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

GATE_FOLDER = "gate"                # models/gate/ holds the trained gate
GATE_MODEL_FILE = "egg_gate.keras"
GATE_METADATA_FILE = "gate_metadata.json"
NEGATIVE_FIELDS = ["path", "split", "source", "detail"]
CIFAR100_SUPERCLASSES = [
    "aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables",
    "household_electrical_devices", "household_furniture", "insects", "large_carnivores",
    "large_man-made_outdoor_things", "large_natural_outdoor_scenes", "large_omnivores_and_herbivores",
    "medium_mammals", "non-insect_invertebrates", "people", "reptiles", "small_mammals", "trees",
    "vehicles_1", "vehicles_2"]
IMAGENETTE_CLASSES = {"n01440764": "tench (fish)", "n02102040": "English springer (dog)",
                      "n02979186": "cassette player", "n03000684": "chain saw", "n03028079": "church",
                      "n03394916": "French horn", "n03417042": "garbage truck", "n03425413": "gas pump",
                      "n03445777": "golf ball", "n03888605": "parachute"}
NOT_EGG_MESSAGE = ("This image does not look like a microscope image of a parasite egg. "
                   "Please upload a microscope photo cropped around a single egg.")


class NotParasiteEggError(Exception):
    """Raised by the Predictor when the egg gate rejects an image."""

    def __init__(self, check: dict):
        super().__init__(NOT_EGG_MESSAGE)
        self.check = check


# ---------------------------------------------------------------------------
# Part 1 - negative ("not an egg") images
# ---------------------------------------------------------------------------

def _save_jpeg(image: Image.Image, path: Path) -> None:
    """All negatives are stored as JPEG quality 95, exactly like the egg crops, so the file
    format and compression cannot become a shortcut the gate learns instead of content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, quality=95)


def _download(url: str, destination: Path, log=print) -> Path:
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {url} ...")
    tmp = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(destination)
    return destination


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    """Extract a downloaded tar archive, refusing members that would land outside `destination`."""
    destination = Path(destination).resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not (destination / member.name).resolve().is_relative_to(destination) or member.issym() \
                    or member.islnk():
                raise ValueError(f"Unsafe member in {archive.name}: {member.name}")
        try:
            tar.extractall(destination, filter="data")   # Python >= 3.12: also blocks special files
        except TypeError:
            tar.extractall(destination)


def imagenette_negatives(gate_dir: Path, counts: dict, url: str, seed: int, log=print) -> list[dict]:
    """Imagenette 160 px: `train` images from its train folder; `val` and `test` from its official
    validation folder, split in half by a hash of the file name (never overlapping)."""
    root = Path(gate_dir) / "downloads" / "imagenette2-160"
    if not root.exists():
        archive = _download(url, Path(gate_dir) / "downloads" / "imagenette2-160.tgz", log)
        log("Extracting Imagenette ...")
        _safe_extract_tar(archive, root.parent)
    rng = random.Random(seed)
    pools = {"train": sorted((root / "train").rglob("*.JPEG")), "val": [], "test": []}
    for path in sorted((root / "val").rglob("*.JPEG")):
        digest = int(hashlib.md5(path.name.encode()).hexdigest(), 16)
        pools["val" if digest % 2 == 0 else "test"].append(path)
    rows = []
    for split, files in pools.items():
        for i, source in enumerate(rng.sample(files, min(counts[split], len(files)))):
            target = Path(gate_dir) / "images" / "imagenette" / split / f"{source.stem}.jpg"
            if not target.exists():
                with Image.open(source) as image:
                    _save_jpeg(image, target)
            rows.append({"path": str(target.relative_to(gate_dir)), "split": split, "source": "imagenette",
                         "detail": IMAGENETTE_CLASSES.get(source.parent.name, source.parent.name)})
    return rows


def load_cifar100(gate_dir: Path, url: str, md5: str, log=print):
    """CIFAR-100 (python version) as uint8 arrays with coarse (superclass) labels.

    The archive is downloaded once to data/gate/downloads/ (it may also be placed there by hand,
    e.g. after a faster parallel download) and its MD5 is checked against the published value
    *before* the pickled batches inside are read."""
    import pickle
    archive = _download(url, Path(gate_dir) / "downloads" / "cifar-100-python.tar.gz", log)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != md5:
        raise ValueError(f"CIFAR-100 archive checksum mismatch ({digest} != {md5}); delete {archive} and retry")
    root = Path(gate_dir) / "downloads"
    if not (root / "cifar-100-python" / "train").exists():
        _safe_extract_tar(archive, root)
    splits = {}
    for name in ("train", "test"):
        with open(root / "cifar-100-python" / name, "rb") as f:
            batch = pickle.load(f, encoding="latin1")
        images = np.asarray(batch["data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        splits[name] = (images, np.asarray(batch["coarse_labels"]))
    return splits["train"], splits["test"]


def cifar100_negatives(gate_dir: Path, per_superclass: dict, url: str, md5: str, seed: int, log=print) -> list[dict]:
    """CIFAR-100 (32 x 32 photos of people, trees, vehicles, animals, objects ...): an equal number
    per superclass. `train` from the CIFAR train set; `val` and `test` from disjoint halves of the
    CIFAR test set."""
    (x_train, y_train), (x_test, y_test) = load_cifar100(gate_dir, url, md5, log)
    rng = np.random.default_rng(seed)
    rows = []
    halves = rng.permutation(len(x_test))
    test_index = {"val": halves[: len(halves) // 2], "test": halves[len(halves) // 2:]}
    for split in ("train", "val", "test"):
        images, labels = (x_train, y_train) if split == "train" else (x_test[test_index[split]], y_test[test_index[split]])
        for superclass, name in enumerate(CIFAR100_SUPERCLASSES):
            candidates = np.nonzero(labels == superclass)[0]
            for index in rng.choice(candidates, min(per_superclass[split], len(candidates)), replace=False):
                target = Path(gate_dir) / "images" / "cifar100" / split / f"{name}_{split}_{index}.jpg"
                if not target.exists():
                    _save_jpeg(Image.fromarray(images[index]), target)
                rows.append({"path": str(target.relative_to(gate_dir)), "split": split, "source": "cifar100",
                             "detail": name})
    return rows


def sample_background_box(image_size: tuple[int, int], egg_boxes, side: int, rng: random.Random,
                          margin: float = 0.25, tries: int = 50):
    """A square (left, top, right, bottom) of `side` px inside the image that does not touch any egg
    box, each box padded by `margin` of its size. Returns None if no free spot is found."""
    width, height = image_size
    side = int(min(side, width, height))
    padded = [(x - margin * w, y - margin * h, x + w * (1 + margin), y + h * (1 + margin))
              for x, y, w, h in egg_boxes]
    for _ in range(tries):
        left, top = rng.randint(0, width - side), rng.randint(0, height - side)
        box = (left, top, left + side, top + side)
        if all(box[2] <= b[0] or box[0] >= b[2] or box[3] <= b[1] or box[1] >= b[3] for b in padded):
            return box
    return None


def background_negatives(cfg: dict, gate_dir: Path, counts: dict, seed: int, log=print) -> list[dict]:
    """Patches of empty microscope background from the dataset's own photos (hard negatives).

    Patch sizes are drawn from the real egg-crop sizes, so size cannot separate the classes.
    Photos are taken from the split they belong to (train / val photos per the crop manifest, the
    official test photos for test), so no photo contributes to two gate splits."""
    from src.preprocessing import index_image_files, load_coco, read_manifest
    paths, ds = cfg["paths"], cfg["dataset"]
    manifest = read_manifest(paths["processed_dir"])
    split_of_photo = {row["source_image"]: row["split"] for row in manifest}
    crop_sides = [json.loads(row["crop_box_ltrb"])[2] - json.loads(row["crop_box_ltrb"])[0] for row in manifest]
    sources = [(paths["extracted_dir"] / ds["train_json"], paths["extracted_dir"] / ds["train_images"]),
               (paths["raw_dir"] / ds["test_json_in_raw"], paths["extracted_dir"] / ds["test_images"])]
    photos = {"train": [], "val": [], "test": []}
    for json_path, image_dir in sources:
        if not Path(json_path).exists() or not Path(image_dir).exists():
            log(f"  background: {json_path} or {image_dir} not found - skipped")
            continue
        _, images, annotations = load_coco(json_path)
        files = index_image_files(image_dir)
        for image_id, image in images.items():
            split = split_of_photo.get(image["file_name"])
            path = files.get(Path(image["file_name"]).name)
            if split and path:
                photos[split].append((path, [a["bbox"] for a in annotations.get(image_id, [])]))
    rng = random.Random(seed)
    rows = []
    for split, wanted in counts.items():
        rng.shuffle(photos[split])
        made = 0
        for path, boxes in photos[split]:
            if made >= wanted:
                break
            target = Path(gate_dir) / "images" / "background" / split / f"{Path(path).stem}_bg.jpg"
            if not target.exists():
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    box = sample_background_box(image.size, boxes, rng.choice(crop_sides), rng)
                    if box is None:
                        continue
                    _save_jpeg(image.crop(box), target)
            rows.append({"path": str(target.relative_to(gate_dir)), "split": split, "source": "background",
                         "detail": Path(path).name})
            made += 1
        if made < wanted:
            log(f"  background/{split}: only {made} of {wanted} patches available")
    return rows


def synthetic_negatives(gate_dir: Path, counts: dict, seed: int, size: int = 256) -> list[dict]:
    """Blank, gradient and noise images: what a user gets from a covered lens, a screenshot
    background or a corrupted capture."""
    rng = np.random.default_rng(seed)
    rows = []
    for split, n in counts.items():
        for i in range(n):
            kind = ("solid", "gradient", "noise", "blurred_noise")[i % 4]
            colour = rng.integers(0, 256, 3)
            if kind == "solid":
                array = np.broadcast_to(colour, (size, size, 3))
            elif kind == "gradient":
                other = rng.integers(0, 256, 3)
                t = np.linspace(0, 1, size)[:, None, None] if i % 8 < 4 else np.linspace(0, 1, size)[None, :, None]
                array = colour * (1 - t) + other * t
                array = np.broadcast_to(array, (size, size, 3))
            else:
                array = rng.integers(0, 256, (size, size, 3))
            image = Image.fromarray(np.asarray(array, dtype=np.uint8))
            if kind == "blurred_noise":
                from PIL import ImageFilter
                image = image.filter(ImageFilter.GaussianBlur(rng.uniform(2, 8)))
            target = Path(gate_dir) / "images" / "synthetic" / split / f"{kind}_{split}_{i}.jpg"
            _save_jpeg(image, target)
            rows.append({"path": str(target.relative_to(gate_dir)), "split": split, "source": "synthetic",
                         "detail": kind})
    return rows


def prepare_gate_data(cfg: dict, log=print) -> dict:
    """Build every negative source (skipping a source that cannot be downloaded, e.g. offline) and
    write data/gate/negatives.csv. Re-running reuses files that already exist."""
    g = cfg["egg_gate"]
    gate_dir = Path(cfg["paths"]["gate_data_dir"])
    seed = cfg["project"]["random_seed"]
    rows, skipped = [], {}
    builders = {
        "imagenette": lambda: imagenette_negatives(gate_dir, g["imagenette"], g["imagenette_url"], seed, log),
        "cifar100": lambda: cifar100_negatives(gate_dir, g["cifar100_per_superclass"], g["cifar100_url"],
                                               g["cifar100_md5"], seed, log),
        "background": lambda: background_negatives(cfg, gate_dir, g["background_patches"], seed, log),
        "synthetic": lambda: synthetic_negatives(gate_dir, g["synthetic"], seed),
    }
    for name, build in builders.items():
        try:
            new_rows = build()
            rows += new_rows
            log(f"  {name}: {len(new_rows)} images")
        except Exception as error:   # one unavailable source must not stop the others
            skipped[name] = f"{error.__class__.__name__}: {error}"
            log(f"  {name}: SKIPPED ({skipped[name]})")
    gate_dir.mkdir(parents=True, exist_ok=True)
    with open(gate_dir / "negatives.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NEGATIVE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter((r["source"], r["split"]) for r in rows)
    return {"per_source_split": {f"{s}/{p}": n for (s, p), n in sorted(counts.items())},
            "total": len(rows), "skipped_sources": skipped}


def read_negatives(gate_dir: Path) -> list[dict]:
    path = Path(gate_dir) / "negatives.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run prepare_gate_data(cfg) / python -m src.gate --prepare")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def gate_split(cfg: dict, split: str):
    """(file paths, labels, sources) for one gate split: egg crops (1) + negatives (0)."""
    from src.preprocessing import list_split
    gate_dir = Path(cfg["paths"]["gate_data_dir"])
    egg_files, _ = list_split(cfg["paths"]["processed_dir"], split)
    negatives = [r for r in read_negatives(gate_dir) if r["split"] == split]
    files = egg_files + [str(gate_dir / r["path"]) for r in negatives]
    labels = [1] * len(egg_files) + [0] * len(negatives)
    sources = ["egg"] * len(egg_files) + [r["source"] for r in negatives]
    return files, labels, sources


# ---------------------------------------------------------------------------
# Part 2 - model, training, threshold, evaluation
# ---------------------------------------------------------------------------

def build_gate_model(image_size: int, dropout: float = 0.2, weights: str | None = "imagenet"):
    """MobileNetV2 feature extractor (frozen, ImageNet weights) + a single sigmoid output P(egg).
    Input: raw RGB in [0, 255], like the classifier; rescaling happens inside the model."""
    import tensorflow as tf
    layers = tf.keras.layers
    backbone = tf.keras.applications.MobileNetV2(input_shape=(image_size, image_size, 3), include_top=False,
                                                 weights=weights)
    backbone.trainable = False
    inputs = layers.Input(shape=(image_size, image_size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale_minus1_1")(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gate_pool")(x)
    x = layers.Dropout(dropout, name="gate_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="egg_probability")(x)
    return tf.keras.Model(inputs, outputs, name="ovascope_egg_gate")


def _binary_dataset(files, labels, image_size, batch_size, training, seed):
    import tensorflow as tf
    from src.preprocessing import make_dataset
    dataset = make_dataset(files, labels, image_size, batch_size, training=training, seed=seed)
    return dataset.map(lambda x, y: (x, tf.cast(tf.expand_dims(y, -1), tf.float32)))


def gate_scores(model, files, image_size: int, batch_size: int) -> np.ndarray:
    from src.preprocessing import make_dataset
    dataset = make_dataset(files, [0] * len(files), image_size, batch_size, training=False).map(lambda x, y: x)
    return model.predict(dataset, verbose=0).ravel()


def choose_threshold(egg_scores, target_egg_recall: float, max_threshold: float = 0.5) -> float:
    """Highest threshold that still accepts `target_egg_recall` of the validation eggs
    (capped at `max_threshold`, so the gate never demands more than 'more likely egg than not')."""
    quantile = float(np.quantile(np.asarray(egg_scores), 1 - target_egg_recall))
    return float(min(max_threshold, max(quantile, 1e-4)))


def gate_report(scores, labels, sources, threshold: float) -> dict:
    """Acceptance of real eggs, rejection of each negative source, ROC-AUC, confusion counts."""
    from sklearn.metrics import roc_auc_score
    scores, labels, sources = np.asarray(scores), np.asarray(labels), np.asarray(sources)
    accepted = scores >= threshold
    report = {
        "threshold": threshold, "images": int(len(labels)),
        "egg_images": int(labels.sum()), "non_egg_images": int((labels == 0).sum()),
        "egg_acceptance_rate": float(accepted[labels == 1].mean()),          # = recall of class "egg"
        "non_egg_rejection_rate": float((~accepted)[labels == 0].mean()),    # = specificity
        "accuracy": float((accepted == (labels == 1)).mean()),
        "roc_auc": float(roc_auc_score(labels, scores)) if 0 < labels.sum() < len(labels) else None,
        "confusion": {"eggs_accepted": int((accepted & (labels == 1)).sum()),
                      "eggs_rejected": int((~accepted & (labels == 1)).sum()),
                      "non_eggs_accepted": int((accepted & (labels == 0)).sum()),
                      "non_eggs_rejected": int((~accepted & (labels == 0)).sum())},
        "per_source": {},
    }
    for source in sorted(set(sources)):
        mask = sources == source
        rate = accepted[mask].mean() if source == "egg" else (~accepted[mask]).mean()
        report["per_source"][source] = {"images": int(mask.sum()),
                                        "rate": float(rate),
                                        "rate_meaning": "accepted" if source == "egg" else "rejected",
                                        "median_score": float(np.median(scores[mask]))}
    return report


def train_gate(cfg: dict, epochs: int | None = None, log=print) -> dict:
    """Train the gate, choose the threshold on validation, evaluate on test, save to models/gate/."""
    import tensorflow as tf
    from src.train import describe_environment, set_seeds, sha256_of
    g, t = cfg["egg_gate"], cfg["training"]
    seed, size, batch = cfg["project"]["random_seed"], t["image_size"], t["batch_size"]
    set_seeds(seed)
    data = {split: gate_split(cfg, split) for split in ("train", "val", "test")}
    for split, (files, labels, sources) in data.items():
        log(f"{split}: {sum(labels)} egg crops + {len(labels) - sum(labels)} non-egg images "
            f"({dict(Counter(s for s in sources if s != 'egg'))})")

    train_files, train_labels, _ = data["train"]
    positives = sum(train_labels)
    negatives = len(train_labels) - positives
    class_weight = {0: len(train_labels) / (2 * negatives), 1: len(train_labels) / (2 * positives)}
    model = build_gate_model(size, dropout=g["dropout"])
    model.compile(optimizer=tf.keras.optimizers.Adam(g["learning_rate"]),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")])
    run_dir = Path(cfg["paths"]["models_dir"]) / "checkpoints" / "egg_gate"
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    history = model.fit(
        _binary_dataset(train_files, train_labels, size, batch, True, seed),
        validation_data=_binary_dataset(data["val"][0], data["val"][1], size, batch, False, seed),
        epochs=epochs or g["epochs"], class_weight=class_weight, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
                   tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv"))])
    seconds = time.time() - start

    val_files, val_labels, val_sources = data["val"]
    val_scores = gate_scores(model, val_files, size, batch)
    egg_val_scores = val_scores[np.asarray(val_labels) == 1]
    threshold = choose_threshold(egg_val_scores, g["target_egg_recall"], g["max_threshold"])
    val_report = gate_report(val_scores, val_labels, val_sources, threshold)
    test_files, test_labels, test_sources = data["test"]
    test_scores = gate_scores(model, test_files, size, batch)
    test_report = gate_report(test_scores, test_labels, test_sources, threshold)
    log(f"Threshold (validation, egg recall >= {g['target_egg_recall']}): {threshold:.4f}")
    log(f"Test: eggs accepted {test_report['egg_acceptance_rate']:.4f} | "
        f"non-eggs rejected {test_report['non_egg_rejection_rate']:.4f} | ROC-AUC {test_report['roc_auc']:.4f}")

    out = Path(cfg["paths"]["models_dir"]) / GATE_FOLDER
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / GATE_MODEL_FILE)
    sources_used = sorted({s for s in data["train"][2] if s != "egg"})
    metadata = {
        "name": "ovascope_egg_gate", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "architecture": "MobileNetV2 (ImageNet, frozen) + GAP + dropout + sigmoid",
        "input_shape": [size, size, 3], "input_range": "[0, 255] RGB (model rescales internally)",
        "threshold": threshold, "target_egg_recall": g["target_egg_recall"],
        "negative_sources": sources_used,
        "training_images": {"egg": positives, "non_egg": negatives},
        "epochs_completed": len(history.history["loss"]), "training_seconds": round(seconds, 1),
        "history": {k: [float(v) for v in values] for k, values in history.history.items()},
        "validation": val_report, "test": test_report,
        "model_size_mb": round((out / GATE_MODEL_FILE).stat().st_size / 1e6, 2),
        "sha256": sha256_of(out / GATE_MODEL_FILE), "environment": describe_environment(),
    }
    (out / GATE_METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    np.save(Path(cfg["paths"]["artifacts_dir"]) / "metrics" / "gate_test_scores.npy", test_scores)
    return {"metadata": metadata, "scores": {"val": val_scores, "test": test_scores}, "data": data}


# ---------------------------------------------------------------------------
# Part 3 - inference (used by src.prediction.Predictor and the API)
# ---------------------------------------------------------------------------

class EggGate:
    """Loaded egg gate: `check(image)` -> {"is_parasite_egg", "egg_score", "threshold"}."""

    def __init__(self, folder: Path):
        import tensorflow as tf
        folder = Path(folder)
        self.model = tf.keras.models.load_model(folder / GATE_MODEL_FILE, compile=False)
        self.metadata = json.loads((folder / GATE_METADATA_FILE).read_text())
        self.threshold = float(self.metadata["threshold"])
        self.image_size = int(self.metadata["input_shape"][0])
        self._forward = tf.function(lambda batch: self.model(batch, training=False),
                                    input_signature=[tf.TensorSpec([None, self.image_size, self.image_size, 3],
                                                                   tf.float32)])

    @classmethod
    def from_models_dir(cls, models_dir: Path) -> "EggGate | None":
        """The trained gate, or None if none has been trained yet (predictions then run unchecked)."""
        folder = Path(models_dir) / GATE_FOLDER
        if not ((folder / GATE_MODEL_FILE).exists() and (folder / GATE_METADATA_FILE).exists()):
            return None
        return cls(folder)

    def score(self, image) -> float:
        from src.preprocessing import preprocess_pil_image
        return float(self._forward(preprocess_pil_image(image, self.image_size)).numpy().ravel()[0])

    def check(self, image) -> dict:
        score = self.score(image)
        return {"is_parasite_egg": score >= self.threshold, "egg_score": round(score, 4),
                "threshold": round(self.threshold, 4)}

    def info(self) -> dict:
        return {"enabled": True, "threshold": self.threshold, "created_at": self.metadata.get("created_at"),
                "test_egg_acceptance_rate": self.metadata.get("test", {}).get("egg_acceptance_rate"),
                "test_non_egg_rejection_rate": self.metadata.get("test", {}).get("non_egg_rejection_rate")}


def main(argv=None):
    from src.config import load_config
    from src.train import configure_gpu
    parser = argparse.ArgumentParser(description="Prepare data for and train the OvaScope egg gate")
    parser.add_argument("--prepare", action="store_true", help="download/build the non-egg images")
    parser.add_argument("--train", action="store_true", help="train, choose threshold, evaluate, save")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args(argv)
    cfg = load_config()
    if args.prepare:
        print(json.dumps(prepare_gate_data(cfg), indent=2))
    if args.train:
        configure_gpu()
        result = train_gate(cfg, epochs=args.epochs)
        print(json.dumps({k: result["metadata"][k] for k in ("threshold", "validation", "test")}, indent=2))
    if not (args.prepare or args.train):
        parser.print_help()


if __name__ == "__main__":
    main()
