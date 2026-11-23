"""Image loading, dataset preparation and preprocessing for OvaScope.

This module is the single source of truth for how images are prepared.
The notebook, training, evaluation, prediction API and retraining all call
these functions, so training-time and serving-time preprocessing cannot drift.

Part A  (no TensorFlow needed)
    Turn the official Chula-ParasiteEgg-11 detection dataset (COCO boxes) into a
    classification dataset of single-egg crops, with a manifest and class map.

Part B  (TensorFlow)
    Resize + batch images for the model, and training-only augmentation.

Design decisions
    * Crops are square (box padded by `margin`, then squared), so resizing to
      224x224 never stretches an egg.
    * The model receives raw RGB pixel values in [0, 255]. Each model rescales
      internally (see src/model.py), so normalisation happens exactly once.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Display-name normalisation. The official label strings stay untouched in
# class_map.json ("original_name"); only display names and folder names change.
DISPLAY_NAMES = {
    "Opisthorchis viverrine": "Opisthorchis viverrini",
    "Paragonimus spp": "Paragonimus spp.",
}

MANIFEST_FIELDS = ["crop_id", "crop_path", "split", "class_id", "class_name", "folder",
                   "source_image", "annotation_id", "bbox_xywh", "crop_box_ltrb"]


# ---------------------------------------------------------------------------
# Part A - dataset preparation
# ---------------------------------------------------------------------------

def md5_of_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archives(raw_dir: Path, expected_md5: dict) -> dict:
    """Return {filename: "ok" | "missing" | "checksum mismatch"} for the official files."""
    status = {}
    for name, md5 in expected_md5.items():
        path = Path(raw_dir) / name
        if not path.exists():
            status[name] = "missing"
        else:
            status[name] = "ok" if md5_of_file(path) == md5 else "checksum mismatch"
    return status


def extract_archive(zip_path: Path, destination: Path) -> None:
    """Unzip, refusing any member that would land outside `destination`."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            target = (destination / member).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe path in archive: {member}")
        archive.extractall(destination)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_class_map(categories: list[dict]) -> list[dict]:
    """Class map that keeps the official ids and names, plus display names and folders."""
    class_map = []
    for category in sorted(categories, key=lambda c: c["id"]):
        display_name = DISPLAY_NAMES.get(category["name"], category["name"])
        class_map.append({
            "class_id": category["id"],
            "original_name": category["name"],
            "display_name": display_name,
            "folder": f"{category['id']:02d}_{slugify(display_name)}",
        })
    return class_map


def load_coco(path: Path):
    data = json.loads(Path(path).read_text())
    images = {image["id"]: image for image in data["images"]}
    annotations = defaultdict(list)
    for annotation in data["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    return data["categories"], images, annotations


def square_crop_box(bbox_xywh, margin: float, image_width: int, image_height: int):
    """Pad a COCO [x, y, w, h] box by `margin` per side, make it square, keep it inside the image."""
    x, y, w, h = bbox_xywh
    center_x, center_y = x + w / 2, y + h / 2
    side = max(w, h) * (1 + 2 * margin)
    side = round(min(side, image_width, image_height))
    left = round(min(max(center_x - side / 2, 0), image_width - side))
    top = round(min(max(center_y - side / 2, 0), image_height - side))
    return left, top, left + side, top + side


def split_train_val(images: dict, annotations: dict, val_fraction: float, seed: int) -> dict:
    """Assign each *source image* to train or val.

    Stratified by the image's class and grouped by source image, so all crops
    from one photo share a split (no photo appears in both train and val).
    """
    image_ids_by_class = defaultdict(list)
    for image_id in images:
        if annotations.get(image_id):
            image_ids_by_class[annotations[image_id][0]["category_id"]].append(image_id)
    rng = random.Random(seed)
    split_of_image = {}
    for class_id in sorted(image_ids_by_class):
        image_ids = sorted(image_ids_by_class[class_id])
        rng.shuffle(image_ids)
        n_val = round(len(image_ids) * val_fraction)
        for position, image_id in enumerate(image_ids):
            split_of_image[image_id] = "val" if position < n_val else "train"
    return split_of_image


def index_image_files(folder: Path) -> dict:
    return {p.name: p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS}


def _crop_one_source_image(job):
    """Crop every annotated egg in one source photo (runs in a worker process)."""
    source_path, annotations, split, margin, out_dir, class_folders, class_names, file_name = job
    rows = []
    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        for annotation in annotations:
            box = square_crop_box(annotation["bbox"], margin, image.width, image.height)
            folder = class_folders[annotation["category_id"]]
            crop_id = f"{Path(source_path).stem}_ann{annotation['id']}"
            destination = Path(out_dir) / "crops" / split / folder / f"{crop_id}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.crop(box).save(destination, quality=95)
            rows.append({
                "crop_id": crop_id,
                "crop_path": str(destination.relative_to(out_dir)),
                "split": split,
                "class_id": annotation["category_id"],
                "class_name": class_names[annotation["category_id"]],
                "folder": folder,
                "source_image": file_name,
                "annotation_id": annotation["id"],
                "bbox_xywh": json.dumps(annotation["bbox"]),
                "crop_box_ltrb": json.dumps(list(box)),
            })
    return rows


def build_crop_dataset(train_json: Path, train_images: Path, test_json: Path, test_images: Path,
                       out_dir: Path, val_fraction: float = 0.15, margin: float = 0.10,
                       seed: int = 42, workers: int = 4) -> dict:
    """Create crops/{train,val,test}/<class>/, class_map.json, manifest.csv, summary.json.

    train/val come from the official training set; test is the official test set
    with its released labels (never used for training or model selection).
    The output is deterministic for a given seed, margin and Pillow version.
    """
    out_dir = Path(out_dir)
    categories, train_coco_images, train_annotations = load_coco(train_json)
    test_categories, test_coco_images, test_annotations = load_coco(test_json)
    if [c["name"] for c in categories] != [c["name"] for c in test_categories]:
        raise ValueError("Train and test annotation files disagree on the category list")

    class_map = build_class_map(categories)
    class_folders = {c["class_id"]: c["folder"] for c in class_map}
    class_names = {c["class_id"]: c["display_name"] for c in class_map}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "class_map.json").write_text(json.dumps(class_map, indent=2))

    split_of_image = split_train_val(train_coco_images, train_annotations, val_fraction, seed)
    jobs, missing = [], []
    for coco_images, annotations, image_folder, split_for in [
        (train_coco_images, train_annotations, train_images, lambda i: split_of_image.get(i, "train")),
        (test_coco_images, test_annotations, test_images, lambda i: "test"),
    ]:
        files = index_image_files(image_folder)
        for image_id, image in coco_images.items():
            if not annotations.get(image_id):
                continue
            source_path = files.get(Path(image["file_name"]).name)
            if source_path is None:
                missing.append(image["file_name"])
                continue
            jobs.append((source_path, annotations[image_id], split_for(image_id), margin,
                         out_dir, class_folders, class_names, image["file_name"]))

    rows = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for job_rows in pool.map(_crop_one_source_image, jobs, chunksize=32):
            rows.extend(job_rows)
    rows.sort(key=lambda r: (r["split"], r["class_id"], r["crop_id"]))

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((r["split"], r["class_id"]) for r in rows)
    summary = {split: {class_folders[c]: counts[(split, c)] for c in sorted(class_folders)}
               for split in ("train", "val", "test")}
    summary["missing_images"] = missing
    summary["settings"] = {"val_fraction": val_fraction, "margin": margin, "seed": seed}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def read_manifest(processed_dir: Path) -> list[dict]:
    with open(Path(processed_dir) / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["class_id"] = int(row["class_id"])
    return rows


def load_class_map(path: Path) -> list[dict]:
    """Load class_map.json (accepts a file or the folder that contains it)."""
    path = Path(path)
    if path.is_dir():
        path = path / "class_map.json"
    class_map = json.loads(path.read_text())
    ids = [c["class_id"] for c in class_map]
    if ids != list(range(len(ids))):
        raise ValueError(f"class_map.json ids must be 0..N-1 in order, got {ids}")
    return class_map


def verify_crop_dataset(processed_dir: Path, check_decode: bool = True) -> dict:
    """Check that every crop in the manifest exists, decodes, and sits in the right folder."""
    processed_dir = Path(processed_dir)
    rows = read_manifest(processed_dir)
    class_map = load_class_map(processed_dir)
    folder_of = {c["class_id"]: c["folder"] for c in class_map}
    missing, corrupt, wrong_folder = [], [], []
    for row in rows:
        path = processed_dir / row["crop_path"]
        if not path.exists():
            missing.append(row["crop_path"])
            continue
        if path.parent.name != folder_of[row["class_id"]] or path.parent.parent.name != row["split"]:
            wrong_folder.append(row["crop_path"])
        if check_decode:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                corrupt.append(row["crop_path"])
    on_disk = {str(p.relative_to(processed_dir)) for p in (processed_dir / "crops").rglob("*.jpg")}
    untracked = sorted(on_disk - {r["crop_path"] for r in rows})
    counts = Counter((r["split"], r["class_id"]) for r in rows)
    return {
        "total_crops": len(rows),
        "per_split": dict(Counter(r["split"] for r in rows)),
        "per_split_class": {f"{s}/{folder_of[c]}": n for (s, c), n in sorted(counts.items())},
        "missing": missing, "corrupt": corrupt, "wrong_folder": wrong_folder,
        "untracked_files": untracked,
        "ok": not (missing or corrupt or wrong_folder or untracked),
    }


def prepare_dataset(cfg: dict, rebuild: bool = False, workers: int = 4, log=print) -> dict:
    """End-to-end data preparation used by the notebook and scripts/prepare_data.py.

    1. If crops already exist and verify cleanly (and rebuild=False), reuse them.
    2. Otherwise: verify archive checksums, unzip if needed, build crops.
    """
    paths, ds = cfg["paths"], cfg["dataset"]
    processed = paths["processed_dir"]
    if not rebuild and (processed / "manifest.csv").exists():
        report = verify_crop_dataset(processed, check_decode=False)
        if report["ok"]:
            log(f"Reusing existing crops in {processed} ({report['total_crops']} crops).")
            return report
        log("Existing crops failed verification; rebuilding.")

    extracted = paths["extracted_dir"]
    train_json = extracted / ds["train_json"]
    log("Verifying the official files against the Zenodo MD5 checksums...")
    status = verify_archives(paths["raw_dir"], ds["archives"])
    log(f"Archive check: {status}")
    archives_ok = all(v == "ok" for v in status.values())
    if not train_json.exists() and not archives_ok:
        raise FileNotFoundError(
            "Official archives missing or corrupt in "
            f"{paths['raw_dir']}. Download them from https://zenodo.org/records/{ds['zenodo_record']}")
    if not archives_ok:
        log("WARNING: archives could not be verified; using the already extracted files.")
    if not train_json.exists():
        for name in ds["archives"]:
            if name.endswith(".zip"):
                log(f"Extracting {name} ...")
                extract_archive(paths["raw_dir"] / name, extracted)

    if (processed / "crops").exists() and rebuild:
        import shutil
        shutil.rmtree(processed / "crops")
    log("Building crops (one image per annotated egg)...")
    summary = build_crop_dataset(
        train_json=train_json, train_images=extracted / ds["train_images"],
        test_json=paths["raw_dir"] / ds["test_json_in_raw"], test_images=extracted / ds["test_images"],
        out_dir=processed, val_fraction=ds["val_fraction"], margin=ds["crop_margin"],
        seed=cfg["project"]["random_seed"], workers=workers)
    if summary["missing_images"]:
        log(f"WARNING: {len(summary['missing_images'])} annotated images were not found on disk")
    report = verify_crop_dataset(processed)
    count = write_protected_hashes(processed, paths["models_dir"] / "protected_test_hashes.txt",
                                   [extracted / ds["test_images"]])
    log(f"Wrote {count} protected test-image hashes.")
    return report


def write_protected_hashes(processed_dir: Path, out_path: Path, extra_folders=()) -> int:
    """SHA-256 of every held-out test crop (and optionally the original test photos).

    The upload endpoint rejects any file with one of these hashes, so the
    protected test set cannot leak into retraining data."""
    hashes = set()
    test_files = [Path(processed_dir) / r["crop_path"] for r in read_manifest(processed_dir) if r["split"] == "test"]
    for folder in extra_folders:
        if Path(folder).exists():
            test_files += [p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    for path in test_files:
        hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(sorted(hashes)) + "\n")
    return len(hashes)


def list_split(processed_dir: Path, split: str) -> tuple[list[str], list[int]]:
    """File paths and integer labels for one split, taken from the manifest (not folder order)."""
    rows = [r for r in read_manifest(processed_dir) if r["split"] == split]
    return [str(Path(processed_dir) / r["crop_path"]) for r in rows], [r["class_id"] for r in rows]


def image_statistics(image: Image.Image) -> dict:
    """Simple, explainable measurements used for EDA and the dashboard.

    brightness  mean grey level (0-255)
    contrast    standard deviation of grey levels (RMS contrast)
    sharpness   variance of a Laplacian filter response - high for crisp edges,
                low for blurry / out-of-focus images (a texture/focus proxy)
    saturation  mean HSV saturation (0-255) - how coloured vs. grey the crop is
    """
    rgb = image.convert("RGB")
    grey = np.asarray(rgb.convert("L"), dtype=np.float32)
    laplacian = (-4 * grey[1:-1, 1:-1] + grey[:-2, 1:-1] + grey[2:, 1:-1]
                 + grey[1:-1, :-2] + grey[1:-1, 2:])
    red, green, blue = (np.asarray(rgb, dtype=np.float32)[..., i].mean() for i in range(3))
    return {
        "width": rgb.width, "height": rgb.height,
        "aspect_ratio": rgb.width / rgb.height,
        "brightness": float(grey.mean()),
        "contrast": float(grey.std()),
        "sharpness": float(laplacian.var()),
        "saturation": float(np.asarray(rgb.convert("HSV"), dtype=np.float32)[..., 1].mean()),
        "mean_red": float(red), "mean_green": float(green), "mean_blue": float(blue),
    }


def _statistics_of_file(path) -> dict:
    with Image.open(path) as image:
        return image_statistics(image)


def image_statistics_for_files(paths, workers: int = 4) -> list[dict]:
    """image_statistics() for many files, in parallel worker processes."""
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(_statistics_of_file, [str(p) for p in paths], chunksize=64))


# ---------------------------------------------------------------------------
# Part A2 - image validation (used by upload + prediction)
# ---------------------------------------------------------------------------

def load_image_from_bytes(data: bytes) -> Image.Image:
    """Decode bytes into an RGB PIL image; raises ValueError for anything unreadable."""
    if not data:
        raise ValueError("Empty file")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()                      # catches truncated / corrupt files
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except Exception as error:
        raise ValueError(f"Not a readable image: {error.__class__.__name__}") from None


# ---------------------------------------------------------------------------
# Part B - TensorFlow preprocessing (shared by training, evaluation, serving)
# ---------------------------------------------------------------------------

def resize_image(image_uint8, image_size: int):
    """THE resize step. Input: HxWx3 uint8 (tensor or array). Output: float32 in [0, 255]."""
    import tensorflow as tf
    image = tf.image.resize(tf.cast(image_uint8, tf.float32), (image_size, image_size),
                            method="bilinear", antialias=True)
    return tf.clip_by_value(image, 0.0, 255.0)


def preprocess_pil_image(image: Image.Image, image_size: int) -> np.ndarray:
    """Serving path: PIL image -> (1, size, size, 3) float32 batch for model.predict."""
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return resize_image(array, image_size).numpy()[np.newaxis, ...]


def decode_image_bytes(contents):
    """Decode JPEG/PNG/BMP/GIF bytes to uint8 RGB inside tf.data.

    JPEGs are decoded with dct_method="INTEGER_ACCURATE", which reproduces
    Pillow's decoder exactly (verified: 0 pixel difference). TensorFlow's
    default fast decoder differs from Pillow by up to 16 grey levels, which
    would make training pixels differ from what the API (Pillow) sees.
    """
    import tensorflow as tf
    image = tf.cond(
        tf.io.is_jpeg(contents),
        lambda: tf.io.decode_jpeg(contents, channels=3, dct_method="INTEGER_ACCURATE"),
        lambda: tf.io.decode_image(contents, channels=3, expand_animations=False))
    image.set_shape([None, None, 3])
    return image


def _load_file(path, image_size):
    import tensorflow as tf
    return resize_image(decode_image_bytes(tf.io.read_file(path)), image_size)


def build_augmentation(seed: int = 42):
    """Training-only augmentation. Eggs have no canonical orientation under the
    microscope, so flips and rotations do not change the class; brightness and
    contrast changes mimic the different microscopes/cameras in the dataset."""
    import tensorflow as tf
    layers = tf.keras.layers
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=seed),
        layers.RandomRotation(0.5, fill_mode="reflect", seed=seed),
        layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=seed),
        layers.RandomZoom((-0.1, 0.1), fill_mode="reflect", seed=seed),
        layers.RandomBrightness(0.1, value_range=(0, 255), seed=seed),
        layers.RandomContrast(0.1, seed=seed),
    ], name="augmentation")


def make_dataset(file_paths, labels, image_size: int, batch_size: int, training: bool,
                 seed: int = 42, augmentation=None):
    """tf.data pipeline: read -> decode -> resize -> (shuffle + augment if training) -> batch -> prefetch."""
    import tensorflow as tf
    autotune = tf.data.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices((list(file_paths), list(labels)))
    if training:
        dataset = dataset.shuffle(len(file_paths), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda p, y: (_load_file(p, image_size), y), num_parallel_calls=autotune)
    dataset = dataset.batch(batch_size)
    if training:
        augmentation = augmentation or build_augmentation(seed)
        dataset = dataset.map(lambda x, y: (tf.clip_by_value(augmentation(x, training=True), 0, 255), y),
                              num_parallel_calls=autotune)
    return dataset.prefetch(autotune)
