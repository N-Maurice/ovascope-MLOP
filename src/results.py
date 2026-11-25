"""Read-only access to saved evaluation results, training histories and dataset statistics.

The web interface shows these files; nothing here trains or evaluates a model. Every loader returns
None when its source file is missing (e.g. in a fresh deployment) so the API can answer 404.
"""
from __future__ import annotations

import csv
import json
import math
from functools import lru_cache
from pathlib import Path

from src import registry
from src.gate import GATE_FOLDER, GATE_METADATA_FILE

AUDIT_KEYS = ("crops_total", "crops_per_split", "crops_per_split_class", "source_image_sizes", "crop_side_px",
              "categories_identical_train_test", "class_map_matches_annotations", "source_images_in_multiple_splits",
              "exact_duplicates_across_splits", "exact_duplicate_crops_across_splits", "near_duplicate_pairs_across_splits",
              "near_duplicate_threshold_bits", "test_to_trainval_closest_distance", "non_square_crops",
              "empty_or_flat_crops")


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _number(value: str):
    """CSV cell -> int/float/None/str (empty cells and NaN become None)."""
    if value in ("", None):
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    if math.isnan(number):
        return None
    return int(number) if number.is_integer() and "." not in value else number


def read_csv_rows(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(f)]


def evaluation(artifacts_dir: Path) -> dict | None:
    """Final test evaluation + the validation evaluation of the same model + model comparison."""
    metrics = Path(artifacts_dir) / "metrics"
    test = _json(metrics / "final_test_metrics.json")
    if test is None:
        return None
    version = test.get("model_version")
    resolution = read_csv_rows(metrics / "accuracy_by_resolution.csv") or []
    return {
        "model_version": version,
        "test": test,
        "validation": _json(metrics / f"{version}_val_metrics.json") if version else None,
        "comparison": read_csv_rows(metrics / "model_comparison.csv") or [],
        "by_resolution": [row for row in resolution if row.get("val_crops") is not None],
    }


def training_history(models_dir: Path, version: str) -> list[dict] | None:
    try:
        folder = registry.version_path(models_dir, version)
    except FileNotFoundError:
        return None
    return read_csv_rows(folder / "training_history.csv")


def dataset_summary(artifacts_dir: Path, class_map: list[dict]) -> dict | None:
    audit = _json(Path(artifacts_dir) / "metrics" / "dataset_audit.json")
    if audit is None:
        return None
    summary = {key: audit.get(key) for key in AUDIT_KEYS}
    summary["source_photos"] = {"train_and_val": (audit.get("train_source") or {}).get("images_in_json"),
                                "test": (audit.get("test_source") or {}).get("images_in_json")}
    summary["classes"] = class_map
    return summary


def gate_report(models_dir: Path, artifacts_dir: Path) -> dict:
    meta = _json(Path(models_dir) / GATE_FOLDER / GATE_METADATA_FILE)
    if meta is None:
        return {"enabled": False}
    keep = ("created_at", "architecture", "threshold", "target_egg_recall", "negative_sources", "training_images",
            "epochs_completed", "training_seconds", "validation", "test", "history", "model_size_mb")
    report = {"enabled": True, **{key: meta.get(key) for key in keep}}
    report["classifier_on_non_eggs"] = read_csv_rows(Path(artifacts_dir) / "metrics" / "classifier_on_non_eggs.csv") or []
    return report


@lru_cache(maxsize=4)
def _example_paths(processed_dir: str, manifest_mtime: float) -> dict[int, list[str]]:
    """Validation crops per class (test crops are never shown: they stay protected)."""
    folder = Path(processed_dir)
    examples: dict[int, list[str]] = {}
    with open(folder / "manifest.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "val" and (folder / row["crop_path"]).exists():
                examples.setdefault(int(row["class_id"]), []).append(row["crop_path"])
    return examples


def example_image(processed_dir: Path, class_id: int, index: int) -> Path | None:
    manifest = Path(processed_dir) / "manifest.csv"
    if not manifest.exists():
        return None
    paths = _example_paths(str(processed_dir), manifest.stat().st_mtime).get(class_id)
    if not paths:
        return None
    step = max(1, len(paths) // 12)   # spread consecutive indices across the class for variety
    return Path(processed_dir) / paths[(index * step) % len(paths)]
