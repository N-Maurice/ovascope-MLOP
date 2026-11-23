"""Package results for download and build the small data bundle used by the deployed app.

    python scripts/export_artifacts.py            # results ZIP + deployment bundle
    python scripts/export_artifacts.py --no-bundle

1. artifacts/exports/ovascope_training_results.zip  (only files that really exist are added)
2. data/bundle/  - what the deployed API needs for retraining, without the 13 GB dataset:
     class_map.json, manifest.csv,
     crops/val/...               full validation set (promotion set for retraining)
     crops/train/...             replay sample (retraining.replay_images_per_class per class)
"""
import argparse
import csv
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import registry  # noqa: E402
from src.config import load_config  # noqa: E402
from src.preprocessing import MANIFEST_FIELDS, read_manifest  # noqa: E402


def build_results_zip(cfg) -> Path:
    models_dir, artifacts = cfg["paths"]["models_dir"], cfg["paths"]["artifacts_dir"]
    active = registry.get_active_version(models_dir)
    if active is None:
        raise SystemExit("No active model to export")
    version = registry.version_path(models_dir, active)
    metrics, plots = artifacts / "metrics", artifacts / "plots"
    files = {
        "best_model.h5": version / "model.h5",
        "best_model.keras": version / "model.keras",
        "class_mapping.json": version / "class_mapping.json",
        "training_config.json": version / "training_config.json",
        "model_metadata.json": version / "model_metadata.json",
        "training_history.csv": version / "training_history.csv",
        "model_summary.txt": version / "model_summary.txt",
        "evaluation_metrics.json": metrics / "final_test_metrics.json",
        "classification_report.csv": metrics / "final_test_classification_report.csv",
        "model_comparison.csv": metrics / "model_comparison.csv",
        "confusion_matrix.png": plots / "final_test_confusion_matrix.png",
        "accuracy_curve.png": plots / f"{active}_accuracy_curve.png",
        "loss_curve.png": plots / f"{active}_loss_curve.png",
        "predictions.png": plots / "final_test_predictions.png",
        "dataset_audit.json": metrics / "dataset_audit.json",
    }
    out = artifacts / "exports" / "ovascope_training_results.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            if Path(path).exists():
                archive.write(path, name)
            else:
                print(f"  skipped (not generated): {name}")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def build_bundle(cfg) -> Path:
    processed = cfg["paths"]["processed_dir"]
    bundle = cfg["paths"]["root"] / "data" / "bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    rows = read_manifest(processed)
    rng = random.Random(cfg["project"]["random_seed"])
    keep = [r for r in rows if r["split"] == "val"]
    per_class = cfg["retraining"]["replay_images_per_class"]
    for class_id in sorted({r["class_id"] for r in rows}):
        train_rows = sorted((r for r in rows if r["split"] == "train" and r["class_id"] == class_id),
                            key=lambda r: r["crop_id"])
        keep += rng.sample(train_rows, min(per_class, len(train_rows)))
    for row in keep:
        target = bundle / row["crop_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(processed / row["crop_path"], target)
    with open(bundle / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(keep)
    shutil.copy2(processed / "class_map.json", bundle / "class_map.json")
    size = sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())
    print(f"Wrote {bundle}: {len(keep)} crops, {size / 1e6:.0f} MB")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-bundle", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    if not args.no_zip:
        build_results_zip(cfg)
    if not args.no_bundle:
        build_bundle(cfg)
