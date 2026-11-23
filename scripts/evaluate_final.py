"""Compare candidate models on VALIDATION data, then evaluate the selected model on the TEST set.

    python scripts/evaluate_final.py                       # compare all initial versions, test the active one
    python scripts/evaluate_final.py --versions v001_baseline_cnn v002_mobilenetv2

Order matters: the model is chosen with validation macro-F1 (src.train.select_best)
*before* this script looks at the test set, and the test set is only run on the
selected (active) model - test results are never used to pick a model.

Writes to artifacts/:
    metrics/model_comparison.csv          validation metrics, size and latency per candidate
    metrics/final_test_metrics.json       test metrics of the active model
    metrics/final_test_classification_report.csv
    plots/final_test_confusion_matrix.png, plots/final_test_predictions.png
    plots/<version>_accuracy_curve.png / _loss_curve.png
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from src import registry  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluation import evaluate_model, measure_inference_time, plot_history  # noqa: E402
from src.preprocessing import list_split  # noqa: E402
from src.train import class_names_from, configure_gpu, update_metadata  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="*", help="versions to compare (default: all non-retrain versions)")
    args = parser.parse_args()
    cfg = load_config()
    configure_gpu()
    models_dir, artifacts = cfg["paths"]["models_dir"], cfg["paths"]["artifacts_dir"]
    metrics_dir, plots_dir = artifacts / "metrics", artifacts / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    t = cfg["training"]
    class_names = class_names_from(cfg)
    processed = cfg["paths"]["processed_dir"]
    versions = args.versions or [v["version"] for v in registry.list_versions(models_dir) if v["kind"] != "retrain"]
    active = registry.get_active_version(models_dir)

    rows = []
    val_files, val_labels = list_split(processed, "val")
    for version in versions:
        folder = registry.version_path(models_dir, version)
        meta = json.loads((folder / "model_metadata.json").read_text())
        model = tf.keras.models.load_model(folder / "model.keras", compile=False)
        history = pd.read_csv(folder / "training_history.csv")
        plot_history(history, plots_dir / f"{version}_accuracy_curve.png", plots_dir / f"{version}_loss_curve.png",
                     title=version)
        val = evaluate_model(model, val_files, val_labels, class_names, t["image_size"], t["batch_size"],
                             metrics_dir, f"{version}_val", plot_examples=False)
        shutil.move(metrics_dir / f"{version}_val_confusion_matrix.png", plots_dir / f"{version}_val_confusion_matrix.png")
        timing = measure_inference_time(model, t["image_size"])
        with tf.device("/CPU:0"):
            cpu_timing = measure_inference_time(model, t["image_size"])
        rows.append({"version": version, "architecture": meta["architecture"],
                     "val_accuracy": val["accuracy"], "val_loss": val["loss"],
                     "val_macro_precision": val["macro_precision"], "val_macro_recall": val["macro_recall"],
                     "val_macro_f1": val["macro_f1"], "val_balanced_accuracy": val["balanced_accuracy"],
                     "parameters": meta["parameters"]["total"], "model_size_mb": meta["model_size_mb"],
                     f"latency_ms_{timing['device'].lower()}": timing["median_ms"],
                     "latency_ms_cpu": cpu_timing["median_ms"],
                     "epochs": meta["epochs_completed"], "training_seconds": meta["training_seconds"],
                     "selected": version == active})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(metrics_dir / "model_comparison.csv", index=False)
    print(comparison.to_string())

    # ---- Final held-out test evaluation: active (selected) model only ----
    if active is None:
        raise SystemExit("No active model. Run `python -m src.train --select` first.")
    folder = registry.version_path(models_dir, active)
    model = tf.keras.models.load_model(folder / "model.keras", compile=False)
    test_files, test_labels = list_split(processed, "test")
    test = evaluate_model(model, test_files, test_labels, class_names, t["image_size"], t["batch_size"],
                          metrics_dir, "final_test")
    for name in ("final_test_confusion_matrix.png", "final_test_predictions.png"):
        shutil.move(metrics_dir / name, plots_dir / name)
    test["model_version"] = active
    (metrics_dir / "final_test_metrics.json").write_text(json.dumps(test, indent=2))
    update_metadata(cfg, active, test_metrics={k: v for k, v in test.items()
                                               if k not in ("per_class", "confusion_matrix", "top_confusions")})
    shutil.copy(folder / "model_metadata.json", models_dir / "model_metadata.json")
    shutil.copy(folder / "training_config.json", models_dir / "training_config.json")
    print(f"\nTEST ({active}): accuracy={test['accuracy']:.4f} macro_f1={test['macro_f1']:.4f} "
          f"macro_precision={test['macro_precision']:.4f} macro_recall={test['macro_recall']:.4f} loss={test['loss']:.4f}")
    print("Most frequent confusions:", test["top_confusions"])


if __name__ == "__main__":
    main()
