"""Train OvaScope models and save them as versioned artifacts.

    python -m src.train --model baseline          # Stage 1: custom CNN
    python -m src.train --model transfer          # Stages 2-4: MobileNetV2 head + fine-tuning
    python -m src.train --model both --select     # train both, activate the better one (validation macro-F1)

Optimisation techniques used (all configured in configs/config.yaml):
    Adam optimiser, dropout, data augmentation, early stopping with best-weight
    restoration, ReduceLROnPlateau, best-checkpoint saving, transfer learning,
    and fine-tuning with a 100x smaller learning rate.
The held-out test set is never touched here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src import registry
from src.config import load_config
from src.evaluation import compute_metrics, predict_dataset
from src.model import build_baseline_cnn, build_transfer_model, count_parameters, unfreeze_top_of_backbone
from src.preprocessing import list_split, load_class_map, make_dataset


# ---------------------------------------------------------------------------
# Environment and data
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow. Some GPU kernels (e.g. cuDNN convolution
    backprop) remain non-deterministic, so repeated GPU runs can differ slightly."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def configure_gpu() -> list[str]:
    """Let TensorFlow grow GPU memory as needed instead of grabbing all of it."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # already initialised
    return [g.name for g in gpus]


def describe_environment() -> dict:
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = []
    for gpu in gpus:
        try:
            gpu_names.append(tf.config.experimental.get_device_details(gpu).get("device_name", gpu.name))
        except Exception:
            gpu_names.append(gpu.name)
    import PIL
    import sklearn
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "tensorflow": tf.__version__, "keras": tf.keras.__version__ if hasattr(tf.keras, "__version__") else "",
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "pillow": PIL.__version__, "cpu_count": os.cpu_count(),
        "accelerator": "GPU" if gpus else "CPU", "gpus": gpu_names,
    }


def class_names_from(cfg: dict) -> list[str]:
    return [c["display_name"] for c in load_class_map(cfg["paths"]["processed_dir"])]


def load_train_val(cfg: dict):
    """tf.data datasets for training (augmented, shuffled) and validation (plain)."""
    t = cfg["training"]
    processed = cfg["paths"]["processed_dir"]
    train_files, train_labels = list_split(processed, "train")
    val_files, val_labels = list_split(processed, "val")
    if not train_files or not val_files:
        raise FileNotFoundError(f"No training/validation crops found in {processed}. Run scripts/prepare_data.py")
    seed = cfg["project"]["random_seed"]
    train_ds = make_dataset(train_files, train_labels, t["image_size"], t["batch_size"], training=True, seed=seed)
    val_ds = make_dataset(val_files, val_labels, t["image_size"], t["batch_size"], training=False)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class EpochMetrics(tf.keras.callbacks.Callback):
    """At the end of every epoch, measure the seven headline metrics on the full training set
    (without augmentation) and on the validation set, and add them to the epoch logs.

    Columns added: train_/val_ + accuracy, precision, recall, f1, roc_auc, pr_auc, log_loss
    (precision/recall/F1 are macro averages; ROC-AUC and PR-AUC are one-vs-rest macro averages).
    They are written to the CSV log and the training history together with Keras' own
    `accuracy`/`loss` (running averages over the *augmented* batches) and `val_accuracy`/`val_loss`.
    Must be placed before CSVLogger/EarlyStopping in the callback list so they see the new values.
    """

    KEYS = {"accuracy": "accuracy", "precision": "macro_precision", "recall": "macro_recall", "f1": "macro_f1",
            "roc_auc": "roc_auc_macro", "pr_auc": "pr_auc_macro", "log_loss": "log_loss"}

    def __init__(self, splits: dict, class_names: list[str], image_size: int, batch_size: int):
        """splits: {"train": (file_paths, labels), "val": (file_paths, labels)}"""
        super().__init__()
        self.datasets = {name: (make_dataset(files, labels, image_size, batch_size, training=False)
                                .map(lambda x, y: x), np.asarray(labels))
                         for name, (files, labels) in splits.items()}
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        parts = []
        for name, (images, labels) in self.datasets.items():
            probabilities = self.model.predict(images, verbose=0)
            metrics = compute_metrics(labels, probabilities, self.class_names)
            for short, key in self.KEYS.items():
                logs[f"{name}_{short}"] = float("nan") if metrics[key] is None else metrics[key]
            parts.append(f"{name}: f1={metrics['macro_f1']:.4f} roc_auc={metrics['roc_auc_macro'] or float('nan'):.4f} "
                         f"pr_auc={metrics['pr_auc_macro'] or float('nan'):.4f} log_loss={metrics['log_loss']:.4f}")
        print(f"  epoch {epoch + 1} metrics | " + " | ".join(parts))


def compile_model(model, learning_rate: float) -> None:
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])


def fit(model, train_ds, val_ds, epochs: int, learning_rate: float, run_dir: Path, cfg: dict,
        phase: str = "train", epoch_metrics: EpochMetrics | None = None) -> tuple[pd.DataFrame, float]:
    """Compile + fit with the standard callbacks. Returns (history, seconds).

    * EpochMetrics (optional) adds accuracy, precision, recall, F1, ROC-AUC, PR-AUC and log loss
      on the training and validation sets to every epoch's history.
    * ModelCheckpoint keeps the epoch with the lowest validation loss.
    * EarlyStopping stops after `patience` epochs without improvement and restores the best weights.
    * ReduceLROnPlateau halves the learning rate when validation loss stalls.
    * BackupAndRestore makes an interrupted run resumable from its last epoch.
    """
    t = cfg["training"]
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    compile_model(model, learning_rate)
    callbacks = [epoch_metrics] if epoch_metrics is not None else []
    callbacks += [
        tf.keras.callbacks.ModelCheckpoint(str(run_dir / f"best_{phase}.keras"), monitor="val_loss",
                                           save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=t["early_stopping_patience"],
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=t["reduce_lr_patience"],
                                             min_lr=1e-7),
        tf.keras.callbacks.CSVLogger(str(run_dir / f"history_{phase}.csv")),
        tf.keras.callbacks.BackupAndRestore(str(run_dir / f"backup_{phase}")),
    ]
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)
    seconds = time.time() - start
    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
    if "train_log_loss" in frame:   # put the per-epoch metric columns in a readable order
        ordered = ["epoch", "loss", "accuracy"] + [f"{s}_{m}" for s in ("train", "val") for m in EpochMetrics.KEYS]
        ordered = [c for c in ordered if c in frame] + [c for c in frame if c not in ordered]
        frame = frame[ordered]
    frame["phase"] = phase
    if "learning_rate" not in frame:
        frame["learning_rate"] = float(model.optimizer.learning_rate.numpy())
    return frame, seconds


def validation_metrics(model, val_ds, class_names) -> dict:
    y_true, probabilities = predict_dataset(model, val_ds)
    return compute_metrics(y_true, probabilities, class_names)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_and_verify_h5(model, keras_path: Path, h5_path: Path, image_size: int) -> dict:
    """Save a legacy .h5 copy, reload both files, and confirm they predict the same thing."""
    model.save(h5_path)
    sample = np.random.default_rng(0).uniform(0, 255, (4, image_size, image_size, 3)).astype("float32")
    from_keras = tf.keras.models.load_model(keras_path, compile=False).predict(sample, verbose=0)
    from_h5 = tf.keras.models.load_model(h5_path, compile=False).predict(sample, verbose=0)
    difference = float(np.abs(from_keras - from_h5).max())
    if difference > 1e-4:
        raise RuntimeError(f".h5 export differs from .keras model (max abs diff {difference})")
    return {"h5_reload_ok": True, "max_abs_difference_vs_keras": difference}


def save_version(model, kind: str, cfg: dict, history: pd.DataFrame, val_metrics: dict, extra: dict,
                 parent_version: str | None = None) -> str:
    """Write a new models/versions/<version>/ folder and return the version name."""
    models_dir = cfg["paths"]["models_dir"]
    version = registry.next_version_name(models_dir, kind)
    out = registry.versions_dir(models_dir) / version
    out.mkdir(parents=True)
    image_size = cfg["training"]["image_size"]

    keras_path, h5_path = out / "model.keras", out / "model.h5"
    model.save(keras_path)
    h5_check = export_and_verify_h5(model, keras_path, h5_path, image_size)
    class_map = load_class_map(cfg["paths"]["processed_dir"])
    (out / "class_mapping.json").write_text(json.dumps(class_map, indent=2))
    history.to_csv(out / "training_history.csv", index=False)
    training_config = {"training": cfg["training"], "retraining": cfg["retraining"],
                       "dataset": {k: v for k, v in cfg["dataset"].items() if k != "archives"},
                       "random_seed": cfg["project"]["random_seed"]}
    (out / "training_config.json").write_text(json.dumps(training_config, indent=2))
    summary_lines = []
    model.summary(print_fn=lambda line, *a, **k: summary_lines.append(line), expand_nested=False)
    (out / "model_summary.txt").write_text("\n".join(summary_lines))

    best_epoch = int(history["val_loss"].idxmin()) + 1
    metadata = {
        "version": version, "kind": kind, "architecture": model.name,
        "parent_version": parent_version,
        "input_shape": [image_size, image_size, 3], "input_range": "[0, 255] RGB (model rescales internally)",
        "num_classes": len(class_map),
        "created_at": registry.utc_now(),
        "epochs_completed": int(len(history)), "best_epoch": best_epoch,
        "final_learning_rate": float(history["learning_rate"].iloc[-1]),
        "final_train_loss": float(history["loss"].iloc[-1]), "final_train_accuracy": float(history["accuracy"].iloc[-1]),
        "best_val_loss": float(history["val_loss"].min()),
        "validation_metrics": {k: v for k, v in val_metrics.items() if k not in ("confusion_matrix", "per_class")},
        "parameters": count_parameters(model),
        "files": {"keras": "model.keras", "h5": "model.h5"},
        "model_size_mb": round(keras_path.stat().st_size / 1e6, 2),
        "sha256": {"model.keras": sha256_of(keras_path), "model.h5": sha256_of(h5_path)},
        "h5_export": h5_check,
        "environment": describe_environment(),
        **extra,
    }
    (out / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    return version


def update_metadata(cfg: dict, version: str, **fields) -> dict:
    path = registry.versions_dir(cfg["paths"]["models_dir"]) / version / "model_metadata.json"
    metadata = json.loads(path.read_text())
    metadata.update(fields)
    path.write_text(json.dumps(metadata, indent=2))
    return metadata


# ---------------------------------------------------------------------------
# Training recipes
# ---------------------------------------------------------------------------

def make_epoch_metrics(cfg: dict, class_names) -> EpochMetrics | None:
    """The per-epoch metrics callback (full training set without augmentation + validation set),
    unless disabled with training.epoch_metrics: false."""
    t = cfg["training"]
    if not t.get("epoch_metrics", True):
        return None
    processed = cfg["paths"]["processed_dir"]
    splits = {"train": list_split(processed, "train"), "val": list_split(processed, "val")}
    return EpochMetrics(splits, class_names, t["image_size"], t["batch_size"])


def train_baseline(cfg: dict, epochs: int | None = None) -> str:
    t = cfg["training"]
    set_seeds(cfg["project"]["random_seed"])
    class_names = class_names_from(cfg)
    train_ds, val_ds = load_train_val(cfg)
    tracker = make_epoch_metrics(cfg, class_names)
    model = build_baseline_cnn(t["image_size"], len(class_names))
    run_dir = cfg["paths"]["models_dir"] / "checkpoints" / "baseline_cnn"
    history, seconds = fit(model, train_ds, val_ds, epochs or t["baseline"]["epochs"],
                           t["baseline"]["learning_rate"], run_dir, cfg, phase="baseline", epoch_metrics=tracker)
    val_metrics = validation_metrics(model, val_ds, class_names)
    print(f"Baseline CNN validation: accuracy={val_metrics['accuracy']:.4f} macro_f1={val_metrics['macro_f1']:.4f}")
    return save_version(model, "baseline_cnn", cfg, history, val_metrics,
                        {"training_seconds": round(seconds, 1), "pretrained_weights": None})


def train_transfer(cfg: dict, head_epochs: int | None = None, fine_tune_epochs: int | None = None,
                   weights: str | None = "imagenet") -> str:
    """Stage 2+3: train the head on a frozen backbone. Stage 4: fine-tune the upper backbone,
    keep whichever checkpoint has the better validation macro-F1."""
    t, tt = cfg["training"], cfg["training"]["transfer"]
    set_seeds(cfg["project"]["random_seed"])
    class_names = class_names_from(cfg)
    train_ds, val_ds = load_train_val(cfg)
    tracker = make_epoch_metrics(cfg, class_names)
    model = build_transfer_model(t["image_size"], len(class_names), dropout=tt["dropout"], weights=weights)
    run_dir = cfg["paths"]["models_dir"] / "checkpoints" / "mobilenetv2"

    head_history, head_seconds = fit(model, train_ds, val_ds, head_epochs or tt["head_epochs"],
                                     tt["head_learning_rate"], run_dir, cfg, phase="head",
                                     epoch_metrics=tracker)
    head_metrics = validation_metrics(model, val_ds, class_names)
    head_weights = model.get_weights()
    print(f"Head-only validation: accuracy={head_metrics['accuracy']:.4f} macro_f1={head_metrics['macro_f1']:.4f}")

    fine_tune_info = {"performed": False}
    history, seconds, final_metrics = head_history, head_seconds, head_metrics
    epochs = fine_tune_epochs if fine_tune_epochs is not None else tt["fine_tune_epochs"]
    if epochs > 0:
        trainable_layers = unfreeze_top_of_backbone(model, tt["fine_tune_from_layer"])
        ft_history, ft_seconds = fit(model, train_ds, val_ds, epochs, tt["fine_tune_learning_rate"],
                                     run_dir, cfg, phase="fine_tune", epoch_metrics=tracker)
        ft_metrics = validation_metrics(model, val_ds, class_names)
        print(f"Fine-tuned validation: accuracy={ft_metrics['accuracy']:.4f} macro_f1={ft_metrics['macro_f1']:.4f}")
        kept = "fine_tune" if ft_metrics["macro_f1"] >= head_metrics["macro_f1"] else "head"
        fine_tune_info = {
            "performed": True, "unfrozen_from_layer": tt["fine_tune_from_layer"],
            "trainable_backbone_layers": trainable_layers,
            "head_val_macro_f1": head_metrics["macro_f1"], "fine_tuned_val_macro_f1": ft_metrics["macro_f1"],
            "head_val_accuracy": head_metrics["accuracy"], "fine_tuned_val_accuracy": ft_metrics["accuracy"],
            "kept": kept,
        }
        history = pd.concat([head_history, ft_history], ignore_index=True)
        history["epoch"] = np.arange(1, len(history) + 1)
        seconds = head_seconds + ft_seconds
        if kept == "head":
            model.set_weights(head_weights)
            final_metrics = head_metrics
        else:
            final_metrics = ft_metrics
    return save_version(model, "mobilenetv2", cfg, history, final_metrics,
                        {"training_seconds": round(seconds, 1), "pretrained_weights": weights or "none",
                         "fine_tuning": fine_tune_info})


def select_best(cfg: dict, versions: list[str]) -> str:
    """Pick the version with the highest *validation* macro-F1 (never the test set)."""
    scores = {}
    for version in versions:
        meta = json.loads((registry.versions_dir(cfg["paths"]["models_dir"]) / version / "model_metadata.json")
                          .read_text())
        scores[version] = meta["validation_metrics"]["macro_f1"]
    best = max(scores, key=scores.get)
    print(f"Validation macro-F1: {scores} -> selected {best}")
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train OvaScope models")
    parser.add_argument("--model", choices=["baseline", "transfer", "both"], default="both")
    parser.add_argument("--epochs", type=int, help="override epochs (baseline) / head epochs (transfer)")
    parser.add_argument("--fine-tune-epochs", type=int)
    parser.add_argument("--select", action="store_true", help="activate the best new version (validation macro-F1)")
    args = parser.parse_args(argv)

    cfg = load_config()
    configure_gpu()
    print(json.dumps(describe_environment(), indent=2))
    versions = []
    if args.model in ("baseline", "both"):
        versions.append(train_baseline(cfg, args.epochs))
    if args.model in ("transfer", "both"):
        versions.append(train_transfer(cfg, args.epochs, args.fine_tune_epochs))
    print("Saved versions:", versions)
    if args.select:
        best = select_best(cfg, versions)
        print("Activated:", registry.activate_version(cfg["paths"]["models_dir"], best))


if __name__ == "__main__":
    main()
