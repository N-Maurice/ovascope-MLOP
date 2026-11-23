"""Evaluation metrics, reports and plots.

Metrics (multiclass): accuracy, loss (cross-entropy), macro and weighted
precision / recall / F1, balanced accuracy, per-class report, confusion matrix.
Macro F1 is the main comparison metric: it weights every parasite class equally.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

# No matplotlib.use("Agg") here: forcing a backend on import would stop notebooks from showing
# figures. Without a display, matplotlib already falls back to Agg automatically.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, precision_recall_fscore_support)


def predict_dataset(model, dataset):
    """Run the model over a (non-shuffled) tf.data dataset. Returns (y_true, probabilities)."""
    y_true, probabilities = [], []
    for images, labels in dataset:
        probabilities.append(model(images, training=False).numpy())
        y_true.append(labels.numpy())
    return np.concatenate(y_true), np.concatenate(probabilities)


def compute_metrics(y_true, probabilities, class_names: list[str]) -> dict:
    y_pred = probabilities.argmax(axis=1)
    eps = 1e-7
    loss = float(-np.mean(np.log(np.clip(probabilities[np.arange(len(y_true)), y_true], eps, 1.0))))
    labels = list(range(len(class_names)))
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "loss": loss,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]), "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "per_class": {name: {"precision": float(per_class[0][i]), "recall": float(per_class[1][i]),
                             "f1": float(per_class[2][i]), "support": int(per_class[3][i])}
                      for i, name in enumerate(class_names)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def classification_report_frame(y_true, probabilities, class_names) -> pd.DataFrame:
    report = classification_report(y_true, probabilities.argmax(axis=1), labels=list(range(len(class_names))),
                                   target_names=class_names, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()


def top_confusions(metrics: dict, class_names: list[str], n: int = 5) -> list[dict]:
    matrix = np.array(metrics["confusion_matrix"])
    np.fill_diagonal(matrix, 0)
    pairs = [(matrix[i, j], i, j) for i in range(len(matrix)) for j in range(len(matrix)) if matrix[i, j] > 0]
    return [{"true": class_names[i], "predicted": class_names[j], "count": int(c)}
            for c, i, j in sorted(pairs, reverse=True)[:n]]


def plot_confusion_matrix(matrix, class_names, path: Path, title: str = "Confusion matrix"):
    matrix = np.array(matrix)
    normalised = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if matrix[i, j]:
                ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=8,
                        color="white" if normalised[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{title} (counts; colour = row-normalised recall)")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_history(history: pd.DataFrame, path_accuracy: Path, path_loss: Path, title: str = ""):
    """Training vs validation curves. `history` needs columns accuracy, val_accuracy, loss, val_loss
    and optionally `phase` (e.g. head / fine_tune) to mark where fine-tuning starts."""
    epochs = np.arange(1, len(history) + 1)
    for metric, path in (("accuracy", path_accuracy), ("loss", path_loss)):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(epochs, history[metric], marker="o", label=f"train {metric}")
        ax.plot(epochs, history[f"val_{metric}"], marker="o", label=f"validation {metric}")
        if "phase" in history and history["phase"].nunique() > 1:
            start = int(np.argmax(history["phase"].values != history["phase"].values[0])) + 1
            ax.axvline(start - 0.5, color="grey", linestyle="--", label="fine-tuning starts")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"{title} {metric}".strip())
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)


def plot_prediction_examples(file_paths, y_true, probabilities, class_names, path: Path, per_group: int = 6,
                             seed: int = 0):
    """Three rows: correct, incorrect, lowest-confidence predictions (real examples, not curated)."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    y_pred = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = np.nonzero(y_pred == y_true)[0]
    wrong = np.nonzero(y_pred != y_true)[0]
    groups = [
        ("Correct", rng.choice(correct, min(per_group, len(correct)), replace=False) if len(correct) else []),
        ("Incorrect", rng.choice(wrong, min(per_group, len(wrong)), replace=False) if len(wrong) else []),
        ("Lowest confidence", np.argsort(confidence)[:per_group]),
    ]
    fig, axes = plt.subplots(3, per_group, figsize=(2.6 * per_group, 9.2))
    for row, (label, indices) in enumerate(groups):
        for col in range(per_group):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(indices):
                continue
            i = indices[col]
            ax.imshow(Image.open(file_paths[i]).convert("RGB"))
            ok = y_pred[i] == y_true[i]
            ax.set_title(f"true: {class_names[y_true[i]][:20]}\npred: {class_names[y_pred[i]][:20]}\n"
                         f"conf: {confidence[i]:.2f}", fontsize=7.5, color="green" if ok else "red")
        axes[row, 0].text(-0.15, 0.5, label, transform=axes[row, 0].transAxes, rotation=90,
                          va="center", ha="right", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def measure_inference_time(model, image_size: int, runs: int = 50) -> dict:
    """Median single-image latency on the current device, using a compiled forward pass
    (tf.function) exactly as src.prediction.Predictor does in the API."""
    import tensorflow as tf
    batch = tf.random.uniform((1, image_size, image_size, 3), 0, 255)
    forward = tf.function(lambda x: model(x, training=False))
    for _ in range(5):
        forward(batch).numpy()
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        forward(batch).numpy()
        timings.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
            "runs": runs, "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU"}


def save_evaluation(metrics: dict, report: pd.DataFrame, class_names, out_dir: Path, prefix: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))
    report.to_csv(out_dir / f"{prefix}_classification_report.csv")


def evaluate_model(model, file_paths, labels, class_names, image_size, batch_size, out_dir: Path,
                   prefix: str, plot_examples: bool = True) -> dict:
    """Full evaluation of one model on one split: metrics JSON, report CSV, confusion matrix, examples."""
    from src.preprocessing import make_dataset
    dataset = make_dataset(file_paths, labels, image_size, batch_size, training=False)
    y_true, probabilities = predict_dataset(model, dataset)
    metrics = compute_metrics(y_true, probabilities, class_names)
    metrics["top_confusions"] = top_confusions(metrics, class_names)
    report = classification_report_frame(y_true, probabilities, class_names)
    save_evaluation(metrics, report, class_names, out_dir, prefix)
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, Path(out_dir) / f"{prefix}_confusion_matrix.png",
                          title=prefix.replace("_", " "))
    if plot_examples:
        plot_prediction_examples(file_paths, y_true, probabilities, class_names,
                                 Path(out_dir) / f"{prefix}_predictions.png")
    np.save(Path(out_dir) / f"{prefix}_probabilities.npy", probabilities)
    return metrics
