import numpy as np
import pytest
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from src.evaluation import compute_metrics, per_class_frame

CLASSES = ["a", "b", "c"]


@pytest.fixture
def predictions():
    rng = np.random.default_rng(0)
    y_true = np.repeat([0, 1, 2], 20)
    logits = rng.normal(size=(60, 3)) + 2.0 * np.eye(3)[y_true]
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return y_true, probabilities


def test_auc_and_log_loss_match_sklearn(predictions):
    y_true, probabilities = predictions
    metrics = compute_metrics(y_true, probabilities, CLASSES)
    assert metrics["roc_auc_macro"] == pytest.approx(
        roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
    one_hot = np.eye(3)[y_true]
    assert metrics["pr_auc_macro"] == pytest.approx(average_precision_score(one_hot, probabilities, average="macro"))
    assert metrics["log_loss"] == pytest.approx(log_loss(y_true, probabilities), rel=1e-5)


def test_per_class_table_has_all_metrics(predictions):
    y_true, probabilities = predictions
    frame = per_class_frame(compute_metrics(y_true, probabilities, CLASSES))
    assert list(frame.index) == CLASSES
    assert list(frame.columns) == ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC",
                                   "Log loss", "Support"]
    assert (frame["Accuracy"] == frame["Recall"]).all()
    assert frame["Support"].sum() == 60


def test_missing_class_gives_undefined_auc_not_crash():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([[.8, .1, .1], [.6, .3, .1], [.2, .7, .1], [.3, .6, .1]])
    metrics = compute_metrics(y_true, probabilities, CLASSES)
    assert metrics["per_class"]["c"]["roc_auc"] is None
    assert metrics["roc_auc_macro"] == pytest.approx(1.0)


def test_epoch_metrics_are_logged_every_epoch(project):
    from src.model import build_baseline_cnn
    from src.preprocessing import list_split, make_dataset
    from src.train import EpochMetrics, class_names_from, fit
    cfg = project["cfg"]
    processed = cfg["paths"]["processed_dir"]
    class_names = class_names_from(cfg)
    train_files, train_labels = list_split(processed, "train")
    val_files, val_labels = list_split(processed, "val")
    tracker = EpochMetrics({"train": (train_files, train_labels), "val": (val_files, val_labels)},
                           class_names, image_size=224, batch_size=8)
    model = build_baseline_cnn(224, len(class_names))
    history, _ = fit(model, make_dataset(train_files, train_labels, 224, 8, training=True),
                     make_dataset(val_files, val_labels, 224, 8, training=False), epochs=2, learning_rate=1e-3,
                     run_dir=project["base"] / "epoch_metrics_run", cfg=cfg, epoch_metrics=tracker)
    for split in ("train", "val"):
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "log_loss"):
            assert f"{split}_{metric}" in history, f"{split}_{metric} missing"
            assert history[f"{split}_{metric}"].notna().all()
    assert len(history) == 2
    csv_log = (project["base"] / "epoch_metrics_run" / "history_train.csv").read_text()
    assert "val_roc_auc" in csv_log.splitlines()[0]
