"""Retrain the *existing* OvaScope model with newly uploaded, labelled images.

Workflow (one job):
    1. validated uploads are read back from the database (image bytes + labels)
    2. they are preprocessed with the same pipeline as training (src.preprocessing)
    3. the currently ACTIVE saved model is loaded - never a fresh random model
    4. training continues for a few epochs at a small learning rate on
       new images (repeated) + a replay sample of historical training crops,
       which limits catastrophic forgetting of the original classes
    5. candidate and active model are both evaluated on the validation set
       (the "promotion set"; it is never used to fit weights)
    6. the candidate is saved as a new version; it is activated only if
       macro-F1(candidate) >= macro-F1(active) - promotion_tolerance
    7. everything is recorded in the retrain_jobs table

The held-out test set is never used. Jobs run in a separate process
(`python -m src.retrain --job-id ...`) so the API stays responsive.
"""
from __future__ import annotations

import argparse
import os
import socket
import random
import shutil
import subprocess
import sys
import threading
import traceback
from pathlib import Path

from src import database, registry
from src.config import load_config


class RetrainError(Exception):
    """A user-facing reason why retraining cannot start."""


def retraining_status(cfg: dict) -> dict:
    """Readiness summary used by the API/UI, including the automatic trigger flag."""
    db_path = cfg["paths"]["database"]
    counts = database.upload_counts(db_path)
    threshold = cfg["retraining"]["trigger_threshold"]
    jobs = database.list_jobs(db_path, limit=1)
    return {
        "pending_new_images": counts["pending"],
        "trigger_threshold": threshold,
        "retraining_recommended": counts["pending"] >= threshold,
        "min_new_images": cfg["retraining"]["min_new_images"],
        "active_model": registry.get_active_version(cfg["paths"]["models_dir"]),
        "job_running": bool(jobs and jobs[0]["status"] in ("queued", "running")),
        "latest_job": jobs[0] if jobs else None,
    }


def start_retraining(cfg: dict, trigger: str = "manual", run_in_process: bool = False,
                     requested_by: str | None = None) -> str:
    """Check preconditions, create a job record and launch the worker. Returns the job id."""
    db_path, models_dir = cfg["paths"]["database"], cfg["paths"]["models_dir"]
    pending = database.upload_counts(db_path)["pending"]
    if pending < cfg["retraining"]["min_new_images"]:
        raise RetrainError(f"Need at least {cfg['retraining']['min_new_images']} new validated images "
                           f"(currently {pending}).")
    active = registry.get_active_version(models_dir)
    if active is None:
        raise RetrainError("No trained model is active; run initial training first.")
    check_class_mapping(cfg, active)
    job_id = database.create_job_if_idle(db_path, trigger=trigger, requested_by=requested_by)
    if job_id is None:
        raise RetrainError("A retraining job is already queued or running.")
    if run_in_process:
        run_job(cfg, job_id)
    else:
        log_dir = cfg["paths"]["artifacts_dir"] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / f"retrain_{job_id}.log", "w") as log:
            worker = subprocess.Popen([sys.executable, "-m", "src.retrain", "--job-id", job_id],
                                      cwd=cfg["paths"]["root"], stdout=log, stderr=subprocess.STDOUT,
                                      env={**os.environ}, start_new_session=True)
        database.update_job(db_path, job_id, worker=f"{socket.gethostname()}:{worker.pid}")
    return job_id


def check_class_mapping(cfg: dict, version: str) -> None:
    from src.preprocessing import load_class_map
    model_map = load_class_map(registry.version_path(cfg["paths"]["models_dir"], version) / "class_mapping.json")
    data_map = load_class_map(cfg["paths"]["processed_dir"])
    if [(c["class_id"], c["original_name"]) for c in model_map] != [(c["class_id"], c["original_name"]) for c in data_map]:
        raise RetrainError(f"Class mapping of model {version} does not match the dataset class map.")


def export_uploads(uploads: list[dict], folder: Path) -> tuple[list[str], list[int]]:
    """Write database images to PNG files so the standard tf.data pipeline can read them."""
    from src.preprocessing import load_image_from_bytes
    paths, labels = [], []
    folder.mkdir(parents=True, exist_ok=True)
    for upload in uploads:
        image = load_image_from_bytes(bytes(upload["image_data"]))
        path = folder / f"{upload['id']:06d}.png"
        image.save(path)
        paths.append(str(path))
        labels.append(int(upload["class_id"]))
    return paths, labels


def replay_sample(processed_dir: Path, per_class: int, seed: int) -> tuple[list[str], list[int]]:
    from src.preprocessing import list_split
    files, labels = list_split(processed_dir, "train")
    by_class: dict[int, list[str]] = {}
    for f, y in zip(files, labels):
        by_class.setdefault(y, []).append(f)
    rng = random.Random(seed)
    out_files, out_labels = [], []
    for y in sorted(by_class):
        chosen = rng.sample(by_class[y], min(per_class, len(by_class[y])))
        out_files += chosen
        out_labels += [y] * len(chosen)
    return out_files, out_labels


def run_job(cfg: dict, job_id: str) -> dict:
    """Execute one retraining job end to end (called inside the worker process)."""
    import tensorflow as tf

    from src.evaluation import compute_metrics, predict_dataset
    from src.preprocessing import list_split, make_dataset
    from src.train import class_names_from, configure_gpu, save_version, set_seeds

    db_path, models_dir = cfg["paths"]["database"], cfg["paths"]["models_dir"]
    r, t = cfg["retraining"], cfg["training"]
    work_dir = cfg["paths"]["artifacts_dir"] / "logs" / f"retrain_{job_id}_data"
    database.update_job(db_path, job_id, status="running", started_at=registry.utc_now(), progress="loading data",
                        worker=f"{socket.gethostname()}:{os.getpid()}")
    beating = threading.Event()

    def beat():   # sign of life for the API; stops when the job ends or the process dies
        while not beating.wait(30):
            database.heartbeat(db_path, job_id)

    database.heartbeat(db_path, job_id)
    threading.Thread(target=beat, daemon=True).start()
    try:
        configure_gpu()
        set_seeds(cfg["project"]["random_seed"])
        base_version = registry.get_active_version(models_dir)
        check_class_mapping(cfg, base_version)
        uploads = database.pending_uploads(db_path)
        if len(uploads) < r["min_new_images"]:
            raise RetrainError(f"only {len(uploads)} pending images")
        new_files, new_labels = export_uploads(uploads, work_dir / "new")
        old_files, old_labels = replay_sample(cfg["paths"]["processed_dir"], r["replay_images_per_class"],
                                              cfg["project"]["random_seed"])
        train_files = new_files * r["new_image_repeat"] + old_files
        train_labels = new_labels * r["new_image_repeat"] + old_labels
        val_files, val_labels = list_split(cfg["paths"]["processed_dir"], "val")
        database.update_job(db_path, job_id, base_version=base_version, n_new_images=len(uploads),
                            progress=f"{len(uploads)} new + {len(old_files)} replay images; loading model {base_version}")

        train_ds = make_dataset(train_files, train_labels, t["image_size"], t["batch_size"], training=True,
                                seed=cfg["project"]["random_seed"])
        val_ds = make_dataset(val_files, val_labels, t["image_size"], t["batch_size"], training=False)

        # 3. Load the existing, previously trained OvaScope model (not a new random one).
        model_path = registry.version_path(models_dir, base_version) / "model.keras"
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer=tf.keras.optimizers.Adam(r["learning_rate"]),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

        class Progress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                database.update_job(db_path, job_id, progress=(
                    f"epoch {epoch + 1}/{r['epochs']}: loss={logs['loss']:.4f} acc={logs['accuracy']:.4f} "
                    f"val_loss={logs['val_loss']:.4f} val_acc={logs['val_accuracy']:.4f}"))

        # Validation loss is only *reported* each epoch; no callback uses it to steer training.
        history = model.fit(train_ds, validation_data=val_ds, epochs=r["epochs"], callbacks=[Progress()], verbose=2)
        import numpy as np
        import pandas as pd
        frame = pd.DataFrame(history.history)
        frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
        frame["phase"] = "retrain"
        frame["learning_rate"] = r["learning_rate"]

        # 5. Evaluate candidate and currently active model on the same promotion set.
        database.update_job(db_path, job_id, progress="evaluating candidate vs active model")
        class_names = class_names_from(cfg)
        y_true, candidate_probs = predict_dataset(model, val_ds)
        candidate_metrics = compute_metrics(y_true, candidate_probs, class_names)
        active_model = tf.keras.models.load_model(model_path, compile=False)
        _, active_probs = predict_dataset(active_model, val_ds)
        active_metrics = compute_metrics(y_true, active_probs, class_names)

        candidate_version = save_version(
            model, "retrain", cfg, frame, candidate_metrics, parent_version=base_version,
            extra={"retrain_job_id": job_id, "new_images": len(uploads), "replay_images": len(old_files),
                   "pretrained_weights": f"continued from {base_version}"})

        # 6. Promotion decision.
        margin = candidate_metrics["macro_f1"] - active_metrics["macro_f1"]
        promote = margin >= -r["promotion_tolerance"]
        decision = (f"{'PROMOTED' if promote else 'REJECTED'}: candidate macro-F1 {candidate_metrics['macro_f1']:.4f} "
                    f"vs active {active_metrics['macro_f1']:.4f} (difference {margin:+.4f}, "
                    f"tolerance -{r['promotion_tolerance']})")
        if promote:
            registry.activate_version(models_dir, candidate_version)
        database.mark_uploads_used(db_path, [u["id"] for u in uploads], job_id)

        def short(m):
            return {k: m[k] for k in ("accuracy", "loss", "macro_precision", "macro_recall", "macro_f1",
                                      "balanced_accuracy", "samples")}
        database.update_job(
            db_path, job_id, status="completed", finished_at=registry.utc_now(), progress="done",
            candidate_version=candidate_version, promotion_decision=decision, promoted=int(promote),
            train_metrics={"final_loss": float(frame["loss"].iloc[-1]), "final_accuracy": float(frame["accuracy"].iloc[-1]),
                           "epochs": int(len(frame)), "train_images": len(train_files)},
            candidate_metrics=short(candidate_metrics), active_metrics=short(active_metrics))
        print(decision)
        return database.get_job(db_path, job_id)
    except Exception as error:
        traceback.print_exc()
        # The active model is untouched: activation only happens after a successful evaluation.
        database.update_job(db_path, job_id, status="failed", finished_at=registry.utc_now(),
                            error=f"{error.__class__.__name__}: {error}", progress="failed")
        return database.get_job(db_path, job_id)
    finally:
        beating.set()
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Run one retraining job")
    parser.add_argument("--job-id", help="existing job id (created by the API)")
    parser.add_argument("--start", action="store_true", help="create a job and run it in this process")
    args = parser.parse_args()
    cfg = load_config()
    database.init_db(cfg["paths"]["database"])
    if args.start:
        job_id = start_retraining(cfg, run_in_process=True)
    else:
        job_id = args.job_id
        run_job(cfg, job_id)
    print(database.get_job(cfg["paths"]["database"], job_id))


if __name__ == "__main__":
    main()
