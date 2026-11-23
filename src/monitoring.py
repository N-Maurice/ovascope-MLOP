"""Operational monitoring: request counts, failures, latency, uptime, retraining events.

This measures how the *service* behaves. It says nothing about whether the
predictions are correct - production requests have no verified labels, so
accuracy is only measured with labelled data (scripts/evaluate_deployed.py).
"""
from __future__ import annotations

import time

import numpy as np

from src.database import connect
from src.registry import utc_now

PROCESS_STARTED = time.time()


def record_request(db_path, endpoint: str, status_code: int, latency_ms: float, request_id: str | None = None,
                   model_version: str | None = None, predicted_class_id: int | None = None,
                   confidence: float | None = None) -> None:
    with connect(db_path) as db:
        db.execute("INSERT INTO request_log (request_id, timestamp, endpoint, status_code, latency_ms, model_version,"
                   " predicted_class_id, confidence) VALUES (?,?,?,?,?,?,?,?)",
                   (request_id, utc_now(), endpoint, status_code, latency_ms, model_version,
                    predicted_class_id, confidence))


def uptime_seconds() -> float:
    """Time since *this* API process started (no historical uptime is invented)."""
    return time.time() - PROCESS_STARTED


def get_metrics(db_path, recent: int = 200) -> dict:
    with connect(db_path) as db:
        totals = db.execute(
            "SELECT COUNT(*) AS total, SUM(status_code < 400) AS ok, SUM(status_code >= 400) AS failed,"
            " AVG(latency_ms) AS avg_ms FROM request_log WHERE endpoint='/predict'").fetchone()
        latest = db.execute("SELECT timestamp, status_code, latency_ms, model_version, predicted_class_id, confidence"
                            " FROM request_log WHERE endpoint='/predict' ORDER BY id DESC LIMIT ?",
                            (recent,)).fetchall()
        since_start = db.execute(
            "SELECT COUNT(*) AS n FROM request_log WHERE endpoint='/predict' AND timestamp >= ?",
            (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(PROCESS_STARTED)),)).fetchone()
        jobs = db.execute("SELECT id, status, created_at, finished_at, candidate_version, promotion_decision"
                          " FROM retrain_jobs ORDER BY created_at DESC LIMIT 5").fetchall()
    latencies = np.array([r["latency_ms"] for r in latest if r["status_code"] < 400])
    return {
        "uptime_seconds": round(uptime_seconds(), 1),
        "process_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(PROCESS_STARTED)),
        "prediction_requests_total": totals["total"] or 0,
        "prediction_requests_successful": totals["ok"] or 0,
        "prediction_requests_failed": totals["failed"] or 0,
        "prediction_requests_since_start": since_start["n"],
        "average_latency_ms": round(totals["avg_ms"], 2) if totals["avg_ms"] else None,
        "recent_latency_ms": {
            "count": int(latencies.size),
            "p50": float(np.percentile(latencies, 50)) if latencies.size else None,
            "p95": float(np.percentile(latencies, 95)) if latencies.size else None,
            "max": float(latencies.max()) if latencies.size else None,
        },
        "recent_requests": [dict(r) for r in latest[:25]],
        "recent_retraining": [dict(j) for j in jobs],
    }
