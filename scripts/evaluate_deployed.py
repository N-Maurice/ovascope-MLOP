"""Evaluate a *deployed* OvaScope API against labelled held-out images.

    python scripts/evaluate_deployed.py --url http://localhost:8000 --per-class 20

Sends real test crops (with verified labels from the manifest) to POST /predict,
compares the returned class ids with the true labels, and reports accuracy,
macro precision/recall/F1 and the sample count. This measures predictive quality
of the running service; it is separate from latency monitoring and load testing.

Writes artifacts/metrics/production_evaluation.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.evaluation import compute_metrics  # noqa: E402
from src.preprocessing import load_class_map, read_manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--per-class", type=int, default=20, help="test crops per class (0 = all)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config()
    processed = cfg["paths"]["processed_dir"]
    class_names = [c["display_name"] for c in load_class_map(processed)]
    rows = [r for r in read_manifest(processed) if r["split"] == "test"]
    rng = random.Random(args.seed)
    if args.per_class:
        sample = []
        for class_id in range(len(class_names)):
            members = sorted((r for r in rows if r["class_id"] == class_id), key=lambda r: r["crop_id"])
            sample += rng.sample(members, min(args.per_class, len(members)))
        rows = sample

    health = requests.get(f"{args.url}/health", timeout=10).json()
    y_true, probabilities, latencies, versions = [], [], [], set()
    for row in rows:
        data = (processed / row["crop_path"]).read_bytes()
        start = time.perf_counter()
        response = requests.post(f"{args.url}/predict", files={"file": (Path(row["crop_path"]).name, data)}, timeout=60)
        latencies.append((time.perf_counter() - start) * 1000)
        response.raise_for_status()
        result = response.json()
        versions.add(result["model_version"])
        # Rebuild a probability vector from the returned top-k scores (others set to 0) for the loss/metrics code.
        vector = np.zeros(len(class_names))
        for item in result["top_k"]:
            vector[item["class_id"]] = item["confidence"]
        probabilities.append(vector)
        y_true.append(row["class_id"])

    metrics = compute_metrics(np.array(y_true), np.array(probabilities), class_names)
    metrics.pop("loss")   # not meaningful from top-3 scores only
    report = {"api_url": args.url, "model_versions": sorted(versions), "api_instance_at_start": health.get("instance"),
              "evaluation_samples": len(rows), "sampling": f"{args.per_class or 'all'} test crops per class, seed {args.seed}",
              "client_latency_ms": {"median": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95))},
              **metrics}
    out = cfg["paths"]["artifacts_dir"] / "metrics" / "production_evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("model_versions", "evaluation_samples", "accuracy", "macro_precision",
                                             "macro_recall", "macro_f1", "client_latency_ms")}, indent=2))


if __name__ == "__main__":
    main()
