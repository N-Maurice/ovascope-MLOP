"""Locust load test for the OvaScope prediction endpoint.

Each simulated user repeatedly POSTs a real crop image to /predict.
The workload is a fixed list of 110 held-out test crops (10 per class, seed 0),
so every configuration is tested with the same images.

    locust -f load_testing/locustfile.py --host http://localhost:8000 \
           --headless -u 20 -r 5 -t 60s --csv load_testing/results/c1_u20

Set OVASCOPE_LOADTEST_IMAGES to a folder of images to use a different workload.
"""
import os
import random
import sys
from pathlib import Path

from locust import HttpUser, between, events, task

ROOT = Path(__file__).resolve().parent.parent


def load_workload():
    folder = os.environ.get("OVASCOPE_LOADTEST_IMAGES")
    if folder:
        paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    else:
        sys.path.insert(0, str(ROOT))
        from src.preprocessing import read_manifest
        processed = Path(os.environ.get("OVASCOPE_PROCESSED_DIR", ROOT / "data" / "processed"))
        rows = [r for r in read_manifest(processed) if r["split"] == "test"]
        rng = random.Random(0)
        paths = []
        for class_id in range(11):
            members = sorted((r for r in rows if r["class_id"] == class_id), key=lambda r: r["crop_id"])
            paths += [processed / r["crop_path"] for r in rng.sample(members, 10)]
    if not paths:
        raise SystemExit("No images found for the load test")
    return [(p.name, p.read_bytes()) for p in paths]


WORKLOAD = load_workload()


@events.init.add_listener
def describe(environment, **kwargs):
    sizes = [len(b) for _, b in WORKLOAD]
    print(f"Workload: {len(WORKLOAD)} images, {min(sizes)}-{max(sizes)} bytes (mean {sum(sizes) / len(sizes):.0f})")


class PredictionUser(HttpUser):
    wait_time = between(0.5, 1.5)   # think time between requests of one user

    @task
    def predict(self):
        name, data = random.choice(WORKLOAD)
        with self.client.post("/predict", files={"file": (name, data, "image/jpeg")}, name="/predict",
                              catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            elif "predicted_class_id" not in response.json():
                response.failure("malformed response")
