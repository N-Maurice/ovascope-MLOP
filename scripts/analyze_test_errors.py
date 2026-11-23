"""Break down validation and test accuracy by source-photo resolution (a proxy for capture device).

    python scripts/analyze_test_errors.py

Uses the probabilities saved by the final evaluation (no new predictions on the test set, no
model changes). Writes artifacts/metrics/accuracy_by_resolution.csv and prints the table.
This is a diagnostic for the validation-to-test gap; it is not used to select or tune models.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import registry  # noqa: E402
from src.config import load_config  # noqa: E402
from src.preprocessing import read_manifest  # noqa: E402

cfg = load_config()
paths, ds = cfg["paths"], cfg["dataset"]
metrics_dir = paths["artifacts_dir"] / "metrics"
active = registry.get_active_version(paths["models_dir"])
manifest = pd.DataFrame(read_manifest(paths["processed_dir"]))

frames = []
for split, probability_file in [("val", f"{active}_val_probabilities.npy"), ("test", "final_test_probabilities.npy")]:
    rows = manifest[manifest.split == split].reset_index(drop=True)
    probabilities = np.load(metrics_dir / probability_file)
    folder = paths["extracted_dir"] / (ds["test_images"] if split == "test" else ds["train_images"])
    rows["resolution"] = [("%dx%d" % Image.open(folder / name).size) for name in rows["source_image"]]
    rows["correct"] = probabilities.argmax(axis=1) == rows["class_id"].values
    frames.append(rows)
data = pd.concat(frames)

table = data.pivot_table(index="resolution", columns="split", values="correct", aggfunc=["mean", "count"])
table.columns = [f"{split}_{'accuracy' if stat == 'mean' else 'crops'}" for stat, split in table.columns]
table = table[["val_crops", "val_accuracy", "test_crops", "test_accuracy"]].sort_values("test_crops", ascending=False)
table.to_csv(metrics_dir / "accuracy_by_resolution.csv")
print(f"Model {active}")
print(table.round(3).to_string())
