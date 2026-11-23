"""Model versioning: every trained or retrained model gets its own folder.

models/
├── versions/
│   ├── v001_baseline_cnn/       model.keras, model.h5, class_mapping.json,
│   ├── v002_mobilenetv2/        model_metadata.json, training_config.json, ...
│   └── v003_retrain/
├── active_model.json            {"version": "...", "previous": "...", "activated_at": "..."}
├── ovascope_best_model.keras    copy of the active model (stable file name)
├── ovascope_best_model.h5       load-tested .h5 export of the active model
└── class_mapping.json           copy of the active class mapping

Nothing is ever overwritten: retraining creates a new version, and activating a
version only changes the pointer file (plus the convenience copies).
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def versions_dir(models_dir: Path) -> Path:
    return Path(models_dir) / "versions"


def list_versions(models_dir: Path) -> list[dict]:
    out = []
    folder = versions_dir(models_dir)
    if not folder.exists():
        return out
    for path in sorted(folder.iterdir()):
        meta = path / "model_metadata.json"
        if meta.exists():
            out.append(json.loads(meta.read_text()))
    return out


def next_version_name(models_dir: Path, kind: str) -> str:
    numbers = [int(m.group(1)) for p in versions_dir(models_dir).glob("v*")
               if (m := re.match(r"v(\d+)_", p.name))]
    return f"v{(max(numbers) if numbers else 0) + 1:03d}_{kind}"


def get_active_version(models_dir: Path) -> str | None:
    pointer = Path(models_dir) / "active_model.json"
    if not pointer.exists():
        return None
    return json.loads(pointer.read_text()).get("version")


def version_path(models_dir: Path, version: str) -> Path:
    path = versions_dir(models_dir) / version
    if not (path / "model.keras").exists():
        raise FileNotFoundError(f"Model version '{version}' not found in {versions_dir(models_dir)}")
    return path


def activate_version(models_dir: Path, version: str) -> dict:
    """Point the application at `version` (atomic pointer write + stable-name copies)."""
    models_dir = Path(models_dir)
    source = version_path(models_dir, version)
    current = get_active_version(models_dir)
    previous = current
    if current == version:   # re-activating the same version keeps the existing rollback target
        previous = json.loads((models_dir / "active_model.json").read_text()).get("previous")
    for name, target in [("model.keras", "ovascope_best_model.keras"), ("model.h5", "ovascope_best_model.h5"),
                         ("class_mapping.json", "class_mapping.json")]:
        if (source / name).exists():
            tmp = models_dir / f".{target}.tmp"
            shutil.copy2(source / name, tmp)
            os.replace(tmp, models_dir / target)
    pointer = {"version": version, "previous": previous, "activated_at": utc_now()}
    tmp = models_dir / ".active_model.json.tmp"
    tmp.write_text(json.dumps(pointer, indent=2))
    os.replace(tmp, models_dir / "active_model.json")   # atomic: readers never see a half-written file
    return pointer


def rollback(models_dir: Path) -> dict:
    pointer = json.loads((Path(models_dir) / "active_model.json").read_text())
    if not pointer.get("previous"):
        raise ValueError("No previous model version to roll back to")
    return activate_version(models_dir, pointer["previous"])
