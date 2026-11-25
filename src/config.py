"""Load configs/config.yaml and resolve paths.

Every module, the notebook, the API and the tests read settings through
`load_config()`, so there is a single place to change them.

Environment-variable overrides (useful in Docker, Colab and Kaggle):
    OVASCOPE_ROOT           project root (default: folder containing configs/)
    OVASCOPE_PROCESSED_DIR  folder holding crops/, class_map.json, manifest.csv
    OVASCOPE_MODELS_DIR     model folder
    OVASCOPE_ARTIFACTS_DIR  outputs folder
    OVASCOPE_DB             SQLite database file
    OVASCOPE_GATE_DATA_DIR  non-egg images for the input check (egg gate)
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ENV_OVERRIDES = {
    "processed_dir": "OVASCOPE_PROCESSED_DIR",
    "models_dir": "OVASCOPE_MODELS_DIR",
    "artifacts_dir": "OVASCOPE_ARTIFACTS_DIR",
    "database": "OVASCOPE_DB",
    "uploads_dir": "OVASCOPE_UPLOADS_DIR",
    "raw_dir": "OVASCOPE_RAW_DIR",
    "extracted_dir": "OVASCOPE_EXTRACTED_DIR",
    "gate_data_dir": "OVASCOPE_GATE_DATA_DIR",
}


def find_project_root() -> Path:
    if os.environ.get("OVASCOPE_ROOT"):
        return Path(os.environ["OVASCOPE_ROOT"]).resolve()
    # src/config.py -> project root is one level up from src/
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> dict:
    root = find_project_root()
    config_path = Path(config_path) if config_path else root / "configs" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    resolved = {}
    for key, value in cfg["paths"].items():
        value = os.environ.get(ENV_OVERRIDES.get(key, ""), value)
        path = Path(value)
        resolved[key] = path if path.is_absolute() else root / path
    cfg["paths"] = resolved
    cfg["paths"]["root"] = root
    cfg["paths"]["crops_dir"] = resolved["processed_dir"] / "crops"
    return cfg
