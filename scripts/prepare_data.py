"""Build (or verify) the single-egg crop dataset.

    python scripts/prepare_data.py            # reuse crops if they verify, else build
    python scripts/prepare_data.py --rebuild  # always rebuild from the official archives

Expects the official Zenodo files (record 15866298) in data/raw/.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.preprocessing import prepare_dataset  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--rebuild", action="store_true", help="delete and rebuild crops")
parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
args = parser.parse_args()

report = prepare_dataset(load_config(), rebuild=args.rebuild, workers=args.workers)
print(json.dumps({k: v for k, v in report.items() if k != "per_split_class"}, indent=2))
sys.exit(0 if report["ok"] else 1)
