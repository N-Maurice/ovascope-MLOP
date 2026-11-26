"""Download the Chula-ParasiteEgg-11 dataset from Zenodo and (optionally) rebuild the OvaScope crops.

    python dataset_download/download_dataset.py                 # download + verify the 3 files (~11.6 GB)
    python dataset_download/download_dataset.py --prepare       # ... then build the 13,259 crops
    python dataset_download/download_dataset.py --prepare --audit   # ... and run the dataset audit
    python dataset_download/download_dataset.py --only test_labels_200.json Chula-ParasiteEgg-11_test.zip

Files (Zenodo record 15866298, MIT licence) are saved to data/raw/:
    Chula-ParasiteEgg-11.zip        training photos + COCO labels   9.1 GB
    Chula-ParasiteEgg-11_test.zip   test photos                     2.4 GB
    test_labels_200.json            COCO labels of the test photos  0.4 MB

Each file is checked against its published MD5. Interrupted downloads resume where they stopped, and
files that are already complete and correct are skipped, so the command can simply be run again.
If `aria2c` is installed it is used automatically (several connections; Zenodo throttles single ones).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402

CHUNK = 1 << 20   # 1 MB


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK * 8), b""):
            digest.update(block)
    return digest.hexdigest()


def file_list(record: str, expected_md5: dict[str, str]) -> list[dict]:
    """Names, sizes and download links from the Zenodo API; falls back to the checksums in the config."""
    try:
        with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=30) as response:
            files = json.load(response)["files"]
        return [{"name": f["key"], "size": f["size"], "md5": f["checksum"].removeprefix("md5:"),
                 "url": f["links"]["self"]} for f in files]
    except Exception as error:   # offline API: use the published values stored in configs/config.yaml
        print(f"Zenodo API not reachable ({error}); using the file list from configs/config.yaml")
        return [{"name": name, "size": None, "md5": md5,
                 "url": f"https://zenodo.org/records/{record}/files/{name}?download=1"}
                for name, md5 in expected_md5.items()]


def download_urllib(url: str, target: Path, size: int | None) -> None:
    """Single-connection download that resumes from a partial file (HTTP Range)."""
    partial = target.with_name(target.name + ".part")
    for attempt in range(1, 11):
        done = partial.stat().st_size if partial.exists() else 0
        if size and done >= size:
            break
        request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, open(partial, "ab" if done else "wb") as out:
                if done and response.status != 206:   # server ignored the range: start again
                    out.seek(0)
                    out.truncate()
                    done = 0
                started, last, resumed_at = time.time(), 0.0, done
                while block := response.read(CHUNK):
                    out.write(block)
                    done += len(block)
                    if time.time() - last > 1:
                        last = time.time()
                        rate = (done - resumed_at) / max(last - started, 1e-6)
                        total = f" / {size / 1e9:.2f} GB ({100 * done / size:.1f}%)" if size else ""
                        print(f"\r  {done / 1e9:.2f} GB{total}  {rate / 1e6:.1f} MB/s   ", end="", flush=True)
            print()
            break
        except Exception as error:
            print(f"\n  interrupted ({error}); retry {attempt}/10 in 5 s - the download resumes")
            time.sleep(5)
    else:
        raise SystemExit(f"Could not download {target.name}; run the command again to resume.")
    partial.replace(target)


def download_aria2(url: str, target: Path) -> None:
    subprocess.run(["aria2c", "-x", "8", "-s", "8", "--continue=true", "--auto-file-renaming=false",
                    "--summary-interval=0", "-d", str(target.parent), "-o", target.name, url], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="FILE", help="download only these files")
    parser.add_argument("--prepare", action="store_true", help="build the crop dataset afterwards")
    parser.add_argument("--audit", action="store_true", help="run the dataset audit afterwards")
    parser.add_argument("--no-aria2", action="store_true", help="use Python's single connection even if aria2c exists")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = cfg["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = file_list(cfg["dataset"]["zenodo_record"], cfg["dataset"]["archives"])
    if args.only:
        unknown = set(args.only) - {f["name"] for f in files}
        if unknown:
            raise SystemExit(f"Unknown file(s): {sorted(unknown)}; available: {[f['name'] for f in files]}")
        files = [f for f in files if f["name"] in args.only]
    use_aria2 = shutil.which("aria2c") and not args.no_aria2
    print(f"Saving to {raw_dir}  ({'aria2c, 8 connections' if use_aria2 else 'single connection'})")

    for f in files:
        target = raw_dir / f["name"]
        size = f" ({f['size'] / 1e9:.2f} GB)" if f["size"] else ""
        if target.exists() and (f["size"] is None or target.stat().st_size == f["size"]):
            print(f"{f['name']}{size}: present, checking MD5 ...", end=" ", flush=True)
            if md5_of(target) == f["md5"]:
                print("OK")
                continue
            print("mismatch, downloading again")
            target.unlink()
        print(f"{f['name']}{size}: downloading")
        if use_aria2:
            download_aria2(f["url"], target)
        else:
            download_urllib(f["url"], target, f["size"])
        actual = md5_of(target)
        if actual != f["md5"]:
            target.unlink()
            raise SystemExit(f"{f['name']}: MD5 {actual} does not match the published {f['md5']}; file removed, "
                             "run the command again")
        print(f"{f['name']}: MD5 verified")

    if args.prepare:
        print("\nBuilding the crop dataset (scripts/prepare_data.py) ...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_data.py")], check=True, cwd=ROOT)
    if args.audit:
        print("\nAuditing the dataset (scripts/audit_dataset.py) ...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_dataset.py")], check=True, cwd=ROOT)
    print("\nDone. Next steps: README.md, sections 6-8 (notebook or `python -m src.train --model both --select`).")


if __name__ == "__main__":
    main()
