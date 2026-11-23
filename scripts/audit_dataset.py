"""Audit the original dataset and the generated crops.

Writes:
    artifacts/metrics/dataset_audit.json      every number used in reports/dataset_audit.md
    artifacts/metrics/crop_image_stats.csv    per-crop measurements (EDA + dashboard)
    artifacts/plots/audit_bbox_overlay.png    boxes drawn on source photos (visual check)
    artifacts/plots/audit_crop_samples.png    random crops per class (visual check)

    python scripts/audit_dataset.py
"""
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.preprocessing import image_statistics, load_class_map, load_coco, read_manifest  # noqa: E402

# Near-duplicate detection uses a 256-bit difference hash (dHash) of the CENTRAL 50% of each
# photo. A whole-photo 64-bit hash was tried first and produced hundreds of false matches:
# it is dominated by the dark circular microscope vignette that every photo shares.
HASH_GRID = 16
NEAR_DUPLICATE_BITS = 20  # <= ~8% of the 256 bits differ


def md5_and_dhash(path):
    """Exact hash of the file bytes + 256-bit dHash (as 32 uint8) of the central field of view."""
    data = Path(path).read_bytes()
    with Image.open(path) as image:
        size = image.size
        image.draft("L", (512, 512))          # fast JPEG downscale while decoding
        grey = image.convert("L")
        w, h = grey.size
        centre = grey.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
        small = np.asarray(centre.resize((HASH_GRID + 1, HASH_GRID), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return hashlib.md5(data).hexdigest(), np.packbits(bits), size


def crop_stats(path):
    with Image.open(path) as image:
        stats = image_statistics(image)
    stats["md5"] = hashlib.md5(Path(path).read_bytes()).hexdigest()
    return stats


def near_duplicate_pairs(hashes_a, names_a, hashes_b, names_b):
    """All (a, b, distance) pairs whose hashes differ in at most NEAR_DUPLICATE_BITS bits."""
    hashes_b = np.stack(hashes_b)
    pairs = []
    for i, h in enumerate(hashes_a):
        distances = np.unpackbits(np.bitwise_xor(hashes_b, h), axis=1).sum(axis=1)
        for j in np.nonzero(distances <= NEAR_DUPLICATE_BITS)[0]:
            pairs.append((names_a[i], names_b[j], int(distances[j])))
    return pairs


def main():
    cfg = load_config()
    paths, ds = cfg["paths"], cfg["dataset"]
    extracted, processed = paths["extracted_dir"], paths["processed_dir"]
    plots, metrics = paths["artifacts_dir"] / "plots", paths["artifacts_dir"] / "metrics"
    plots.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    workers = os.cpu_count() or 4
    audit = {}

    # ---- 1. Annotation files -------------------------------------------------
    train_dir, test_dir = extracted / ds["train_images"], extracted / ds["test_images"]
    categories, train_images, train_anns = load_coco(extracted / ds["train_json"])
    test_categories, test_images, test_anns = load_coco(paths["raw_dir"] / ds["test_json_in_raw"])
    audit["categories_train_json"] = [[c["id"], c["name"]] for c in categories]
    audit["categories_identical_train_test"] = categories == test_categories
    class_map = load_class_map(processed)
    audit["class_map_matches_annotations"] = all(
        c["class_id"] == cat["id"] and c["original_name"] == cat["name"]
        for c, cat in zip(class_map, sorted(categories, key=lambda c: c["id"])))

    def describe(images, anns, image_dir):
        files = {p.name for p in image_dir.iterdir() if p.suffix.lower() == ".jpg"}
        eggs_per_image = Counter(len(anns.get(i, [])) for i in images)
        mixed = [images[i]["file_name"] for i in images if len({a["category_id"] for a in anns.get(i, [])}) > 1]
        out_of_bounds = []
        for image_id, image in images.items():
            for a in anns.get(image_id, []):
                x, y, w, h = a["bbox"]
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image["width"] + 1 or y + h > image["height"] + 1:
                    out_of_bounds.append(a["id"])
        names = [images[i]["file_name"] for i in images]
        return {
            "images_in_json": len(images), "image_files_on_disk": len(files),
            "json_images_missing_on_disk": sorted(set(names) - files),
            "files_without_annotation_entry": sorted(files - set(names)),
            "annotations": sum(len(v) for v in anns.values()),
            "images_by_egg_count": {str(k): v for k, v in sorted(eggs_per_image.items())},
            "images_with_mixed_classes": mixed,
            "annotations_per_class": dict(sorted(Counter(a["category_id"] for v in anns.values() for a in v).items())),
            "boxes_outside_image": out_of_bounds,
        }

    audit["train_source"] = describe(train_images, train_anns, train_dir)
    audit["test_source"] = describe(test_images, test_anns, test_dir)

    # Training file names start with the class name ("Ascaris lumbricoides_0001.jpg"):
    # an independent check that each annotation's category id is right.
    name_of = {c["id"]: c["name"] for c in categories}
    label_mismatch = []
    for image_id, image in train_images.items():
        prefix = image["file_name"].rsplit("_", 1)[0]
        for a in train_anns.get(image_id, []):
            if name_of[a["category_id"]] != prefix:
                label_mismatch.append([image["file_name"], name_of[a["category_id"]]])
    audit["train_filename_vs_label_mismatches"] = label_mismatch

    # ---- 2. Source-image hashes: exact + near duplicates ---------------------
    manifest = read_manifest(processed)
    split_of_source = {}
    for row in manifest:
        key = ("test" if row["split"] == "test" else "trainval", row["source_image"])
        split_of_source.setdefault(key, set()).add(row["split"])
    audit["source_images_in_multiple_splits"] = [
        list(k) for k, v in split_of_source.items() if len(v) > 1]

    source_paths = [(("test" if p.parent == test_dir else "trainval"), p)
                    for d in (train_dir, test_dir) for p in sorted(d.glob("*.jpg"))]
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(md5_and_dhash, [p for _, p in source_paths], chunksize=64))
    split_by_name = {}
    for row in manifest:
        split_by_name[(row["split"] == "test", row["source_image"])] = row["split"]
    records = []
    for (origin, path), (md5, dhash, size) in zip(source_paths, results):
        split = split_by_name.get((origin == "test", path.name), "unannotated")
        records.append({"name": f"{origin}/{path.name}", "split": split, "md5": md5, "dhash": dhash, "size": size})

    audit["source_image_sizes"] = {f"{w}x{h}": n for (w, h), n in Counter(r["size"] for r in records).most_common()}
    by_md5 = defaultdict(list)
    for r in records:
        by_md5[r["md5"]].append(r)
    exact_groups = [[(r["name"], r["split"]) for r in g] for g in by_md5.values() if len(g) > 1]
    audit["exact_duplicate_source_groups"] = exact_groups
    audit["exact_duplicates_across_splits"] = [g for g in exact_groups if len({s for _, s in g}) > 1]

    by_split = defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r)
    near = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        near[f"{a}-{b}"] = near_duplicate_pairs([r["dhash"] for r in by_split[a]], [r["name"] for r in by_split[a]],
                                                [r["dhash"] for r in by_split[b]], [r["name"] for r in by_split[b]])
    audit["near_duplicate_threshold_bits"] = NEAR_DUPLICATE_BITS
    # Positive control: an edited copy of a real photo must be detected.
    import tempfile
    from PIL import ImageEnhance
    original = source_paths[0][1]
    with tempfile.TemporaryDirectory() as tmp:
        copy = Image.open(original).convert("RGB")
        copy = ImageEnhance.Brightness(copy.resize((copy.width // 2, copy.height // 2))).enhance(1.1)
        copy.save(Path(tmp) / "copy.jpg", quality=60)
        control = int(np.unpackbits(md5_and_dhash(original)[1] ^ md5_and_dhash(Path(tmp) / "copy.jpg")[1]).sum())
    audit["near_duplicate_positive_control_distance"] = control
    # Closest cross-split distance for every test photo (how far the nearest train/val photo is).
    trainval_hashes = np.stack([r["dhash"] for r in records if r["split"] in ("train", "val")])
    closest = [int(np.unpackbits(trainval_hashes ^ r["dhash"], axis=1).sum(axis=1).min())
               for r in records if r["split"] == "test"]
    audit["test_to_trainval_closest_distance"] = {
        "min": min(closest), "p5": float(np.percentile(closest, 5)), "median": float(np.median(closest))}
    audit["near_duplicate_pairs_across_splits"] = {k: len(v) for k, v in near.items()}
    audit["near_duplicate_examples"] = {k: v[:20] for k, v in near.items()}

    # ---- 3. Crop-level statistics --------------------------------------------
    crop_paths = [processed / r["crop_path"] for r in manifest]
    with ProcessPoolExecutor(workers) as pool:
        stats = list(pool.map(crop_stats, crop_paths, chunksize=64))
    with open(metrics / "crop_image_stats.csv", "w", newline="") as f:
        fields = ["crop_id", "split", "class_id", "folder"] + [k for k in stats[0] if k != "md5"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row, s in zip(manifest, stats):
            writer.writerow({**{k: row[k] for k in ("crop_id", "split", "class_id", "folder")}, **s})
    crop_md5 = defaultdict(set)
    for row, s in zip(manifest, stats):
        crop_md5[s["md5"]].add(row["split"])
    audit["crops_total"] = len(manifest)
    audit["crops_per_split"] = dict(Counter(r["split"] for r in manifest))
    audit["crops_per_split_class"] = {
        split: dict(sorted(Counter(r["class_id"] for r in manifest if r["split"] == split).items()))
        for split in ("train", "val", "test")}
    audit["exact_duplicate_crops_across_splits"] = sum(1 for v in crop_md5.values() if len(v) > 1)
    sides = np.array([s["width"] for s in stats])
    audit["crop_side_px"] = {"min": int(sides.min()), "median": int(np.median(sides)), "max": int(sides.max()),
                             "share_below_224": float((sides < 224).mean())}
    audit["non_square_crops"] = int(sum(s["width"] != s["height"] for s in stats))
    audit["empty_or_flat_crops"] = int(sum(s["contrast"] < 2 for s in stats))

    # ---- 4. Visual checks ----------------------------------------------------
    rng = random.Random(0)
    overlay_ids = rng.sample(sorted(i for i in train_images if len(train_anns.get(i, [])) >= 1), 3) + \
        [i for i in train_images if len(train_anns.get(i, [])) > 1][:1]
    tiles = []
    for image_id in overlay_ids:
        image = Image.open(train_dir / train_images[image_id]["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for a in train_anns[image_id]:
            x, y, w, h = a["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=max(3, image.width // 300))
        image.thumbnail((480, 480))
        tiles.append(image)
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + 10 * len(tiles), max(t.height for t in tiles)), "white")
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 10
    sheet.save(plots / "audit_bbox_overlay.png")

    per_class = defaultdict(list)
    for row in manifest:
        per_class[row["class_id"]].append(row)
    tile, columns = 110, 8
    sheet = Image.new("RGB", (columns * tile, len(per_class) * tile), "white")
    for c, rows in sorted(per_class.items()):
        for k, row in enumerate(rng.sample(rows, columns)):
            sheet.paste(Image.open(processed / row["crop_path"]).resize((tile - 4, tile - 4)), (k * tile, c * tile))
    sheet.save(plots / "audit_crop_samples.png")

    (metrics / "dataset_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    print(json.dumps({k: v for k, v in audit.items() if "examples" not in k}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
