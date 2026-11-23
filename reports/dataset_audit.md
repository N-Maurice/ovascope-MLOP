# Dataset Audit — Chula-ParasiteEgg-11

Audit date: 2026-11-23. Every number below comes from `scripts/audit_dataset.py`
(raw output: `artifacts/metrics/dataset_audit.json`, per-crop measurements:
`artifacts/metrics/crop_image_stats.csv`). Re-run it with `python scripts/audit_dataset.py`.

## Summary

| Item | Status | Evidence | Required action |
|---|---|---|---|
| Dataset integrity | Verified | Zenodo MD5 matched for both archives and `test_labels_200.json`; 11,000 train + 2,200 test photos on disk, all listed in their JSON, none missing; 0 corrupt crops | None |
| Class mapping | Verified | 11 categories, ids 0–10, identical in train and test JSON; `class_map.json` matches both; **0 mismatches** between each training file-name prefix (e.g. `Ascaris lumbricoides_0001.jpg`) and its annotation's category | None. Display names normalised for ids 7 and 8 only (see below) |
| Crop quality | Verified | Box overlay (`artifacts/plots/audit_bbox_overlay.png`) shows `[x,y,w,h]` read correctly; per-class sample sheet (`artifacts/plots/audit_crop_samples.png`) shows each egg centred; 0 empty/flat crops; all 13,259 crops square | Note: 119 boxes extend beyond the photo edge (eggs cut off by the photo itself); crops are clamped inside the image |
| Data leakage | Verified | 0 source photos in more than one split; 0 exact-duplicate photos (MD5); 0 near-duplicate photo pairs across train/val/test (256-bit dHash, threshold 20 bits); 0 identical crops across splits | None |
| Test labels | Verified | `test_labels_200.json` labels **all 2,200** test photos (2,228 eggs, ~200 per class) | Official test set used as the held-out test set |
| Preprocessing | Verified | `src/preprocessing.py` rebuilds all 13,259 crops **byte-for-byte identical** to the previous run (MD5 of every file compared) | None |

## 1. Source images (original photos)

| | Train (official) | Test (official) |
|---|---|---|
| Photos listed in JSON | 11,000 | 2,200 |
| Photos on disk | 11,000 | 2,200 |
| Annotated eggs (boxes) | 11,031 | 2,228 |
| Photos with 1 / 2 / 3 eggs | 10,971 / 27 / 2 | 2,172 / 28 / 0 |
| Photos mixing two classes | 0 | 0 |
| Boxes extending past the photo edge | 101 | 18 |

Annotation format: COCO JSON (`images`, `annotations` with `bbox = [x, y, width, height]` in pixels,
`categories`). Bounding boxes are available for every egg in both sets.

**Photo sizes** (13 distinct resolutions, from 3024×4032 smartphone photos to 896×960 camera frames):
3024×4032 (2,226), 1280×960 (1,662), 1920×1080 (1,649), 3024×2822 (1,122), 2448×3264 (1,082),
2117×4032 (1,074), 1344×1080 (845), 896×960 (835), 1280×672 (824), 1920×756 (795), 1714×3264 (558),
2448×2285 (527), 1546×505 (1). No photo carries an EXIF rotation flag, and every JSON width/height
matches the real file, so boxes need no orientation correction.

Annotations per class — train: 1,009 / 1,000 / 1,000 / 1,000 / 1,010 / 1,000 / 1,001 / 1,000 / 1,000 / 1,011 / 1,000;
test: 207 / 203 / 200 / 199 / 204 / 200 / 202 / 200 / 209 / 204 / 200 (classes 0–10).

## 2. Class mapping

| id | Official label (kept in code) | Display name | Folder |
|---|---|---|---|
| 0 | Ascaris lumbricoides | Ascaris lumbricoides | 00_ascaris_lumbricoides |
| 1 | Capillaria philippinensis | Capillaria philippinensis | 01_capillaria_philippinensis |
| 2 | Enterobius vermicularis | Enterobius vermicularis | 02_enterobius_vermicularis |
| 3 | Fasciolopsis buski | Fasciolopsis buski | 03_fasciolopsis_buski |
| 4 | Hookworm egg | Hookworm egg | 04_hookworm_egg |
| 5 | Hymenolepis diminuta | Hymenolepis diminuta | 05_hymenolepis_diminuta |
| 6 | Hymenolepis nana | Hymenolepis nana | 06_hymenolepis_nana |
| 7 | Opisthorchis viverrine | **Opisthorchis viverrini** | 07_opisthorchis_viverrini |
| 8 | Paragonimus spp | **Paragonimus spp.** | 08_paragonimus_spp |
| 9 | Taenia spp. egg | Taenia spp. egg | 09_taenia_spp_egg |
| 10 | Trichuris trichiura | Trichuris trichiura | 10_trichuris_trichiura |

The established numbering is correct and unchanged. Code never relies on alphabetical folder
order: labels come from `manifest.csv` / `class_map.json`.

## 3. Cropped images (classification dataset)

| Split | Crops | Source photos | Origin |
|---|---|---|---|
| train | 9,379 | 9,350 | official train, 85% |
| val | 1,652 | 1,650 | official train, 15% |
| test (held out) | 2,228 | 2,200 | official test + `test_labels_200.json` |
| **total** | **13,259** | 13,200 | |

Per class: train 850–860, val 150–151, test 199–209. The largest/smallest class ratio is
1.012 in train, so the classes are **balanced**: no class weighting is needed.

How the crops are made (`src/preprocessing.py::build_crop_dataset`):
1. Each box is padded by 10% per side (context around the shell), then made square around its centre.
2. The square is shifted/clamped to stay inside the photo (never padded with artificial pixels).
3. Saved at native resolution as JPEG (quality 95). Resizing to the model size happens later.
4. Train/val split: stratified by class, **grouped by source photo** (all eggs of a photo share a split), seed 42.
5. `manifest.csv` records crop id, path, split, class id/name, source photo, annotation id, original box and crop box.

Crop sizes: square, side 72–967 px (median 265 px); 34.2% are smaller than 224 px and are
upsampled for the model.

## 4. Leakage and duplicate checks

| Check | Method | Result |
|---|---|---|
| Same photo in two splits | manifest source-photo comparison | 0 |
| Exact duplicate photos | MD5 of all 13,200 files | 0 duplicate groups |
| Near-duplicate photos across splits | 256-bit dHash of the central 50% of each photo, Hamming distance ≤ 20 | 0 pairs (train–val, train–test, val–test) |
| Detector sanity check | edited copy of a real photo (half size, +10% brightness, JPEG q=60) | distance 14 → detected |
| Closest real test→train/val photo | same hash | min 70, 5th percentile 86, median 93 bits |
| Identical crops across splits | MD5 of all crops | 0 |

A first attempt with a 64-bit whole-photo dHash reported hundreds of "near-duplicates".
Visual inspection showed they were different photos, often of different classes: the hash was
dominated by the dark circular microscope vignette that every photo shares. The central-region
256-bit hash fixes this, and the positive control confirms it still detects true near-copies.

## 5. Image quality variation

The photos come from several microscopes, cameras and phones. The crops show strong
variation in focus, colour cast and illumination (see `audit_crop_samples.png`, and the brightness,
contrast and sharpness distributions in the notebook and dashboard). This is expected and is
the main reason for brightness/contrast augmentation during training.

## 6. Distinguishing the image sets

| Set | Where | Used for |
|---|---|---|
| Original source photos | `data/extracted/` | crop generation only |
| Cropped parasite images | `data/processed/crops/` | everything below |
| Training images | `crops/train` (9,379) | fitting weights |
| Validation images | `crops/val` (1,652) | early stopping, model selection, retraining promotion |
| Held-out test images | `crops/test` (2,228) | final evaluation only |
| Newly uploaded retraining images | SQLite `db/ovascope.db` (+ copy in `data/uploads/`) | retraining; checked against test-set hashes |

## 7. Limitations

- The 119 boxes that run past the photo edge produce crops where part of the egg is outside the photo.
- "Near-duplicate" detection is hash-based; it rules out re-saved/resized copies but cannot rule out
  two different photos of the same egg taken from very different positions.
- The official test set was collected for the same challenge; it measures in-distribution
  generalisation, not performance on a new laboratory's microscopes.
