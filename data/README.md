# Data — Chula-ParasiteEgg-11

Source: Zenodo record [15866298](https://zenodo.org/records/15866298) (ICIP 2022 Parasitic Egg Detection and
Classification challenge; Chulalongkorn University and collaborators). Full audit: [reports/dataset_audit.md](../reports/dataset_audit.md).

## Layout

```
data/
├── raw/                      # official downloads (git-ignored)
│   ├── Chula-ParasiteEgg-11.zip        # train: 11,000 photos + COCO labels (~9.1 GB)
│   ├── Chula-ParasiteEgg-11_test.zip   # test: 2,200 photos (~2.4 GB)
│   └── test_labels_200.json            # COCO labels for all 2,200 test photos
├── extracted/                # unzipped archives (git-ignored) - kept for future object detection
├── processed/                # classification dataset (crops git-ignored)
│   ├── crops/{train,val,test}/<NN_class>/*.jpg   # one egg per image
│   ├── class_map.json        # class_id ↔ official label ↔ display name ↔ folder
│   ├── manifest.csv          # one row per crop: id, path, split, class, source photo, annotation id, boxes
│   └── summary.json          # crop counts per split/class + settings used
├── bundle/                   # small subset for the deployed API (validation set + replay sample)
├── uploads/                  # browsable copy of images uploaded for retraining (the database is the source of truth)
└── chula_parasite_egg11_crops.zip   # crops + metadata, ready to upload to Colab / Kaggle
```

## Getting the data

1. Download the three files from <https://zenodo.org/records/15866298> into `data/raw/`
   (Zenodo throttles single connections; a download manager with several connections is much faster).
2. Build the crops — either run the notebook (`notebook/ovascope_ml_pipeline.ipynb`, Section 5), or:

```bash
python scripts/prepare_data.py            # verifies MD5s, unzips if needed, builds + verifies crops
python scripts/prepare_data.py --rebuild  # force a rebuild
python scripts/audit_dataset.py           # full audit (duplicates, leakage, statistics)
```

The rebuild is deterministic: rebuilding produced all 13,259 crops **byte-identical** to the previous
build (checked by MD5 of every file), with both Pillow 12.3 and Pillow 13.0.

| File | MD5 (from Zenodo) |
|---|---|
| Chula-ParasiteEgg-11.zip | 1621521969547b0291c0cfbf7e450074 |
| Chula-ParasiteEgg-11_test.zip | a47fdab77ced45bc40b0e8b1e6ff8d22 |
| test_labels_200.json | 4ea3e8dec6b8e8dc269441dd5e3cde11 |

## Crop counts (one egg per image, 13,259 in total)

| Class folder | train | val | test |
|---|---|---|---|
| 00_ascaris_lumbricoides | 859 | 150 | 207 |
| 01_capillaria_philippinensis | 850 | 150 | 203 |
| 02_enterobius_vermicularis | 850 | 150 | 200 |
| 03_fasciolopsis_buski | 850 | 150 | 199 |
| 04_hookworm_egg | 859 | 151 | 204 |
| 05_hymenolepis_diminuta | 850 | 150 | 200 |
| 06_hymenolepis_nana | 851 | 150 | 202 |
| 07_opisthorchis_viverrini | 850 | 150 | 200 |
| 08_paragonimus_spp | 850 | 150 | 209 |
| 09_taenia_spp_egg | 860 | 151 | 204 |
| 10_trichuris_trichiura | 850 | 150 | 200 |
| **Total** | **9,379** | **1,652** | **2,228** |

The classes are balanced. Some classes have slightly more than 1,000 crops because 29 training photos
and 28 test photos contain two or three eggs.

## How the crops are made (`src/preprocessing.py`)

- Each COCO box `[x, y, w, h]` is padded by 10% per side and made **square** around its centre (so resizing to
  224×224 does not distort the egg), then shifted/clamped to stay inside the photo.
- Crops are saved at native resolution (JPEG q=95); resizing and augmentation happen in the training pipeline.
- **train/val**: 85/15 split of the official training photos, stratified by class, seed 42, **grouped by source
  photo** — eggs from the same photo never appear in both splits.
- **test**: the official test set with its released labels (`test_labels_200.json` covers all 2,200 photos). It is
  used only for the final evaluation, and its hashes are blocked from retraining uploads.
- The test archive contains a stray `irfanview/` image-viewer settings folder; it holds no images and is ignored.

## Label names

Official category ids (0–10) and label strings are preserved unchanged in `class_map.json` (`original_name`).
Display-name normalisation (folder names and `display_name` only):

| id | Official label | Display name |
|---|---|---|
| 7 | Opisthorchis viverrine | Opisthorchis viverrini |
| 8 | Paragonimus spp | Paragonimus spp. |

## Disk use

About 24 GB in total. `data/extracted/` can be deleted and re-created from `data/raw/` at any time.
