# Dataset download

`download_dataset.py` downloads the original **Chula-ParasiteEgg-11** dataset from
[Zenodo record 15866298](https://zenodo.org/records/15866298) and can rebuild the OvaScope crop dataset from scratch.

```bash
conda activate ovascope                                        # or any environment with requirements.txt installed
python dataset_download/download_dataset.py                    # download + MD5-verify the 3 files into data/raw/
python dataset_download/download_dataset.py --prepare          # ... then build the 13,259 single-egg crops
python dataset_download/download_dataset.py --prepare --audit  # ... and run the dataset audit
```

| File | Size | Content |
|---|---|---|
| `Chula-ParasiteEgg-11.zip` | 9.1 GB | 11,000 training photos + COCO bounding-box labels |
| `Chula-ParasiteEgg-11_test.zip` | 2.4 GB | 2,200 test photos |
| `test_labels_200.json` | 0.4 MB | COCO labels of the test photos |

- **Resumable:** if the connection drops, run the same command again; it continues where it stopped.
  Complete files with the correct MD5 are skipped.
- **Faster:** Zenodo throttles single connections. If `aria2c` is installed (`sudo apt install aria2`), the script
  uses it automatically with 8 connections. `--no-aria2` forces the built-in downloader.
- **Partial download:** `--only test_labels_200.json Chula-ParasiteEgg-11_test.zip`.

What `--prepare` produces (`scripts/prepare_data.py`):
```
data/processed/
├── crops/{train,val,test}/<NN_class>/*.jpg   9,379 / 1,652 / 2,228 crops (one egg each)
├── class_map.json   manifest.csv   summary.json
```
The rebuild is deterministic: the crops come out byte-identical to the ones used to train the published models.
Afterwards, train with the notebook (`notebook/ovascope_ml_pipeline.ipynb`) or
`python -m src.train --model both --select`, and create the deployment bundle with
`python scripts/export_artifacts.py` (needed for retraining in Docker).

Dataset licence: MIT. Please cite the dataset authors (see the main README, References).
