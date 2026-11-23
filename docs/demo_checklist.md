# Video demonstration checklist

Camera **on** for the whole recording (rubric requirement). Show live functionality only; do not present
screenshots or simulated responses as live behaviour. Suggested length: 8–12 minutes.

## Before recording

- [ ] `docker compose up -d --scale api=2` (or the cloud URL) and `curl localhost:8000/health` shows `model_loaded: true`
- [ ] Note the active model version (Dashboard) — it should be the version reported in `reports/model_evaluation.md`
- [ ] Prepare **one test crop** whose true class you know (e.g. from `data/processed/crops/test/10_trichuris_trichiura/`)
- [ ] Prepare a folder of **≥ 10 labelled crops for upload** that are *not* test images
      (e.g. copy some from `data/processed/crops/train/<class>/` into a ZIP laid out as `<class folder>/<image>`)
- [ ] Set `retraining.min_new_images` ≤ the number you will upload; enter the admin key in the sidebar
- [ ] Open the notebook with outputs, the Locust plots in `load_testing/results/`, and `reports/`

## Segment 1 — Introduction (≈1 min)
- [ ] OvaScope MLOps: classifies a crop of one parasite egg into 11 classes
- [ ] Problem: manual microscopy is slow and needs expertise; this is a research prototype, not a diagnostic device
- [ ] Dataset: Chula-ParasiteEgg-11 (Zenodo 15866298), detection boxes converted to single-egg crops
- [ ] Functions: prediction, bulk upload to database, retraining, monitoring, Docker, load testing

## Segment 2 — Dataset and notebook (≈2 min)
- [ ] Section 5: crops rebuilt from the official archives and shown byte-identical
- [ ] Section 7–8: class distribution, brightness, size, texture plots and their interpretation
- [ ] Section 9: preprocessing and augmentation examples
- [ ] Sections 11/14: model architectures; Sections 12/15: training logs and curves
- [ ] Sections 17–18: comparison table, test metrics, confusion matrix

## Segment 3 — Prediction (≈1 min)
- [ ] *Predict* page → upload the prepared test crop → **Predict**
- [ ] Read out predicted class, confidence and model version; compare with the known true label
- [ ] Explain: confidence is a model score, not a guarantee

## Segment 4 — Database upload (≈1 min)
- [ ] *Upload training data* → ZIP or multiple images with a label → **Upload**
- [ ] Show accepted/rejected lists and "Saved N image(s) to the database"
- [ ] Show the database records table (and optionally `sqlite3 db/ovascope.db "select id, filename, class_label, length(image_data) from uploads"`)

## Segment 5 — Retraining (≈2 min)
- [ ] *Train & retrain* → current active model and number of new images
- [ ] Press **Retrain Model**; click *Refresh status* to show progress (epoch lines)
- [ ] Show the completed job: base version → candidate version, active vs candidate metrics, promotion decision
- [ ] Show the new version in the *Model versions* table (and rollback button)

## Segment 6 — Deployment (≈1 min)
- [ ] `docker compose ps` showing gateway, ui and api containers (or the cloud URL in the browser)
- [ ] `curl .../health` twice → different `instance` values when 2+ API containers run

## Segment 7 — Monitoring (≈30 s)
- [ ] *Monitoring* page: health, uptime since process start, request count, latency, failures, recent requests

## Segment 8 — Locust (≈1 min)
- [ ] `reports/load_test_report.md` plots: users vs response time, containers vs throughput / p95, failure rate
- [ ] State the measured result honestly (including if more containers did not help)

## Segment 9 — Conclusion (≈30 s)
- [ ] What was implemented, the actual test macro-F1/accuracy, what worked, remaining limitations
