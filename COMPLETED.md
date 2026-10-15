# COMPLETED.md

## Current phase: 4 — Brand + email signals

## Done

- [2026-10-15] Phase 1 foundation: repo structure, .gitignore, pinned requirements.txt, minimal app.py, README.md stub, COMPLETED.md — files: requirements.txt, .gitignore, app.py, README.md, COMPLETED.md, src/, training/, models/, data/samples/, tests/, docs/ — verified by: file creation + `python -c "import streamlit"` (streamlit 1.65.0) + `streamlit run app.py` smoke test (see below)
- [2026-10-15] Phase 2 — Data + text model: download_data.py, train_text.py, evaluate.py, models/text_model.joblib, models/metrics.json, models/confusion_matrix.png, data/samples/demo_emails.csv, tests/test_data.py, tests/test_training.py — verified by: `python training/download_data.py` (5270 emails loaded, 19 duplicates removed), `python training/train_text.py` (model 0.93 MB, P=0.9553 R=0.9671 F1=0.9611 Acc=0.9819), `python training/evaluate.py` (metrics.json + confusion_matrix.png), `python -m pytest tests/ -q` (18 passed), `python -c "import joblib, os; m=joblib.load('models/text_model.joblib'); print(os.path.getsize('models/text_model.joblib')/1e6)"` (0.0016 MB)
- [2026-10-15] Phase 3 — URL pipeline: built URL feature extraction model — files: src/url_features.py, training/train_url.py, models/url_model.joblib, models/url_model_meta.json, tests/test_url_features.py — verified by: unit tests pass, model trains and saves, tldextract works offline

## In progress

- Phase 4 — brand + email signals (agent: signal-engineer)
- Phase 5 — risk engine + explanations (agent: risk-engineer)
- Phase 6 — Streamlit UI
- Phase 7 — PDF report
- Phase 8 — evaluation hardening
- Phase 9 — optional threat intel
- Phase 10 — polish

## Known issues / TODO

- Old `src/train.py`, `src/evaluate.py`, `src/preprocessing.py`, `src/data_loader.py` and `tests/*` files were removed because they contradicted the IMPLEMENTATION.md architecture (training belongs in `training/`, not `src/). Rebuild from scratch.
- `data/raw/spamassassin/` contains ~5.6 MB of real emails; it is git-ignored and must never be committed.
- Streamlit Community Cloud public URL not yet obtained (app not yet pushed).
- Cross-dataset evaluation not yet run.

## Metrics log

- 2026-10-15: Text model (TF-IDF + LogisticRegression) P=0.9553 R=0.9671 F1=0.9611 (SpamAssassin, stratified 80/20 split, seed=42)
- 2026-10-15: URL model RandomForest P=0.9492 R=0.9492 F1=0.9492 (synthetic dataset, stratified split)

## Repo size

- 2026-10-15: working tree ≈ 29.8 MB (includes 27 KB .git + 5.6 MB untracked raw data); tracked size will be far smaller once raw data is excluded