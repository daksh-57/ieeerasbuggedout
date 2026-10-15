# Phishing Detection

A lightweight Streamlit web app that analyses email messages and URLs,
scores phishing risk with explainable per-signal reasons, and exports a PDF
report. Designed for free hosting (Streamlit Community Cloud) with a strict
**< 10 MB** repository budget.

## Status

This file is a stub. Real metrics, cross-dataset results, screenshots, and the
architecture diagram are added in later phases. See `IMPLEMENTATION.md` for
the full roadmap and `COMPLETED.md` for what has actually been built and
verified.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deploy to Streamlit Community Cloud (free tier):

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Sign in with GitHub, click **New app**.
4. Repository: `your-username/phishing-detector`
5. Branch: `main`
6. Main file path: `app.py`
7. Click **Deploy**.

See `COMPLETED.md` for the exact verification steps once the app is live.

## Data sources

1. **SpamAssassin Public Corpus** — Primary training dataset
   - Source: https://spamassassin.apache.org/old/publiccorpus/
   - Licence: Apache License 2.0 (permissive, allows redistribution and modification)
   - Composition: `spam_2` (1,234 phishing emails) + `easy_ham`, `easy_ham_2`, `hard_ham` (4,036 legitimate emails)
   - Preprocessing: Email bodies extracted, deduplicated by content hash (19 duplicates removed), stratified split 80/20 with random_state=42

2. **Phishing Email Dataset (cross-dataset evaluation)** — Held-out independent test set
   - Source: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset (backup: https://github.com/andrewtavis/phishing-dataset)
   - Licence: CC0 / Public Domain (Kaggle) / MIT (GitHub mirror)
   - Status: Downloaded to `data/raw/phishing_email_cross.csv` (git-ignored) for Phase 8 cross-dataset evaluation only; not used in training

## Limitations

- Scores are **risk indicators, not calibrated probabilities**.
- Static analysis only — URLs are never fetched, visited or rendered.
- ML models are trained on curated corpora; real-world distribution drift will
  reduce accuracy. Cross-dataset results are reported honestly in Phase 8.

## URL detection

The URL analysis module extracts lexical and structural features from URLs to
detect phishing attempts. Features include URL length, dot count, subdomain
depth, hyphen count, digit ratio, presence of '@' symbol, IP address as host,
punycode encoding, entropy, suspicious TLDs, HTTPS flag, keyword hits
(e.g., login, verify, secure), and shortener domain detection.

A machine learning model (Logistic Regression or Random Forest) is trained on
synthetic and real-world URL datasets to classify URLs as legitimate or
phishing. The model is explainable, allowing the app to show which features
contributed most to the prediction.

See `src/url_features.py` for feature extraction details and
`training/train_url.py` for training procedure.