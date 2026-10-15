# IMPLEMENTATION.md — Phishing Detection Platform

Streamlit web app that analyses emails/messages and URLs, scores phishing risk, explains why, and exports a PDF report.

## 0. Hard constraints (never violate)

1. GitHub repo **< 10 MB total** (check with `du -sh .git .` before every push).
2. Public **web app link** (Streamlit Community Cloud).
3. **Never fetch, visit or render user-submitted URLs.** Static analysis only (avoids SSRF, malware, tracking).
4. No raw datasets in git. Commit only: code, a tiny sample (< 200 KB), trained model files (< 3 MB combined), small config JSON.
5. Every claim in the UI/report must be traceable to a named signal. No unexplained scores.
6. After every completed task, update `COMPLETED.md` (see section 9).

## 1. Scope

| Feature | Approach |
|---|---|
| ML phishing classifier | TF-IDF + Logistic Regression (text); lexical-feature model for URLs |
| Evaluation | precision, recall, F1, confusion matrix, **plus cross-dataset test** |
| Email/message analysis | Paste text or upload `.eml`; extract sender, reply-to, subject, body, links |
| Suspicious URL detection | Lexical/structural features + rules |
| Brand impersonation | Curated list of ~15 brands, lookalike-domain detection |
| Explainable risk | Per-signal contributions, shown as a ranked list |
| Recommendations | Rule-mapped actions per triggered signal |
| PDF report | `fpdf2`, generated in memory |
| Threat-intel API | Optional, last phase, must fail gracefully |

Out of scope: browser extension, live crawling, user accounts, database, deep learning (too heavy for the 10 MB / free-hosting limits and unnecessary).

## 2. Stack

- Python 3.11, Streamlit
- scikit-learn, pandas, numpy, joblib
- `tldextract` (use bundled snapshot: `tldextract.TLDExtract(suffix_list_urls=())` so it never hits the network)
- `rapidfuzz` (edit distance for lookalikes)
- `fpdf2` (PDF)
- `matplotlib` (confusion matrix image)
- Pin **exact** versions in `requirements.txt`. The scikit-learn version used for training must match deployment, or the pickled model can break.
- Do NOT add: torch, transformers, shap, tensorflow (size and cold-start cost).

## 3. Repo structure

```
phishing-detector/
├── app.py                    # Streamlit entry point
├── requirements.txt
├── README.md
├── IMPLEMENTATION.md
├── COMPLETED.md
├── .gitignore                # data/raw, *.csv (except samples), __pycache__
├── src/
│   ├── parsing.py            # email/message parsing, URL extraction
│   ├── url_features.py       # lexical URL features
│   ├── brand_check.py        # brand impersonation logic
│   ├── text_signals.py       # urgency/credential/threat heuristics
│   ├── models.py             # load models, predict, explain
│   ├── risk.py               # combine signals -> score + reasons
│   ├── recommendations.py    # signal -> action mapping
│   ├── report.py             # PDF generation
│   └── intel.py              # optional threat-intel (feature-flagged)
├── training/
│   ├── download_data.py      # fetches datasets (not committed)
│   ├── train_text.py
│   ├── train_url.py
│   └── evaluate.py           # metrics + cross-dataset
├── models/                   # *.joblib, metrics.json, confusion_matrix.png
├── data/
│   ├── brands.json
│   └── samples/              # a few demo emails/URLs (tiny)
└── tests/
    └── test_*.py
```

## 4. Data (VERIFY availability and licences yourself)

Candidates (FACT: these are commonly used; UNCERTAINTY: current links/licences, check before use):
- Phishing + legitimate emails: Nazario phishing corpus, Enron or SpamAssassin ham, Kaggle "phishing email" collections
- URLs: PhishTank / OpenPhish feeds (phishing), Tranco top list (legitimate), Kaggle URL datasets

Rules:
- Document every source and licence in `README.md`.
- **Deduplicate** before splitting. Duplicates across train/test inflate scores.
- Split stratified, fixed random seed, then **never touch the test set during tuning**.
- Keep a **second, independent dataset** only for final cross-dataset evaluation.

## 5. Modelling design

**Text model:** `TfidfVectorizer(ngram_range=(1,2), max_features≈20000, sublinear_tf=True, min_df=2)` + `LogisticRegression(class_weight="balanced")`. Explanation = per-token `coef × tfidf value`, top 10 pushing towards phishing.

**URL model:** hand-built features: length, dot count, subdomain depth, hyphens, digits ratio, `@`, IP-as-host, punycode (`xn--`), entropy, suspicious TLD, HTTPS flag, keyword hits (login, verify, secure, update…), shortener domain. Model: Logistic Regression (explainable) vs Random Forest. Compare both, ship the better one, and state in README how you explain whichever you ship.

**Brand impersonation (rule-based, not ML):** for each URL, get registered domain. Flag if (a) brand name appears in subdomain/path but registered domain is not on the brand's allowlist, (b) registered domain is within edit distance ≤ 2 of a brand domain, (c) homoglyphs (`0→o`, `1→l`, `rn→m`). Brand list in `data/brands.json`.

**Email-specific signals:** display text vs actual `href` mismatch, Reply-To domain ≠ From domain, urgency/threat language, credential requests, attachment mentions, SPF/DKIM/DMARC results if present in headers (report as "not available" if absent, never assume).

**Risk score:** weighted combination of text prob, max URL prob, brand flag, heuristic signals. Weights live in one config dict, documented. Label output as a **risk score, not a calibrated probability**. Bands: Low / Medium / High / Critical, with each band's rule written down.

## 6. Phases

### Phase 1 — Foundation (Day 1)
- Repo, structure, `.gitignore`, pinned `requirements.txt`, hello-world Streamlit deployed to Community Cloud.
- **Exit:** public URL loads. (Deploy early. Deployment problems are the biggest hidden risk.)

### Phase 2 — Data + text model
- `download_data.py`, dedupe, split, train text model, save to `models/`.
- **Exit:** `metrics.json` with precision/recall/F1 and confusion matrix PNG; model file < 3 MB.

### Phase 3 — URL pipeline
- `url_features.py`, `train_url.py`, unit tests for feature extraction.
- **Exit:** URL model metrics saved; tests pass.

### Phase 4 — Rules: brand + email signals
- `brand_check.py`, `text_signals.py`, `parsing.py` (.eml support).
- **Exit:** tests with known lookalikes (e.g. `paypa1.com`, `micros0ft-login.net`) and a legit control (`paypal.com`) pass.

### Phase 5 — Risk engine + explanations
- `risk.py`, `recommendations.py`; ranked reasons, each linked to evidence (the token, URL part or header).
- **Exit:** every score in the UI has ≥ 1 visible reason; no reason without evidence.

### Phase 6 — Streamlit UI
- Tabs: Analyze Message, Analyze URL, Model Performance (shows metrics + confusion matrix), About/Limitations.
- Use `st.cache_resource` for model loading; handle empty/huge input; cap input size.
- **Exit:** full flow works on 5 demo samples in the deployed app.

### Phase 7 — PDF report
- `report.py` with fpdf2: verdict, score, reasons, URLs, recommendations, timestamp, disclaimer. Sanitise non-Latin-1 characters (fpdf2 core fonts break on them) or bundle a small TTF.
- **Exit:** download button works on deployed app.

### Phase 8 — Evaluation hardening (this is what separates you)
- Cross-dataset test, error analysis (look at 20 false positives and 20 false negatives and write what you find), limitations section.
- **Exit:** README states in-dataset AND cross-dataset scores honestly.

### Phase 9 — Optional threat intel
- Google Safe Browsing or VirusTotal behind a feature flag; keys in `st.secrets`; timeouts; app works with it off. Send only URLs, never message bodies.
- **Exit:** disabled-by-default, no crash on API failure.

### Phase 10 — Polish
- README with screenshots, architecture diagram, demo samples, size check, final deploy verification.

## 7. Known traps

1. **Suspiciously high scores (>99% F1):** usually dataset artefacts. Phishing and ham emails often come from different sources with different formatting, so the model learns the source, not phishing. Cross-dataset testing exposes this. Expect a drop and report it honestly.
2. **Pickle/version mismatch** between training and Streamlit Cloud.
3. **Repo bloat** from committing CSVs or notebooks with outputs. Clear notebook outputs.
4. **Cold starts / memory:** free hosting sleeps and has limited RAM. Keep models small.
5. **PSL download at runtime:** configure `tldextract` offline.
6. **Class imbalance:** report per-class precision/recall, not just accuracy.
7. **Over-claiming:** this detects *likely* phishing, not certainty. Say so in UI and PDF.

## 8. Definition of done

- Public app link works end to end; repo < 10 MB; tests pass
- Metrics + confusion matrix visible in app; cross-dataset result in README
- Every risk output has evidence-backed reasons and recommendations
- PDF downloads correctly; limitations documented

## 9. COMPLETED.md rules (for the coding assistant)

Create `COMPLETED.md` at the start and update it **after every finished task, before reporting back**. Format:

```
# COMPLETED.md
## Current phase: <n — name>
## Done
- [YYYY-MM-DD] Phase X: <what was built> — files: <paths> — verified by: <test/command/result>
## In progress
- <task> — what remains
## Known issues / TODO
- <issue>
## Metrics log
- <date>: <model> P=… R=… F1=… (dataset, split)
## Repo size
- <date>: <du -sh output>
```

Rules: only mark Done when verified by running code or tests, record failures honestly, and never delete earlier entries.