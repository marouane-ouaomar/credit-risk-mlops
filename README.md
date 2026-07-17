# Credit Risk Scoring — End-to-End MLOps Project

Predicts probability of loan default from applicant data, using the
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset,
served as a REST API with explainability, containerized, and CI-tested.

This is not "train a model in a notebook." It's a small but complete
production pipeline: **data → model → API → container → CI → monitoring.**

---

## Architecture

```
Kaggle tables (bureau, previous_application, POS_CASH,
installments, credit_card) ──▶ src/build_features.py ──▶ feature_store.parquet
                                                                  │
application_train.csv ───────────────────────────────────────────┤
                                                                  ▼
                                                src/data_processing.py (join + clean)
                                                                  │
                                                                  ▼
                                                     processed parquet + feature_medians.json
                                                                  │
                                                                  ▼
                                                src/train.py (XGBoost + SHAP)
                                                                  │
                                                                  ▼
                                         models/model.pkl + models/explainer.pkl
                                                                  │
                                                                  ▼
                                     app/main.py (FastAPI, optional credit_history input) ──▶ Docker image
                                                                  │
                                                                  ▼
                                             Render/Railway deployment (live URL)
                                                                  │
                                                                  ▼
                                     src/monitor.py (drift check on logged requests)
```

---

## 0. Prerequisites

- Python 3.10+
- Git + a GitHub account
- Docker Desktop (or Docker Engine)
- A free [Kaggle](https://www.kaggle.com) account (for the dataset + API token)
- (Optional, for deployment) a free [Render](https://render.com) account

---

## 1. Initialize the repo

```bash
git init credit-risk-mlops
cd credit-risk-mlops
git add .
git commit -m "Initial scaffold: MLOps credit risk project"
```

Create a repo on GitHub (via the website, name it `credit-risk-mlops`), then:

```bash
git remote add origin https://github.com/<your-username>/credit-risk-mlops.git
git branch -M main
git push -u origin main
```

---

## 2. Environment setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Get the data (Kaggle API)

1. Kaggle → Account → "Create New API Token" → downloads `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` (Linux/Mac) and `chmod 600 ~/.kaggle/kaggle.json`.
3. Join the competition on the Kaggle website once (required, one click) — Home Credit Default Risk.
4. Download:

```bash
pip install kaggle --break-system-packages   # if needed
kaggle competitions download -c home-credit-default-risk -p data/raw
cd data/raw && unzip home-credit-default-risk.zip && cd ../..
```

This project uses **all** the tables, not just `application_train.csv`:
`bureau.csv`, `bureau_balance.csv`, `previous_application.csv`,
`POS_CASH_balance.csv`, `installments_payments.csv`, and
`credit_card_balance.csv`. Unzip everything into `data/raw/` — the scripts
below expect them all there.

**Heads up on file size:** combined, these tables are a few GB. Building
the feature store (next step) can take a few minutes and needs a
reasonable amount of RAM — that's expected, not a bug.

---

## 4. Build the multi-table credit-history feature store

```bash
python -m src.build_features
```

This aggregates the five auxiliary tables into one row per applicant
(`SK_ID_CURR`) — things like "number of prior bureau credits", "share of
past installments paid late", "average credit card utilization" — and
writes `data/processed/feature_store.parquet`. This is the same pattern a
real bank's feature pipeline uses: raw transactional history gets
aggregated **offline**, and both the model and the serving API consume the
aggregates, never the raw tables directly.

Run this once per data refresh — it doesn't need to be repeated every time
you retrain the model on the same data.

---

## 5. Feature engineering + processing

```bash
python -m src.data_processing
```

Reads `data/raw/application_train.csv`, joins it with the feature store
from the previous step, engineers/cleans the combined feature set, and
writes `data/processed/train_processed.parquet` plus
`models/feature_medians.json` (used by the API as fallback values — see
step 7).

**Why a curated feature set, not every column across six tables?**
Real risk models are often *required* to be explainable to regulators and
loan officers — throwing hundreds of raw + aggregated columns at XGBoost
fails that test, and most of them are redundant anyway. `src/data_processing.py`
keeps ~11 application-level fields (income, credit amount, employment
length, external bureau scores) plus 10 curated credit-history aggregates
(bureau count, past-due history, prior-application approval rate, card
utilization, etc.) — each with a clear, defensible business meaning. This
is a deliberate modeling decision, not a limitation — say so explicitly in
interviews.

**Cold start, handled explicitly:** a meaningful share of applicants have
no bureau history, no prior Home Credit applications, or both — new
customers, not a data error. Rather than silently imputing zeros (which
would read as "definitely no risk"), two flags — `HAS_BUREAU_HISTORY` and
`HAS_PREVIOUS_APPLICATION` — let the model treat "no history" as its own
signal, and missing numeric aggregates fall back to the training median
(saved in `feature_medians.json`) instead of zero.

---

## 6. Train the model

```bash
python -m src.train
```

This:
- Splits train/validation
- Trains an XGBoost classifier
- Computes **AUC** and the **KS statistic** (the two metrics real credit
  risk teams report — accuracy is close to meaningless on ~8% default rate data)
- Fits a SHAP `TreeExplainer` for per-prediction explainability
- Saves `models/model.pkl`, `models/explainer.pkl`, `models/metrics.json`,
  `models/feature_list.json`

Expect **AUC around 0.78–0.80** with the full multi-table feature set
(vs. ~0.75–0.77 using application-level fields alone) — the credit-history
aggregates (bureau, previous applications, payment lateness) are exactly
what closes that gap, which is *why* it's worth the extra pipeline step.
Top Kaggle solutions with hundreds of hand-tuned features across all
tables reach ~0.80; don't chase further than that with this feature set —
explain the tradeoff instead of over-fitting for a marginal gain.

---

## 7. Run the API locally

```bash
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive Swagger docs.

**Example 1 — existing customer (credit history looked up and included):**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "AMT_INCOME_TOTAL": 180000,
    "AMT_CREDIT": 500000,
    "AMT_ANNUITY": 25000,
    "DAYS_BIRTH": -14000,
    "DAYS_EMPLOYED": -2500,
    "DAYS_ID_PUBLISH": -3000,
    "DAYS_LAST_PHONE_CHANGE": -500,
    "CNT_CHILDREN": 1,
    "EXT_SOURCE_1": 0.55,
    "EXT_SOURCE_2": 0.60,
    "EXT_SOURCE_3": 0.50,
    "CODE_GENDER": "F",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y",
    "NAME_EDUCATION_TYPE": "Higher education",
    "credit_history": {
      "BUREAU_COUNT": 5,
      "BUREAU_ACTIVE_COUNT": 2,
      "BUREAU_CREDIT_SUM_DEBT_MEAN": 40000,
      "BUREAU_CREDIT_DAY_OVERDUE_MAX": 0,
      "PREV_APP_COUNT": 3,
      "PREV_APP_APPROVED_RATE": 0.8,
      "PREV_APP_REFUSED_RATE": 0.2,
      "INSTALLMENTS_LATE_RATE": 0.02,
      "CC_UTILIZATION_MEAN": 0.25,
      "CC_DPD_MEAN": 0
    }
  }'
```

**Example 2 — brand-new customer, no credit history at all:** just omit
`credit_history` entirely. The model falls back to
`HAS_BUREAU_HISTORY=0` / `HAS_PREVIOUS_APPLICATION=0` and training-median
values for the rest (see `models/feature_medians.json`) — it does *not*
default to zero, which would misleadingly read as "zero risk."

Response:

```json
{
  "probability_of_default": 0.084,
  "risk_score": 916,
  "decision": "APPROVE",
  "top_reasons": [
    {"feature": "BUREAU_CREDIT_SUM_DEBT_MEAN", "impact": -0.041},
    {"feature": "EXT_SOURCE_2", "impact": -0.031}
  ],
  "used_credit_history": true,
  "model_version": "2026-07-17"
}
```

`used_credit_history` tells you whether the caller supplied any real
history values or the response leaned entirely on median fallbacks —
useful for your own monitoring of how often you're scoring blind on new
customers.

---

## 8. Tests

```bash
pytest tests/ -v
```

Covers: health check, valid prediction request, malformed input rejected,
response schema is well-formed.

---


## 9. Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

---

## 10. CI (GitHub Actions)

`.github/workflows/ci.yml` runs on every push: installs deps, runs pytest,
builds the Docker image. Push to `main` and check the "Actions" tab on
GitHub — a green check next to your commits is itself a signal recruiters
notice.

---

## 11. Deploy (Render, free tier)

1. Push this repo to GitHub (done above).
2. On Render: New → Web Service → connect your repo.
3. Environment: Docker. Render auto-detects the `Dockerfile`.
4. Deploy. You'll get a public URL like `https://credit-risk-api.onrender.com`.
5. Put that URL in your README and LinkedIn post.

---

## 12. Monitoring / drift check

```bash
python -m src.monitor
```

Logs every `/predict` call to `data/logs/predictions.db` (SQLite).
`monitor.py` runs a Kolmogorov–Smirnov test per feature, comparing the
distribution of *live* requests against the *training* distribution, and
flags any feature whose distribution has drifted (p < 0.05). This is the
piece almost no student portfolio has — it's exactly what's asked about in
MLOps interviews ("how do you know your model still works after 3 months?").

---

## Next steps (to extend further)

- Swap the "top reasons" `/predict` output for a full SHAP force-plot image endpoint.
- Add a simple Streamlit front-end so non-technical reviewers can try it
  without hitting the API directly (and can toggle the credit_history
  block on/off to see the cold-start behavior).
- Add MLflow for experiment tracking if you retrain multiple times.
- Feed `src/build_features.py`'s full aggregate set (not just the curated
  10) into a feature-importance pass, and promote any more of them into
  the API schema if they earn their place.
- Add a real feature store lookup (`GET /applicant/{id}/history`) backed
  by the feature store parquet, so the API can resolve `credit_history`
  from an ID instead of requiring the caller to pass it — closer to how
  this would work in production.

## Repo structure

```
credit-risk-mlops/
├── app/
│   ├── main.py          FastAPI app
│   └── schemas.py       Pydantic request/response models
├── src/
│   ├── build_features.py   multi-table feature store builder
│   ├── data_processing.py  joins application data + feature store
│   ├── train.py
│   └── monitor.py
├── tests/
│   └── test_api.py
├── models/               generated: model.pkl, explainer.pkl, metrics.json,
│                         feature_list.json, feature_medians.json
├── data/                 generated: raw/, processed/ (incl. feature_store.parquet), logs/
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
└── README.md
```
