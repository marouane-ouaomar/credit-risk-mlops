# Credit Risk Scoring API

End-to-end credit default prediction system — from raw multi-table data to a
deployed, explainable, monitored REST API.

**[Live demo →](https://credit-risk-mlops-production.up.railway.app/docs)**

![CI](https://github.com/marouane-ouaomar/credit-risk-mlops/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What this is

Most ML portfolio projects stop at a notebook with an accuracy score. This
one doesn't: it's a small but complete production system — data pipeline,
model, API, container, CI, and monitoring — built on the
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
dataset (307k applicants, 6 relational tables, ~10% default rate).

Given an applicant's profile — and optionally their credit history from
other institutions and prior loans — the API returns a default probability,
an approve/review/reject decision, and a SHAP-based explanation of *why*.

## Results

| Metric | Value |
|---|---|
| AUC (validation) | 0.78–0.80 |
| KS statistic | reported alongside AUC — the metric credit risk teams actually use, given the ~10% class imbalance |
| Features | 11 application-level + 10 aggregated credit-history features, selected from ~150 engineered candidates for explainability |

## Highlights

- **Multi-table feature engineering** — aggregates 5 auxiliary tables
  (bureau history, prior applications, POS/cash loans, installment
  payments, credit cards) into a proper feature store, not just the flat
  application table.
- **Explainable by design** — every prediction ships with its top SHAP
  drivers, and the feature set itself is curated to stay defensible to a
  regulator or loan officer, not just accuracy-maximizing.
- **Cold-start handling** — new applicants with no credit history are
  detected explicitly (`HAS_BUREAU_HISTORY` flag) and scored on
  training-median fallbacks instead of misleading zeros.
- **Actually deployed** — containerized with Docker, live on Railway, not
  just runnable on `localhost`.
- **CI-tested** — GitHub Actions runs the full pipeline (synthetic data →
  feature store → training → API tests) on every push.
- **Drift monitoring** — a KS-test script flags feature drift between
  training and live traffic, the question every MLOps interview asks
  ("how do you know your model still works after 3 months?").

## Architecture

```
                    ┌─────────────────────────┐
Kaggle tables ────▶ │ src/build_features.py   │ ──▶ feature_store.parquet
(bureau, prev_app,  │ (multi-table aggregation)│
 pos_cash, install-  └─────────────────────────┘
 ments, cc_balance)              │
                                 ▼
application_train.csv ──▶ src/data_processing.py ──▶ processed data + medians
                                 │
                                 ▼
                          src/train.py
                     (XGBoost + SHAP + AUC/KS)
                                 │
                                 ▼
                     models/*.pkl + metrics.json
                                 │
                                 ▼
               app/main.py (FastAPI) ──▶ Docker ──▶ Railway (live URL)
                                 │
                                 ▼
                    src/monitor.py (drift detection)
```

## Tech stack

| Layer | Tools |
|---|---|
| Modeling | XGBoost, scikit-learn, SHAP |
| Data | pandas, PyArrow (Parquet) |
| Serving | FastAPI, Pydantic, Uvicorn |
| Testing | pytest, httpx |
| Ops | Docker, GitHub Actions, Railway |
| Monitoring | SciPy (KS-test drift detection), SQLite (request logging) |

---

## Quickstart

```bash
git clone https://github.com/marouane-ouaomar/credit-risk-mlops.git
cd credit-risk-mlops
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# get data: https://www.kaggle.com/c/home-credit-default-risk (join comp, download to data/raw/)
python -m src.build_features      # aggregate the 5 auxiliary tables
python -m src.data_processing     # join + clean
python -m src.train                # AUC ~0.78-0.80

uvicorn app.main:app --reload --port 8000
# → http://localhost:8000/docs
```

## API usage

**Existing customer** (credit history known — normally resolved from a
bureau/internal lookup, not typed by hand):

```bash
curl -X POST https://credit-risk-mlops-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 500000, "AMT_ANNUITY": 25000,
    "DAYS_BIRTH": -14000, "DAYS_EMPLOYED": -2500, "DAYS_ID_PUBLISH": -3000,
    "DAYS_LAST_PHONE_CHANGE": -500, "CNT_CHILDREN": 1,
    "EXT_SOURCE_1": 0.55, "EXT_SOURCE_2": 0.60, "EXT_SOURCE_3": 0.50,
    "CODE_GENDER": "F", "FLAG_OWN_CAR": "N", "FLAG_OWN_REALTY": "Y",
    "NAME_EDUCATION_TYPE": "Higher education",
    "credit_history": {
      "BUREAU_COUNT": 5, "BUREAU_ACTIVE_COUNT": 2,
      "BUREAU_CREDIT_SUM_DEBT_MEAN": 40000, "BUREAU_CREDIT_DAY_OVERDUE_MAX": 0,
      "PREV_APP_COUNT": 3, "PREV_APP_APPROVED_RATE": 0.8, "PREV_APP_REFUSED_RATE": 0.2,
      "INSTALLMENTS_LATE_RATE": 0.02, "CC_UTILIZATION_MEAN": 0.25, "CC_DPD_MEAN": 0
    }
  }'
```

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

**New customer, no history** — omit `credit_history` entirely; the model
falls back to training medians and flags the cold-start case internally
rather than defaulting to zero (which would misleadingly read as
"zero risk").

Other endpoints: `GET /health`, `GET /model-info`. Full interactive docs
at [`/docs`](https://credit-risk-mlops-production.up.railway.app/docs).

## Project structure

```
credit-risk-mlops/
├── app/
│   ├── main.py              FastAPI app
│   └── schemas.py            Pydantic request/response models
├── src/
│   ├── build_features.py     multi-table feature store builder
│   ├── data_processing.py    joins application data + feature store
│   ├── train.py               XGBoost + SHAP + AUC/KS
│   └── monitor.py             drift detection
├── tests/test_api.py
├── models/                    model.pkl, encoder.pkl, explainer.pkl,
│                               metrics.json, feature_list.json, feature_medians.json
├── data/                       raw/, processed/, logs/
├── .github/workflows/ci.yml
├── Dockerfile
└── requirements.txt
```

## Design decisions

A few choices worth calling out (also useful shorthand for explaining this
project in an interview):

- **Curated features over "throw everything at XGBoost."** ~150 features
  are engineerable across all 6 tables; the model uses 21. Each one has a
  clear business meaning a loan officer or regulator could understand —
  a defensible tradeoff, not a limitation.
- **Cold start is a first-class case, not an edge case.** A meaningful
  share of applicants have no bureau or prior-application history.
  `HAS_BUREAU_HISTORY` / `HAS_PREVIOUS_APPLICATION` flags plus
  median-fallback imputation let the model treat "no history" as its own
  signal instead of conflating it with "history exists and is good."
- **Explainability is part of the response, not an afterthought.** SHAP
  values ship with every prediction — real risk models are frequently
  required to justify individual decisions.
- **Model artifacts are committed to git**, not pulled from a registry at
  runtime — the right call at this scale (single model, infrequent
  retraining); a real production system would use MLflow/S3 instead, and
  that tradeoff is worth naming explicitly rather than over-engineering it.

## Testing & CI

```bash
pytest tests/ -v
```

GitHub Actions (`.github/workflows/ci.yml`) runs the full pipeline —
synthetic multi-table data → feature store → training → API tests — on
every push, so CI doesn't depend on Kaggle credentials or multi-GB data
being available in the runner.

## Roadmap

- Full SHAP force-plot endpoint (image, not just top-5 JSON)
- Streamlit front-end for non-technical reviewers
- MLflow experiment tracking
- `GET /applicant/{id}/history` — resolve `credit_history` server-side from
  the feature store instead of requiring the caller to pass it

## License

MIT
