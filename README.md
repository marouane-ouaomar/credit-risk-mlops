# Credit Risk Scoring — End-to-End MLOps Project

Predicts probability of loan default from applicant data, using the
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset,
served as a REST API with explainability, containerized, and CI-tested.

This is not "train a model in a notebook." It's a small but complete
production pipeline: **data → model → API → container → CI → monitoring.**

---

## Architecture

```
Kaggle data ──▶ src/data_processing.py ──▶ processed parquet
                                                  │
                                                  ▼
                                        src/train.py (XGBoost + SHAP)
                                                  │
                                                  ▼
                                   models/model.pkl + models/explainer.pkl
                                                  │
                                                  ▼
                              app/main.py (FastAPI) ──▶ Docker image
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

You only need `application_train.csv` and `application_test.csv` for this
build. The other tables (`bureau.csv`, `previous_application.csv`, etc.)
are there if you want to extend the project later (see "Next steps").

---

## 4. Feature engineering + processing

```bash
python -m src.data_processing
```

Reads `data/raw/application_train.csv`, engineers a curated feature set
(see `FEATURE_LIST` in `src/data_processing.py`), handles missing values,
and writes `data/processed/train_processed.parquet`.

**Why a curated feature set, not all ~220 raw columns?**
Real risk models are often *required* to be explainable to regulators and
loan officers — a 220-feature black box fails that test. Fifteen
well-chosen, explainable features (income, credit amount, employment
length, external bureau scores, etc.) is closer to how a real scorecard is
built, and it keeps the API schema sane. This is a deliberate modeling
decision — say so explicitly in interviews.

---

## 5. Train the model

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

Expect **AUC around 0.75–0.77** on the curated feature set — a realistic,
defensible number (top Kaggle solutions with hundreds of features across
all tables reach ~0.80). Don't chase 0.80 with 15 features; explain *why*
your number is what it is instead.

---

## 6. Run the API locally

```bash
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive Swagger docs.

Example request:

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
    "NAME_EDUCATION_TYPE": "Higher education"
  }'
```

Response:

```json
{
  "probability_of_default": 0.084,
  "risk_score": 916,
  "decision": "APPROVE",
  "top_reasons": [
    {"feature": "EXT_SOURCE_2", "impact": -0.031},
    {"feature": "AMT_CREDIT", "impact": 0.014}
  ],
  "model_version": "2026-07-16"
}
```

---

## 7. Tests

```bash
pytest tests/ -v
```

Covers: health check, valid prediction request, malformed input rejected,
response schema is well-formed.

---

## 8. Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

---

## 9. CI (GitHub Actions)

`.github/workflows/ci.yml` runs on every push: installs deps, runs pytest,
builds the Docker image. Push to `main` and check the "Actions" tab on
GitHub — a green check next to your commits is itself a signal recruiters
notice.

---

## 10. Deploy (Render, free tier)

1. Push this repo to GitHub (done above).
2. On Render: New → Web Service → connect your repo.
3. Environment: Docker. Render auto-detects the `Dockerfile`.
4. Deploy. You'll get a public URL like `https://credit-risk-api.onrender.com`.
5. Put that URL in your README and LinkedIn post.

---

## 11. Monitoring / drift check

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

- Bring in `bureau.csv` and `previous_application.csv` for a true
  multi-table feature engineering pipeline (shows data engineering depth).
- Swap the "top reasons" `/predict` output for a full SHAP force-plot image endpoint.
- Add a simple Streamlit front-end so non-technical reviewers can try it
  without hitting the API directly.
- Add MLflow for experiment tracking if you retrain multiple times.

## Repo structure

```
credit-risk-mlops/
├── app/
│   ├── main.py          FastAPI app
│   └── schemas.py       Pydantic request/response models
├── src/
│   ├── data_processing.py
│   ├── train.py
│   └── monitor.py
├── tests/
│   └── test_api.py
├── models/               generated: model.pkl, explainer.pkl, metrics.json
├── data/                 generated: raw/, processed/, logs/
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
└── README.md
```
