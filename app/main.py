import json
import sqlite3
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (
    ApplicantFeatures,
    PredictionResponse,
    ReasonCode,
    HealthResponse,
    ModelInfoResponse,
)

MODEL_PATH = "models/model.pkl"
ENCODER_PATH = "models/encoder.pkl"
EXPLAINER_PATH = "models/explainer.pkl"
METRICS_PATH = "models/metrics.json"
FEATURE_LIST_PATH = "models/feature_list.json"
LOG_DB_PATH = "data/logs/predictions.db"

_model = None
_encoder = None
_explainer = None
_metrics = None
_feature_cfg = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _encoder, _explainer, _metrics, _feature_cfg
    _model = joblib.load(MODEL_PATH)
    _encoder = joblib.load(ENCODER_PATH)
    _explainer = joblib.load(EXPLAINER_PATH)
    with open(METRICS_PATH) as f:
        _metrics = json.load(f)
    with open(FEATURE_LIST_PATH) as f:
        _feature_cfg = json.load(f)

    os.makedirs(os.path.dirname(LOG_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(LOG_DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS predictions (
            timestamp TEXT,
            features_json TEXT,
            probability_of_default REAL
        )"""
    )
    conn.commit()
    conn.close()

    yield  # app runs here

    # (no teardown needed — nothing to release)


app = FastAPI(
    title="Credit Risk Scoring API",
    description="Predicts probability of loan default (Home Credit Default Risk dataset).",
    version="1.0.0",
    lifespan=lifespan,
)


def _build_feature_row(applicant: ApplicantFeatures) -> pd.DataFrame:
    data = applicant.model_dump()
    data["DAYS_EMPLOYED_ANOM"] = 1 if data["DAYS_EMPLOYED"] == 365243 else 0
    if data["DAYS_EMPLOYED"] == 365243:
        data["DAYS_EMPLOYED"] = np.nan  # will be imputed to 0 contribution via median below

    numeric_features = _feature_cfg["numeric_features"]
    categorical_features = _feature_cfg["categorical_features"]

    row = {}
    for feat in numeric_features:
        val = data.get(feat, 0)
        row[feat] = 0 if (val is None or (isinstance(val, float) and np.isnan(val))) else val
    for feat in categorical_features:
        row[feat] = data.get(feat, "Unknown")

    return pd.DataFrame([row])


def _log_prediction(applicant: ApplicantFeatures, prob: float):
    try:
        conn = sqlite3.connect(LOG_DB_PATH)
        conn.execute(
            "INSERT INTO predictions VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), json.dumps(applicant.model_dump()), float(prob)),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        # Logging should never break a prediction response.
        print(f"WARNING: failed to log prediction: {e}")


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return {
        "model_version": _metrics["trained_on"],
        "auc": _metrics["auc"],
        "ks_statistic": _metrics["ks_statistic"],
        "trained_on": _metrics["trained_on"],
        "n_train": _metrics["n_train"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantFeatures):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    row = _build_feature_row(applicant)
    numeric_features = _feature_cfg["numeric_features"]
    categorical_features = _feature_cfg["categorical_features"]

    X_num = row[numeric_features].values
    X_cat = _encoder.transform(row[categorical_features])
    X = np.hstack([X_num, X_cat])

    prob = float(_model.predict_proba(X)[:, 1][0])

    # Simple business decision policy on top of the raw probability —
    # a real deployment would tune these thresholds against a cost matrix
    # (cost of a bad loan vs. cost of rejecting a good customer).
    if prob < 0.10:
        decision = "APPROVE"
    elif prob < 0.25:
        decision = "REVIEW"
    else:
        decision = "REJECT"

    # FICO-like scale: higher score = lower risk. Purely for presentation.
    risk_score = int(300 + (1 - prob) * 550)

    shap_values = _explainer.shap_values(X)
    cat_names = _encoder.get_feature_names_out(categorical_features).tolist()
    all_names = numeric_features + cat_names
    contributions = list(zip(all_names, shap_values[0].tolist()))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_reasons = [
        ReasonCode(feature=name, impact=round(val, 4))
        for name, val in contributions[:5]
    ]

    _log_prediction(applicant, prob)

    return PredictionResponse(
        probability_of_default=round(prob, 4),
        risk_score=risk_score,
        decision=decision,
        top_reasons=top_reasons,
        model_version=_metrics["trained_on"],
    )
