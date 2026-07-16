"""
Train an XGBoost credit-default classifier on the processed Home Credit
data, evaluate with AUC + KS statistic, fit a SHAP explainer, and save all
artifacts the API needs.

Run as: python -m src.train
"""
import json
import joblib
import numpy as np
import pandas as pd
import shap
from datetime import date
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

PROCESSED_PATH = "data/processed/train_processed.parquet"
FEATURE_LIST_PATH = "models/feature_list.json"
MODEL_PATH = "models/model.pkl"
ENCODER_PATH = "models/encoder.pkl"
EXPLAINER_PATH = "models/explainer.pkl"
METRICS_PATH = "models/metrics.json"


def ks_statistic(y_true, y_prob):
    """KS statistic: max separation between the cumulative distributions of
    predicted scores for defaulters vs non-defaulters. This — not accuracy —
    is the metric credit risk teams actually report, because default rates
    are heavily imbalanced (~8% here)."""
    good = y_prob[y_true == 0]
    bad = y_prob[y_true == 1]
    return ks_2samp(good, bad).statistic


def main():
    with open(FEATURE_LIST_PATH) as f:
        feat_cfg = json.load(f)
    numeric_features = feat_cfg["numeric_features"]
    categorical_features = feat_cfg["categorical_features"]
    target = feat_cfg["target"]

    df = pd.read_parquet(PROCESSED_PATH)

    X_num = df[numeric_features].values
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = encoder.fit_transform(df[categorical_features])
    cat_feature_names = encoder.get_feature_names_out(categorical_features).tolist()

    X = np.hstack([X_num, X_cat])
    all_feature_names = numeric_features + cat_feature_names
    y = df[target].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # scale_pos_weight compensates for the ~8% default rate imbalance —
    # without it the model just predicts "no default" for everyone.
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_prob)
    ks = ks_statistic(y_val, val_prob)

    print(f"Validation AUC: {auc:.4f}")
    print(f"Validation KS:  {ks:.4f}")

    explainer = shap.TreeExplainer(model)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    joblib.dump(explainer, EXPLAINER_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "auc": round(float(auc), 4),
                "ks_statistic": round(float(ks), 4),
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "default_rate": round(float(y.mean()), 4),
                "trained_on": str(date.today()),
                "feature_names": all_feature_names,
            },
            f,
            indent=2,
        )

    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved encoder -> {ENCODER_PATH}")
    print(f"Saved explainer -> {EXPLAINER_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
