"""
Feature engineering for the Home Credit Default Risk dataset.

Reads data/raw/application_train.csv, engineers a curated, explainable
feature set, handles missing values, and writes a processed parquet file
that src/train.py consumes.

Run as: python -m src.data_processing
"""
import os
import json
import pandas as pd
import numpy as np

RAW_PATH = "data/raw/application_train.csv"
PROCESSED_DIR = "data/processed"
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "train_processed.parquet")
FEATURE_LIST_PATH = "models/feature_list.json"

TARGET = "TARGET"

# Curated, explainable feature set. Chosen because each one has a clear,
# defensible business meaning a loan officer or regulator could understand
# — deliberately not just "throw every column at XGBoost".
NUMERIC_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",            # negative = days before application; age proxy
    "DAYS_EMPLOYED",         # negative = days employed; has known anomaly (365243)
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "CNT_CHILDREN",
    "EXT_SOURCE_1",          # normalized external credit bureau score
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

CATEGORICAL_FEATURES = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_EDUCATION_TYPE",
]

FEATURE_LIST = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _clean_days_employed(df: pd.DataFrame) -> pd.DataFrame:
    """Home Credit has a well-known anomaly: DAYS_EMPLOYED == 365243 means
    'not employed / pensioner', encoded as a sentinel instead of NaN.
    Flagging + fixing this is a classic "did you actually look at your data"
    signal in interviews."""
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    return df


def load_and_process(raw_path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path, usecols=[TARGET, "SK_ID_CURR"] + FEATURE_LIST)
    df = _clean_days_employed(df)

    # Numeric missing values: median impute (simple, defensible baseline —
    # note in your README that a real system would use a fitted imputer
    # saved alongside the model to avoid train/serve skew).
    for col in NUMERIC_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical missing values: explicit "Unknown" category.
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str)

    return df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = load_and_process()

    df.to_parquet(PROCESSED_PATH, index=False)

    final_features = FEATURE_LIST + ["DAYS_EMPLOYED_ANOM"]
    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(
            {
                "numeric_features": NUMERIC_FEATURES + ["DAYS_EMPLOYED_ANOM"],
                "categorical_features": CATEGORICAL_FEATURES,
                "all_features": final_features,
                "target": TARGET,
            },
            f,
            indent=2,
        )

    print(f"Processed {len(df)} rows -> {PROCESSED_PATH}")
    print(f"Default rate: {df[TARGET].mean():.2%}")
    print(f"Feature list saved -> {FEATURE_LIST_PATH}")


if __name__ == "__main__":
    main()
