"""
Feature drift monitor: compares the distribution of features in *live*
logged prediction requests against the *training* distribution, using a
Kolmogorov-Smirnov test per numeric feature.

This answers the question every MLOps interview asks: "how would you know
if your model quietly stopped working?" A model's accuracy doesn't degrade
gracefully in production — it silently drifts as the population it sees
changes, and you only find out from monitoring like this (or a slow
realization that your default rate crept up).

Run as: python -m src.monitor
"""
import json
import sqlite3
import pandas as pd
from scipy.stats import ks_2samp

LOG_DB_PATH = "data/logs/predictions.db"
PROCESSED_PATH = "data/processed/train_processed.parquet"
FEATURE_LIST_PATH = "models/feature_list.json"
ALPHA = 0.05  # significance threshold for flagging drift


def load_live_requests() -> pd.DataFrame:
    conn = sqlite3.connect(LOG_DB_PATH)
    rows = conn.execute("SELECT features_json FROM predictions").fetchall()
    conn.close()
    records = [json.loads(r[0]) for r in rows]
    return pd.DataFrame(records)


def main():
    with open(FEATURE_LIST_PATH) as f:
        feat_cfg = json.load(f)
    numeric_features = [f for f in feat_cfg["numeric_features"] if f != "DAYS_EMPLOYED_ANOM"]

    train_df = pd.read_parquet(PROCESSED_PATH)
    live_df = load_live_requests()

    if len(live_df) < 30:
        print(f"Only {len(live_df)} logged predictions — need at least ~30 for a "
              f"meaningful drift check. Run more /predict requests first.")
        return

    print(f"Comparing {len(live_df)} live requests against {len(train_df)} training rows.\n")
    flagged = []
    for feat in numeric_features:
        if feat not in live_df.columns:
            continue
        stat, p_value = ks_2samp(train_df[feat].dropna(), live_df[feat].dropna())
        status = "DRIFT DETECTED" if p_value < ALPHA else "ok"
        if p_value < ALPHA:
            flagged.append(feat)
        print(f"  {feat:<28} KS={stat:.4f}  p={p_value:.4f}  [{status}]")

    print()
    if flagged:
        print(f"⚠️  {len(flagged)} feature(s) show significant drift: {', '.join(flagged)}")
        print("Consider investigating the applicant population and/or retraining.")
    else:
        print("✅ No significant drift detected.")


if __name__ == "__main__":
    main()
