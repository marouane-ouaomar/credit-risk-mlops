"""
Builds a "credit history feature store" by aggregating the five auxiliary
Home Credit tables (bureau, bureau_balance, previous_application,
POS_CASH_balance, installments_payments, credit_card_balance) up to one row
per SK_ID_CURR — the same grain as application_train/application_test.

This is what a real bank's feature pipeline looks like: raw transactional
tables get aggregated offline into a feature store, and the model (and the
serving API) consume the aggregates, not the raw tables. A loan officer
doesn't hand-type "average days overdue across 14 prior loans" — that
number is looked up.

Run as: python -m src.build_features
(run BEFORE src/data_processing.py)

Output: data/processed/feature_store.parquet, indexed by SK_ID_CURR.

NOTE on scale: bureau.csv, POS_CASH_balance.csv, installments_payments.csv,
and credit_card_balance.csv are large (hundreds of MB to a few GB combined).
This script reads only the columns it needs (usecols) and aggregates with
groupby, which keeps memory reasonable, but expect this step to take a few
minutes on the full dataset — that's normal, not a bug.
"""
import os
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
OUT_PATH = "data/processed/feature_store.parquet"


def _read_if_exists(filename, usecols=None):
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        print(f"  (skipping {filename} — not found in {RAW_DIR})")
        return None
    return pd.read_csv(path, usecols=usecols)


def build_bureau_features() -> pd.DataFrame:
    """Aggregates bureau.csv (prior credits reported by other institutions)
    and bureau_balance.csv (monthly status of those credits) up to one row
    per SK_ID_CURR."""
    bureau = _read_if_exists(
        "bureau.csv",
        usecols=[
            "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT",
            "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_OVERDUE",
        ],
    )
    if bureau is None:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    bureau["IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)

    # Optional: bureau_balance.csv gives monthly delinquency status per
    # SK_ID_BUREAU. STATUS: '0'=no DPD, '1'-'5'=increasing delinquency,
    # 'C'=closed, 'X'=unknown. We compute, per SK_ID_BUREAU, the share of
    # months with any reported delinquency, then roll that up to SK_ID_CURR.
    balance = _read_if_exists("bureau_balance.csv", usecols=["SK_ID_BUREAU", "STATUS"])
    if balance is not None:
        balance["IS_DPD"] = balance["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)
        bureau_balance_agg = balance.groupby("SK_ID_BUREAU")["IS_DPD"].mean().rename(
            "BUREAU_BALANCE_DPD_RATE"
        )
        bureau = bureau.merge(bureau_balance_agg, on="SK_ID_BUREAU", how="left")
    else:
        bureau["BUREAU_BALANCE_DPD_RATE"] = np.nan

    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BUREAU_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
        BUREAU_CREDIT_SUM_DEBT_MEAN=("AMT_CREDIT_SUM_DEBT", "mean"),
        BUREAU_CREDIT_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_CREDIT_SUM_OVERDUE_MEAN=("AMT_CREDIT_SUM_OVERDUE", "mean"),
        BUREAU_BALANCE_DPD_RATE_MEAN=("BUREAU_BALANCE_DPD_RATE", "mean"),
    ).reset_index()

    return agg


def build_previous_application_features() -> pd.DataFrame:
    """Aggregates previous_application.csv — this applicant's prior loan
    applications with Home Credit itself (as opposed to bureau.csv, which
    is prior credit with OTHER institutions)."""
    prev = _read_if_exists(
        "previous_application.csv",
        usecols=[
            "SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS",
            "AMT_CREDIT", "AMT_ANNUITY", "DAYS_DECISION",
        ],
    )
    if prev is None:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    prev["IS_APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    prev["IS_REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)

    agg = prev.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APP_APPROVED_RATE=("IS_APPROVED", "mean"),
        PREV_APP_REFUSED_RATE=("IS_REFUSED", "mean"),
        PREV_APP_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_APP_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        PREV_APP_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
    ).reset_index()

    return agg


def build_pos_cash_features() -> pd.DataFrame:
    """Aggregates POS_CASH_balance.csv — monthly balance snapshots of this
    applicant's point-of-sale and cash loans."""
    pos = _read_if_exists(
        "POS_CASH_balance.csv",
        usecols=["SK_ID_CURR", "SK_ID_PREV", "SK_DPD", "SK_DPD_DEF"],
    )
    if pos is None:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    agg = pos.groupby("SK_ID_CURR").agg(
        POS_CASH_COUNT=("SK_ID_PREV", "count"),
        POS_CASH_DPD_MEAN=("SK_DPD", "mean"),
        POS_CASH_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
    ).reset_index()

    return agg


def build_installments_features() -> pd.DataFrame:
    """Aggregates installments_payments.csv — whether/how late this
    applicant paid past installments. Lateness and underpayment on past
    installments is one of the strongest real-world default predictors."""
    inst = _read_if_exists(
        "installments_payments.csv",
        usecols=[
            "SK_ID_CURR", "SK_ID_PREV", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT", "AMT_PAYMENT",
        ],
    )
    if inst is None:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    inst["IS_LATE"] = (inst["DAYS_ENTRY_PAYMENT"] > inst["DAYS_INSTALMENT"]).astype(int)
    inst["PAYMENT_DIFF"] = inst["AMT_PAYMENT"] - inst["AMT_INSTALMENT"]

    agg = inst.groupby("SK_ID_CURR").agg(
        INSTALLMENTS_COUNT=("SK_ID_PREV", "count"),
        INSTALLMENTS_LATE_RATE=("IS_LATE", "mean"),
        INSTALLMENTS_PAYMENT_DIFF_MEAN=("PAYMENT_DIFF", "mean"),
    ).reset_index()

    return agg


def build_credit_card_features() -> pd.DataFrame:
    """Aggregates credit_card_balance.csv — monthly balance/limit/DPD for
    this applicant's revolving credit card products with Home Credit."""
    cc = _read_if_exists(
        "credit_card_balance.csv",
        usecols=["SK_ID_CURR", "SK_ID_PREV", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD"],
    )
    if cc is None:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    cc["UTILIZATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)

    agg = cc.groupby("SK_ID_CURR").agg(
        CC_COUNT=("SK_ID_PREV", "count"),
        CC_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_UTILIZATION_MEAN=("UTILIZATION", "mean"),
        CC_DPD_MEAN=("SK_DPD", "mean"),
    ).reset_index()

    return agg


def main():
    os.makedirs("data/processed", exist_ok=True)

    print("Building bureau features...")
    bureau_feats = build_bureau_features()
    print("Building previous_application features...")
    prev_feats = build_previous_application_features()
    print("Building POS_CASH features...")
    pos_feats = build_pos_cash_features()
    print("Building installments features...")
    inst_feats = build_installments_features()
    print("Building credit_card features...")
    cc_feats = build_credit_card_features()

    # Outer-merge everything on SK_ID_CURR — an applicant may be missing
    # from any of these tables (e.g. a brand-new customer with no bureau
    # history), which is exactly the "cold start" case we want to preserve,
    # not silently drop.
    feature_frames = [bureau_feats, prev_feats, pos_feats, inst_feats, cc_feats]
    feature_store = feature_frames[0]
    for frame in feature_frames[1:]:
        feature_store = feature_store.merge(frame, on="SK_ID_CURR", how="outer")

    # Cold-start flags — did we find ANY record for this applicant in each
    # source table? These end up being some of the most important features:
    # having no credit history at all is itself informative.
    feature_store["HAS_BUREAU_HISTORY"] = feature_store["BUREAU_COUNT"].notna().astype(int)
    feature_store["HAS_PREVIOUS_APPLICATION"] = feature_store["PREV_APP_COUNT"].notna().astype(int)

    feature_store.to_parquet(OUT_PATH, index=False)
    print(f"\nFeature store built: {feature_store.shape[0]} applicants, "
          f"{feature_store.shape[1] - 1} engineered features -> {OUT_PATH}")


if __name__ == "__main__":
    main()
