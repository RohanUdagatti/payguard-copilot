"""Benchmark XGBoost fraud detection pipeline on Kaggle Credit Card Fraud dataset.

If data/raw_kaggle/creditcard.csv exists:
- Loads the dataset.
- Evaluates time-based or chronological split.
- Trains the XGBoost pipeline.
- Reports PR-AUC and ROC-AUC.
- Notes that features V1-V28 are PCA-anonymized (making SHAP non-interpretable).

If the file is missing:
- Prints instructions on how to download it from Kaggle.
- Exits cleanly with code 0 (no download performed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

KAGGLE_PATH = Path("data/raw_kaggle/creditcard.csv")


def main() -> int:
    """Entry point for Kaggle credit card fraud benchmark."""
    print("=" * 78)
    print("KAGGLE CREDIT CARD FRAUD DETECTION BENCHMARK")
    print("=" * 78)

    if not KAGGLE_PATH.exists():
        print(f"Dataset file '{KAGGLE_PATH}' was not found.")
        print("\nTo run this benchmark, download the dataset from Kaggle:")
        print("  1. Ensure you have the Kaggle CLI configured (~/.kaggle/kaggle.json)")
        print("  2. Create directory and download the dataset:")
        print("       mkdir -p data/raw_kaggle")
        cmd = (
            "       kaggle datasets download -d mlg-ulb/creditcardfraud "
            "-p data/raw_kaggle/ --unzip"
        )
        print(cmd)
        print("  3. Verify the file exists at:")
        print("       data/raw_kaggle/creditcard.csv")
        print("  4. Re-run this script: make ml-benchmark-kaggle")
        print("\nExiting cleanly without downloading.")
        return 0

    print(f"Loading Kaggle dataset from {KAGGLE_PATH}...")
    df = pd.read_csv(KAGGLE_PATH)
    print(f"Loaded {len(df):,} rows. Target fraud count: {df['Class'].sum():,}")

    # Dataset contains: Time, V1-V28 (PCA components), Amount, Class
    # Time is chronological elapsed seconds from first transaction
    df = df.sort_values(by="Time").reset_index(drop=True)

    n = len(df)
    n_train = int(0.70 * n)
    n_val = int(0.85 * n)

    feature_cols = [c for c in df.columns if c not in ("Class", "Time")]

    X_train = df[feature_cols].iloc[:n_train]
    y_train = df["Class"].iloc[:n_train].to_numpy()

    X_val = df[feature_cols].iloc[n_train:n_val]
    y_val = df["Class"].iloc[n_train:n_val].to_numpy()

    X_test = df[feature_cols].iloc[n_val:]
    y_test = df["Class"].iloc[n_val:].to_numpy()

    print(f"\nTrain rows: {len(X_train):,}, Positives: {int(y_train.sum())}")
    print(f"Val rows:   {len(X_val):,}, Positives: {int(y_val.sum())}")
    print(f"Test rows:  {len(X_test):,}, Positives: {int(y_test.sum())}")

    print("\nTraining XGBoost on Kaggle features...")
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    test_preds = clf.predict_proba(X_test)[:, 1]
    prauc = float(average_precision_score(y_test, test_preds))
    rocauc = float(roc_auc_score(y_test, test_preds))

    print("\n" + "=" * 78)
    print("KAGGLE BENCHMARK RESULTS")
    print("=" * 78)
    print(f"Test PR-AUC:  {prauc:.4f}")
    print(f"Test ROC-AUC: {rocauc:.4f}")
    print("-" * 78)
    print(
        "NOTE: Features V1-V28 are PCA-transformed components (anonymized for privacy)."
    )
    print("While XGBoost achieves high precision-recall, individual SHAP feature names")
    print(
        "correspond to principal component vectors and cannot be interpreted as "
        "domain attributes."
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
