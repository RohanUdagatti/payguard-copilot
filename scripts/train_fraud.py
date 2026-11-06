"""Train, calibrate, evaluate, and serialize the PayGuard fraud model.

Outputs:
- Split date ranges and positive counts per split
- Baseline vs XGBoost table with PR-AUC + 95% bootstrap CI, ROC-AUC + CI, Brier
- Chosen threshold
- Test confusion matrix and financial cost comparison vs alternatives
- Top-10 global SHAP features (in log-odds)
"""

from __future__ import annotations

import argparse
import sys
import time

from payguard.config import get_settings
from payguard.ml.pipeline import (
    load_dataset_from_db,
    prepare_dataset_splits,
    train_and_evaluate_fraud_pipeline,
)


def format_currency(amount: float) -> str:
    """Format floating point numbers as USD currency string."""
    return f"${amount:,.2f}"


def main() -> int:
    """Main CLI entrypoint for fraud model training."""
    parser = argparse.ArgumentParser(description="Train PayGuard Fraud XGBoost Model")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on rows loaded from database (for rapid testing)",
    )
    args = parser.parse_args()

    settings = get_settings()

    print("=" * 78)
    print("PAYGUARD FRAUD DETECTION MODEL TRAINING (PHASE 5)")
    print("=" * 78)
    print("DISCLAIMER: All results are computed on synthetic benchmark data.")
    print("=" * 78)

    t0 = time.perf_counter()
    print("\n[1/4] Loading transactions from PostgreSQL...")
    raw_df = load_dataset_from_db(settings, limit=args.limit)
    print(f"Loaded {len(raw_df):,} records in {time.perf_counter() - t0:.2f}s.")

    t1 = time.perf_counter()
    print("\n[2/4] Computing point-in-time features and creating time splits...")
    splits = prepare_dataset_splits(raw_df)
    print(f"Features and splits prepared in {time.perf_counter() - t1:.2f}s.")

    # Report Split Date Ranges and Positives
    print("\n" + "-" * 78)
    print("DATASET TIME SPLITS & POSITIVE COUNTS")
    print("-" * 78)
    split_info = [
        ("Train (70%)", splits.train_df),
        ("Val A (7.5%)", splits.val_a_df),
        ("Val B (7.5%)", splits.val_b_df),
        ("Test (15%)", splits.test_df),
    ]
    hdr = (
        f"{'Split':<15} {'Total Rows':>12} {'Positives':>12} "
        f"{'Fraud Rate':>12} {'Date Range':>28}"
    )
    print(hdr)
    print("-" * 78)
    for name, s_df in split_info:
        total = len(s_df)
        pos = int(s_df["is_fraud"].sum())
        rate = (pos / total * 100) if total > 0 else 0.0
        min_ts = str(s_df["timestamp"].min())[:10]
        max_ts = str(s_df["timestamp"].max())[:10]
        date_range = f"{min_ts} to {max_ts}"
        print(f"{name:<15} {total:>12,} {pos:>12,} {rate:>11.3f}% {date_range:>28}")
    print("-" * 78)

    print("\n[3/4] Training models, fitting calibration, and optimizing threshold...")
    t2 = time.perf_counter()
    metadata = train_and_evaluate_fraud_pipeline(splits, settings)
    print(f"Training and evaluation completed in {time.perf_counter() - t2:.2f}s.")

    metrics = metadata["metrics"]
    cis = metrics["bootstrap_cis"]
    costs = metrics["costs"]
    cm = metrics["confusion_matrix"]

    # Model Performance Table
    print("\n" + "=" * 78)
    print("MODEL PERFORMANCE ON TEST SET (WITH 95% BOOTSTRAP CONFIDENCE INTERVALS)")
    print("=" * 78)
    print(
        f"{'Model':<22} {'PR-AUC (95% CI)':<26} "
        f"{'ROC-AUC (95% CI)':<26} {'Brier Score':<12}"
    )
    print("-" * 78)

    # Prevalence baseline
    prev_pr_m = cis["prevalence"]["prauc"]["mean"]
    prev_pr_l = cis["prevalence"]["prauc"]["ci_lower"]
    prev_pr_u = cis["prevalence"]["prauc"]["ci_upper"]
    prev_pr = f"{prev_pr_m:.4f} [{prev_pr_l:.4f}, {prev_pr_u:.4f}]"

    prev_rc_m = cis["prevalence"]["rocauc"]["mean"]
    prev_rc_l = cis["prevalence"]["rocauc"]["ci_lower"]
    prev_rc_u = cis["prevalence"]["rocauc"]["ci_upper"]
    prev_roc = f"{prev_rc_m:.4f} [{prev_rc_l:.4f}, {prev_rc_u:.4f}]"
    print(f"{'Prevalence Baseline':<22} {prev_pr:<26} {prev_roc:<26} {'N/A':<12}")

    # Logistic Regression
    lr_pr_m = cis["logistic_regression"]["prauc"]["mean"]
    lr_pr_l = cis["logistic_regression"]["prauc"]["ci_lower"]
    lr_pr_u = cis["logistic_regression"]["prauc"]["ci_upper"]
    lr_pr = f"{lr_pr_m:.4f} [{lr_pr_l:.4f}, {lr_pr_u:.4f}]"

    lr_rc_m = cis["logistic_regression"]["rocauc"]["mean"]
    lr_rc_l = cis["logistic_regression"]["rocauc"]["ci_lower"]
    lr_rc_u = cis["logistic_regression"]["rocauc"]["ci_upper"]
    lr_roc = f"{lr_rc_m:.4f} [{lr_rc_l:.4f}, {lr_rc_u:.4f}]"
    print(f"{'Logistic Regression':<22} {lr_pr:<26} {lr_roc:<26} {'N/A':<12}")

    # XGBoost (Uncalibrated)
    xgb_pr_m = cis["xgb"]["prauc"]["mean"]
    xgb_pr_l = cis["xgb"]["prauc"]["ci_lower"]
    xgb_pr_u = cis["xgb"]["prauc"]["ci_upper"]
    xgb_pr = f"{xgb_pr_m:.4f} [{xgb_pr_l:.4f}, {xgb_pr_u:.4f}]"

    xgb_rc_m = cis["xgb"]["rocauc"]["mean"]
    xgb_rc_l = cis["xgb"]["rocauc"]["ci_lower"]
    xgb_rc_u = cis["xgb"]["rocauc"]["ci_upper"]
    xgb_roc = f"{xgb_rc_m:.4f} [{xgb_rc_l:.4f}, {xgb_rc_u:.4f}]"

    brier_uncal = f"{metrics['brier_test_uncalibrated']:.6f}"
    print(f"{'XGBoost (Uncalibrated)':<22} {xgb_pr:<26} {xgb_roc:<26} {brier_uncal:<12}")

    # XGBoost (Calibrated)
    brier_cal = f"{metrics['brier_test_calibrated']:.6f}"
    print(f"{'XGBoost (Calibrated)':<22} {xgb_pr:<26} {xgb_roc:<26} {brier_cal:<12}")
    print("-" * 78)
    b_uncal = metrics["brier_test_uncalibrated"]
    b_cal = metrics["brier_test_calibrated"]
    print(f"Calibration Impact: Brier score changed from {b_uncal:.6f} to {b_cal:.6f}")
    print(f"Precision @ Top 1%: {metrics['precision_top_1pct'] * 100:.2f}%")

    # Calibration Reliability Table
    print("\n" + "-" * 78)
    print("CALIBRATION RELIABILITY TABLE (TEST SET)")
    print("-" * 78)
    print(
        f"{'Probability Bin':<22} {'Count':>10} "
        f"{'Mean Predicted Prob':>22} {'Empirical Fraud Rate':>22}"
    )
    print("-" * 78)
    for bin_row in metadata["calibration_table"]:
        b_str = bin_row["bin"]
        b_cnt = bin_row["count"]
        b_mp = bin_row["mean_pred_prob"]
        b_ep = bin_row["empirical_prob"]
        print(f"{b_str:<22} {b_cnt:>10,} {b_mp:>22.4f} {b_ep:>22.4f}")
    print("-" * 78)

    # Cost Matrix & Threshold Results
    print("\n" + "=" * 78)
    print("COST MATRIX EVALUATION ON TEST SET")
    fa_cost = format_currency(metadata["false_alarm_cost"])
    print(f"Assumptions: False Alarm Cost = {fa_cost}, Missed Fraud = Txn Amount")
    print("=" * 78)
    th_val = metadata["threshold"]
    print(f"Chosen Decision Threshold (from Val B): {th_val:.4f}")
    tp, fp = cm["true_positives"], cm["false_positives"]
    fn, tn = cm["false_negatives"], cm["true_negatives"]
    print(f"Test Confusion Matrix at T={th_val:.4f}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    prec = metrics["test_precision"] * 100
    rec = metrics["test_recall"] * 100
    f1 = metrics["test_f1"] * 100
    print(f"Precision: {prec:.2f}%, Recall: {rec:.2f}%, F1: {f1:.2f}%")
    print("-" * 78)
    hdr_cost = (
        f"{'Decision Strategy':<30} {'Total Cost':>15} "
        f"{'False Positives':>16} {'False Negatives':>16}"
    )
    print(hdr_cost)
    print("-" * 78)
    c_nothing = format_currency(costs["flag_nothing_cost"])
    c_every = format_currency(costs["flag_everything_cost"])
    c_lr = format_currency(costs["logistic_regression_cost"])
    c_xgb = format_currency(costs["xgb_total_cost"])
    print(f"{'Flag Nothing (T=1.0)':<30} {c_nothing:>15} {'0':>16} {tp + fn:>16}")
    print(f"{'Flag Everything (T=0.0)':<30} {c_every:>15} {tn + fp:>16} {'0':>16}")
    print(f"{'Logistic Regression':<30} {c_lr:>15} {'-':>16} {'-':>16}")
    print(f"{'XGBoost (Optimal Threshold)':<30} {c_xgb:>15} {fp:>16} {fn:>16}")
    print("-" * 78)

    # Global SHAP Features
    print("\n" + "=" * 78)
    print("TOP-10 GLOBAL SHAP FEATURES (LOG-ODDS OF BASE MODEL)")
    print("=" * 78)
    print(f"{'Rank':<6} {'Feature Name':<32} {'Mean |SHAP| (Log-Odds)':>24}")
    print("-" * 78)
    for feat in metadata["top_10_shap_features"]:
        f_rk = feat["rank"]
        f_name = feat["feature"]
        f_shap = feat["mean_abs_shap_log_odds"]
        print(f"{f_rk:<6} {f_name:<32} {f_shap:>24.4f}")
    print("=" * 78)

    print("\n[4/4] Model Artifact & Metadata Saved:")
    print(f"  Artifact Path:   {settings.model_dir / 'fraud_model.joblib'}")
    print(f"  Artifact SHA256: {metadata['artifact_sha256']}")
    print(f"  Model Version:   {metadata['model_version']}")
    print(f"  Git Commit:      {metadata['git_commit']}")
    print("\nTraining completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
