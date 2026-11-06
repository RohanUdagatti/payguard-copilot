"""Training pipeline, calibration, and evaluation for fraud model.

Strictly adheres to:
- Chronological time split (70% train, 7.5% val_a, 7.5% val_b, 15% test).
- Baselines: Prevalence baseline & Logistic Regression.
- XGBoost scale_pos_weight optimization on val_a PR-AUC.
- Isotonic regression calibration fitted on val_a.
- Cost threshold optimization on val_b.
- Test set evaluated once for final report.
- Bootstrap 95% confidence intervals (1,000 resamples).
- SHAP explanations in log-odds.
- Model artifact integrity (SHA-256 in model_metadata.json).
- MLflow logging to local mlruns/.
- Tripwire: PR-AUC > 0.90 halts with LEAKAGE WARNING.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

import joblib
import mlflow
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import shap
import xgboost as xgb
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from payguard.config import Settings, get_settings
from payguard.ml.features import (
    ALLOWED_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    compute_features_batch,
    validate_feature_columns,
)

logger = logging.getLogger(__name__)


class LeakageTripwireError(RuntimeError):
    """Raised when evaluation metrics trip the leakage threshold."""


@dataclass
class DatasetSplits:
    """Chronologically split datasets."""

    train_df: pd.DataFrame
    val_a_df: pd.DataFrame
    val_b_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols: list[str]


@dataclass
class CostMetrics:
    """Evaluation costs on a dataset."""

    threshold: float
    total_cost: float
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int
    fn_amount_cost: float
    fp_cost: float


@dataclass
class BootstrapCI:
    """Bootstrap confidence intervals."""

    mean: float
    ci_lower: float
    ci_upper: float


def get_git_commit() -> str:
    """Retrieve the current git commit hash."""
    try:
        out = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        return out
    except Exception:
        return "unknown"


def compute_deterministic_version(
    min_ts: str,
    max_ts: str,
    n_rows: int,
    features: list[str],
    params: dict[str, Any],
) -> str:
    """Generate a deterministic version hash from dataset window, features, params."""
    feat_str = ",".join(sorted(features))
    param_str = json.dumps(params, sort_keys=True)
    raw = f"{min_ts}|{max_ts}|{n_rows}|{feat_str}|{param_str}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_dataset_from_db(
    settings: Settings | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load transactions joined with cards and merchants strictly ordered by time."""
    import psycopg

    s = settings or get_settings()
    conn_str = (
        f"host={s.postgres_host} port={s.postgres_port} "
        f"dbname={s.postgres_db} user={s.postgres_user} "
        f"password={s.postgres_password}"
    )

    query = """
    SELECT 
        t.transaction_id,
        t.card_token,
        t.merchant_id,
        t.amount::float AS amount,
        t.timestamp,
        t.txn_hour,
        t.channel,
        t.mcc,
        t.risk_tier AS merchant_risk_tier,
        t.is_international,
        (c.issuer_country != m.country) AS country_mismatch,
        t.is_fraud
    FROM transactions t
    JOIN cards c ON t.card_token = c.card_token
    JOIN merchants m ON t.merchant_id = m.merchant_id
    ORDER BY t.timestamp ASC, t.transaction_id ASC
    """
    if limit is not None:
        query += f" LIMIT {limit}"

    with psycopg.connect(conn_str) as conn:
        df = pd.read_sql(query, conn)

    return df


def prepare_dataset_splits(df: pd.DataFrame) -> DatasetSplits:
    """Extract features and partition chronologically into train, val_a, val_b, test."""
    logger.info("Computing point-in-time features on dataset of %d rows...", len(df))
    df_feat = compute_features_batch(df)

    validate_feature_columns(ALLOWED_FEATURES)

    # Chronological partition
    n = len(df_feat)
    n_train = int(0.70 * n)
    n_val_a = int(0.775 * n)
    n_val_b = int(0.85 * n)

    train_df = df_feat.iloc[:n_train].copy()
    val_a_df = df_feat.iloc[n_train:n_val_a].copy()
    val_b_df = df_feat.iloc[n_val_a:n_val_b].copy()
    test_df = df_feat.iloc[n_val_b:].copy()

    # Format categorical columns as pd.Categorical
    for col in CATEGORICAL_FEATURES:
        train_df[col] = train_df[col].astype("category")
        val_a_df[col] = pd.Categorical(
            val_a_df[col], categories=train_df[col].cat.categories
        )
        val_b_df[col] = pd.Categorical(
            val_b_df[col], categories=train_df[col].cat.categories
        )
        test_df[col] = pd.Categorical(
            test_df[col], categories=train_df[col].cat.categories
        )

    return DatasetSplits(
        train_df=train_df,
        val_a_df=val_a_df,
        val_b_df=val_b_df,
        test_df=test_df,
        feature_cols=ALLOWED_FEATURES,
    )


def compute_cost(
    y_true: NDArray[Any],
    y_probs: NDArray[Any],
    amounts: NDArray[Any],
    threshold: float,
    false_alarm_cost: float,
) -> CostMetrics:
    """Compute financial fraud cost at a given probability threshold."""
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Cost calculation:
    # Missed fraud (FN) cost = sum of transaction amounts
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_amount_cost = float(np.sum(amounts[fn_mask]))

    # False alarm (FP) cost = number of FPs * false_alarm_cost
    fp_cost = float(fp * false_alarm_cost)

    total_cost = fn_amount_cost + fp_cost

    return CostMetrics(
        threshold=threshold,
        total_cost=total_cost,
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        true_negatives=int(tn),
        fn_amount_cost=fn_amount_cost,
        fp_cost=fp_cost,
    )


def find_optimal_threshold(
    y_true: NDArray[Any],
    y_probs: NDArray[Any],
    amounts: NDArray[Any],
    false_alarm_cost: float,
) -> float:
    """Find the threshold minimizing cost on the given dataset (val_b)."""
    # Candidate thresholds across distribution percentiles and boundaries
    percentiles = np.linspace(0, 100, 1000)
    candidate_thresholds = np.unique(
        np.clip(np.percentile(y_probs, percentiles), 1e-6, 1.0 - 1e-6)
    )
    # Include grid points
    grid_points = np.linspace(0.001, 0.999, 500)
    all_candidates = np.unique(
        np.concatenate([candidate_thresholds, grid_points, [0.0, 1.0]])
    )

    best_thresh = 0.5
    min_cost = float("inf")

    for th in all_candidates:
        cost_metric = compute_cost(y_true, y_probs, amounts, th, false_alarm_cost)
        if cost_metric.total_cost < min_cost:
            min_cost = cost_metric.total_cost
            best_thresh = float(th)

    return best_thresh


def compute_calibration_table(
    y_true: NDArray[Any],
    y_calibrated: NDArray[Any],
    n_bins: int = 5,
) -> list[dict[str, Any]]:
    """Compute reliability table for probability calibration."""
    # Group by quantiles
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(y_calibrated, quantiles)
    # Ensure strictly increasing
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    table: list[dict[str, Any]] = []
    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            mask = (y_calibrated >= low) & (y_calibrated <= high)
        else:
            mask = (y_calibrated >= low) & (y_calibrated < high)

        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(y_calibrated[mask]))
            empirical_prob = float(np.mean(y_true[mask]))
        else:
            mean_pred = float((low + high) / 2.0)
            empirical_prob = 0.0

        table.append(
            {
                "bin": f"[{low:.4f}, {high:.4f}]",
                "count": count,
                "mean_pred_prob": mean_pred,
                "empirical_prob": empirical_prob,
            }
        )

    return table


def bootstrap_metric_ci(
    y_true: NDArray[Any],
    y_probs: NDArray[Any],
    amounts: NDArray[Any],
    threshold: float,
    false_alarm_cost: float,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> tuple[BootstrapCI, BootstrapCI, BootstrapCI]:
    """Compute 95% bootstrap confidence intervals for PR-AUC, ROC-AUC, and cost."""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    praucs: list[float] = []
    rocaucs: list[float] = []
    costs: list[float] = []

    for _ in range(n_bootstraps):
        idxs = rng.integers(0, n, size=n)
        b_true = y_true[idxs]
        b_probs = y_probs[idxs]
        b_amts = amounts[idxs]

        # Guard against sample with 0 positives
        if np.sum(b_true) == 0:
            continue

        prauc = average_precision_score(b_true, b_probs)
        rocauc = roc_auc_score(b_true, b_probs) if len(np.unique(b_true)) > 1 else 0.5
        cost_m = compute_cost(b_true, b_probs, b_amts, threshold, false_alarm_cost)

        praucs.append(prauc)
        rocaucs.append(rocauc)
        costs.append(cost_m.total_cost)

    prauc_ci = BootstrapCI(
        mean=float(np.mean(praucs)),
        ci_lower=float(np.percentile(praucs, 2.5)),
        ci_upper=float(np.percentile(praucs, 97.5)),
    )
    rocauc_ci = BootstrapCI(
        mean=float(np.mean(rocaucs)),
        ci_lower=float(np.percentile(rocaucs, 2.5)),
        ci_upper=float(np.percentile(rocaucs, 97.5)),
    )
    cost_ci = BootstrapCI(
        mean=float(np.mean(costs)),
        ci_lower=float(np.percentile(costs, 2.5)),
        ci_upper=float(np.percentile(costs, 97.5)),
    )

    return prauc_ci, rocauc_ci, cost_ci


def train_and_evaluate_fraud_pipeline(
    splits: DatasetSplits,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Execute the full training pipeline, calibration, thresholding, and evaluation."""
    s = settings or get_settings()
    false_alarm_cost = s.false_alarm_cost
    model_dir = s.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = splits.feature_cols
    X_train = splits.train_df[feature_cols]
    y_train = splits.train_df["is_fraud"].to_numpy().astype(int)

    X_val_a = splits.val_a_df[feature_cols]
    y_val_a = splits.val_a_df["is_fraud"].to_numpy().astype(int)

    X_val_b = splits.val_b_df[feature_cols]
    y_val_b = splits.val_b_df["is_fraud"].to_numpy().astype(int)
    amounts_val_b = splits.val_b_df["amount"].to_numpy().astype(float)

    X_test = splits.test_df[feature_cols]
    y_test = splits.test_df["is_fraud"].to_numpy().astype(int)
    amounts_test = splits.test_df["amount"].to_numpy().astype(float)

    # 1. Prevalence Baseline
    train_prevalence = float(np.mean(y_train))
    prev_preds_test = np.full(len(y_test), train_prevalence)

    # 2. Logistic Regression Baseline
    logger.info("Fitting Logistic Regression baseline...")
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    lr_pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=42
                ),
            ),
        ]
    )
    lr_pipeline.fit(X_train, y_train)
    lr_probs_val_b = lr_pipeline.predict_proba(X_val_b)[:, 1]
    lr_probs_test = lr_pipeline.predict_proba(X_test)[:, 1]

    # Find optimal threshold for Logistic Regression on val_b
    lr_opt_thresh = find_optimal_threshold(
        y_val_b, lr_probs_val_b, amounts_val_b, false_alarm_cost
    )
    lr_test_cost = compute_cost(
        y_test, lr_probs_test, amounts_test, lr_opt_thresh, false_alarm_cost
    )

    # 3. XGBoost: evaluate scale_pos_weight on val_a PR-AUC
    logger.info("Comparing XGBoost scale_pos_weight configurations on val_a...")
    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    scale_pos_val = float(neg_count / max(1, pos_count))

    candidate_spw = [1.0, scale_pos_val]
    best_spw = 1.0
    best_val_a_prauc = -1.0
    spw_results: dict[float, float] = {}

    for spw in candidate_spw:
        clf_candidate = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            enable_categorical=True,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        clf_candidate.fit(X_train, y_train)
        val_a_preds = clf_candidate.predict_proba(X_val_a)[:, 1]
        val_a_prauc = float(average_precision_score(y_val_a, val_a_preds))
        spw_results[spw] = val_a_prauc
        logger.info("scale_pos_weight=%.2f -> val_a PR-AUC: %.4f", spw, val_a_prauc)
        if val_a_prauc > best_val_a_prauc:
            best_val_a_prauc = val_a_prauc
            best_spw = spw

    logger.info("Chosen XGBoost scale_pos_weight: %.2f", best_spw)

    final_xgb = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=best_spw,
        enable_categorical=True,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    final_xgb.fit(X_train, y_train)

    # 4. Probability Calibration on val_a
    logger.info("Fitting IsotonicRegression probability calibration on val_a...")
    uncal_probs_val_a = final_xgb.predict_proba(X_val_a)[:, 1]
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(uncal_probs_val_a, y_val_a)

    # Check Brier score on val_a
    brier_val_a_uncal = float(brier_score_loss(y_val_a, uncal_probs_val_a))
    cal_probs_val_a = calibrator.predict(uncal_probs_val_a)
    brier_val_a_cal = float(brier_score_loss(y_val_a, cal_probs_val_a))

    # Calibration on val_b and test
    uncal_probs_val_b = final_xgb.predict_proba(X_val_b)[:, 1]
    cal_probs_val_b = calibrator.predict(uncal_probs_val_b)

    uncal_probs_test = final_xgb.predict_proba(X_test)[:, 1]
    cal_probs_test = calibrator.predict(uncal_probs_test)

    brier_test_uncal = float(brier_score_loss(y_test, uncal_probs_test))
    brier_test_cal = float(brier_score_loss(y_test, cal_probs_test))

    calibration_table = compute_calibration_table(y_test, cal_probs_test, n_bins=5)

    # 5. Threshold Selection on val_b
    logger.info("Optimizing cost threshold on val_b...")
    chosen_threshold = find_optimal_threshold(
        y_val_b, cal_probs_val_b, amounts_val_b, false_alarm_cost
    )
    logger.info("Chosen optimal threshold: %.4f", chosen_threshold)

    # 6. Test Evaluation
    logger.info("Evaluating chosen threshold on test set...")
    test_cost_xgb = compute_cost(
        y_test, cal_probs_test, amounts_test, chosen_threshold, false_alarm_cost
    )

    # Alternatives on test set
    # Alternative A: Flag Nothing (threshold = 1.0)
    cost_flag_nothing = compute_cost(
        y_test, cal_probs_test, amounts_test, 1.0 + 1e-5, false_alarm_cost
    )
    # Alternative B: Flag Everything (threshold = 0.0)
    cost_flag_everything = compute_cost(
        y_test, cal_probs_test, amounts_test, 0.0, false_alarm_cost
    )

    test_prauc = float(average_precision_score(y_test, cal_probs_test))
    test_rocauc = float(roc_auc_score(y_test, cal_probs_test))

    # Precision@top-1%
    top_1pct_cutoff = int(np.ceil(0.01 * len(y_test)))
    top_1pct_indices = np.argsort(cal_probs_test)[::-1][:top_1pct_cutoff]
    precision_top_1pct = float(np.mean(y_test[top_1pct_indices]))

    y_test_pred = (cal_probs_test >= chosen_threshold).astype(int)
    test_precision = float(precision_score(y_test, y_test_pred, zero_division=0.0))
    test_recall = float(recall_score(y_test, y_test_pred, zero_division=0))
    test_f1 = float(f1_score(y_test, y_test_pred, zero_division=0))

    # TRIPWIRE CHECK: PR-AUC > 0.90
    if test_prauc > 0.90:
        warning_msg = (
            f"LEAKAGE WARNING: Test PR-AUC is {test_prauc:.4f} (> 0.90)! "
            f"Potential target leakage detected! Halting for review."
        )
        print("*" * 70)
        print(warning_msg)
        print("*" * 70)
        raise LeakageTripwireError(warning_msg)

    # 7. Bootstrap Confidence Intervals (1,000 resamples)
    logger.info(
        "Computing 1,000 bootstrap confidence intervals on test set for all models..."
    )
    xgb_prauc_ci, xgb_rocauc_ci, xgb_cost_ci = bootstrap_metric_ci(
        y_test,
        cal_probs_test,
        amounts_test,
        chosen_threshold,
        false_alarm_cost,
        n_bootstraps=1000,
    )
    prev_prauc_ci, prev_rocauc_ci, prev_cost_ci = bootstrap_metric_ci(
        y_test,
        prev_preds_test,
        amounts_test,
        0.5,
        false_alarm_cost,
        n_bootstraps=1000,
    )
    lr_prauc_ci, lr_rocauc_ci, lr_cost_ci = bootstrap_metric_ci(
        y_test,
        lr_probs_test,
        amounts_test,
        lr_opt_thresh,
        false_alarm_cost,
        n_bootstraps=1000,
    )

    # 8. SHAP Explanations in Log-Odds
    logger.info("Computing SHAP values in log-odds...")
    explainer = shap.TreeExplainer(final_xgb)
    # Explain test set sample (or entire test set)
    test_sample = X_test.iloc[:5000]
    shap_vals = explainer(test_sample)

    # Mean absolute SHAP per feature in log-odds
    mean_abs_shap = np.mean(np.abs(shap_vals.values), axis=0)
    feature_importance_order = np.argsort(mean_abs_shap)[::-1]
    top_10_shap = [
        {
            "rank": rank + 1,
            "feature": feature_cols[idx],
            "mean_abs_shap_log_odds": float(mean_abs_shap[idx]),
        }
        for rank, idx in enumerate(feature_importance_order[:10])
    ]

    # 9. Artifact Serialization and Deterministic Metadata
    model_artifact_path = model_dir / "fraud_model.joblib"
    metadata_path = model_dir / "model_metadata.json"

    # Save model bundle
    bundle = {
        "model": final_xgb,
        "calibrator": calibrator,
        "feature_names": feature_cols,
        "categorical_features": CATEGORICAL_FEATURES,
        "threshold": chosen_threshold,
    }
    joblib.dump(bundle, model_artifact_path)

    # Compute SHA-256 of the saved artifact
    with open(model_artifact_path, "rb") as f:
        artifact_sha256 = hashlib.sha256(f.read()).hexdigest()

    min_ts_str = str(splits.train_df["timestamp"].min())
    max_ts_str = str(splits.test_df["timestamp"].max())
    total_rows = (
        len(splits.train_df)
        + len(splits.val_a_df)
        + len(splits.val_b_df)
        + len(splits.test_df)
    )

    model_params = {
        "n_estimators": 120,
        "max_depth": 5,
        "learning_rate": 0.08,
        "scale_pos_weight": best_spw,
    }

    model_version = compute_deterministic_version(
        min_ts=min_ts_str,
        max_ts=max_ts_str,
        n_rows=total_rows,
        features=feature_cols,
        params=model_params,
    )

    metadata: dict[str, Any] = {
        "model_version": model_version,
        "artifact_sha256": artifact_sha256,
        "threshold": chosen_threshold,
        "false_alarm_cost": false_alarm_cost,
        "features": feature_cols,
        "categorical_features": CATEGORICAL_FEATURES,
        "hyperparameters": model_params,
        "data_window": {
            "start": min_ts_str,
            "end": max_ts_str,
            "total_rows": total_rows,
            "train_positives": int(np.sum(y_train)),
            "val_a_positives": int(np.sum(y_val_a)),
            "val_b_positives": int(np.sum(y_val_b)),
            "test_positives": int(np.sum(y_test)),
        },
        "metrics": {
            "val_a_prauc": best_val_a_prauc,
            "brier_val_a_uncalibrated": brier_val_a_uncal,
            "brier_val_a_calibrated": brier_val_a_cal,
            "brier_test_uncalibrated": brier_test_uncal,
            "brier_test_calibrated": brier_test_cal,
            "test_prauc": test_prauc,
            "test_rocauc": test_rocauc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "precision_top_1pct": precision_top_1pct,
            "confusion_matrix": asdict(test_cost_xgb),
            "costs": {
                "xgb_total_cost": test_cost_xgb.total_cost,
                "flag_nothing_cost": cost_flag_nothing.total_cost,
                "flag_everything_cost": cost_flag_everything.total_cost,
                "logistic_regression_cost": lr_test_cost.total_cost,
            },
            "bootstrap_cis": {
                "xgb": {
                    "prauc": asdict(xgb_prauc_ci),
                    "rocauc": asdict(xgb_rocauc_ci),
                    "cost": asdict(xgb_cost_ci),
                },
                "prevalence": {
                    "prauc": asdict(prev_prauc_ci),
                    "rocauc": asdict(prev_rocauc_ci),
                    "cost": asdict(prev_cost_ci),
                },
                "logistic_regression": {
                    "prauc": asdict(lr_prauc_ci),
                    "rocauc": asdict(lr_rocauc_ci),
                    "cost": asdict(lr_cost_ci),
                },
            },
        },
        "git_commit": get_git_commit(),
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "calibration_table": calibration_table,
        "top_10_shap_features": top_10_shap,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 10. MLflow Logging
    try:
        import os

        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri(s.mlflow_tracking_uri)
        mlflow.set_experiment("fraud_detection")
        with mlflow.start_run(run_name="xgboost_fraud_service"):
            mlflow.log_params(model_params)
            mlflow.log_params(
                {
                    "chosen_threshold": chosen_threshold,
                    "false_alarm_cost": false_alarm_cost,
                    "model_version": model_version,
                }
            )
            mlflow.log_metrics(
                {
                    "test_prauc": test_prauc,
                    "test_rocauc": test_rocauc,
                    "brier_test_calibrated": brier_test_cal,
                    "brier_test_uncalibrated": brier_test_uncal,
                    "test_cost": test_cost_xgb.total_cost,
                    "precision_top_1pct": precision_top_1pct,
                }
            )
            mlflow.set_tags(
                {
                    "git_commit": metadata["git_commit"],
                    "data_start": min_ts_str,
                    "data_end": max_ts_str,
                    "artifact_sha256": artifact_sha256,
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow logging skipped: %s", exc)

    return metadata
