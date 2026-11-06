# PayGuard Phase 5: Fraud Model Service

## Overview

The PayGuard Fraud Model Service provides low-latency risk scoring and point-in-time explainability for payment transactions. The service is built with an **XGBoost** classification model calibrated via **Isotonic Regression**, with real-time **SHAP (SHapley Additive exPlanations)** feature attribution.

> [!IMPORTANT]
> **Synthetic Data Disclaimer**: All models, performance metrics, confidence intervals, and thresholds reported in this phase are trained and evaluated on **synthetic benchmark data** generated during Phase 1. They represent system mechanics and evaluation integrity, not real-world merchant production distributions.

> [!WARNING]
> **Authentication Notice**: The `POST /ml/score` endpoint currently operates **without authentication**. Full API gateway authentication, JWT token verification, and role-based access control (RBAC) will be introduced in **Phase 8 (Security & Compliance)**.

---

## 1. Feature Architecture & Leakage Prevention

### 1.1 Allowed Features (14 Total)
To eliminate target leakage, only information knowable *at authorization time* is permitted:
1. `txn_hour` (0–23): Hour of transaction.
2. `day_of_week` (0–6): Day of week.
3. `amount`: Transaction amount in USD.
4. `log_amount`: `log1p(amount)`.
5. `channel`: Transaction channel (`POS`, `ECOMMERCE`, `ATM`, `MOTO`).
6. `mcc`: Merchant Category Code (4 digits).
7. `merchant_risk_tier`: Merchant risk classification (`LOW`, `MEDIUM`, `HIGH`).
8. `is_international`: Binary indicator (1 if card issuer country != merchant country, else 0).
9. `country_mismatch`: Binary indicator verifying cross-border mismatch.
10. `txn_count_1h`: Number of strictly earlier transactions of the same card in the last 1 hour.
11. `txn_count_24h`: Number of strictly earlier transactions of the same card in the last 24 hours.
12. `distinct_merchants_24h`: Distinct merchants visited by the card in the last 24 hours.
13. `is_new_merchant_for_card`: 1 if this card has never transacted with this merchant before, else 0.
14. `amount_past_mean_zscore`: Z-score of transaction amount relative to the card's past mean and standard deviation.

### 1.2 Forbidden Columns Guard
The pipeline strictly rejects and asserts against forbidden columns:
- `is_fraud`: Ground truth target label.
- `status`: Transaction settlement status (`APPROVED`, `DECLINED`, `REFUNDED`).
- `decline_code`: Post-authorization bank decline codes.
- `dispute_id`, `reason_code`, `filed_at`, `evidence_due_date`, `resolved_at`, `disputes`: Chargeback data that occurs days/weeks after transaction authorization.

### 1.3 Point-in-Time History Computation & Serving Path
History features use rows strictly before the scored transaction's `(timestamp, transaction_id)`.
- **Database Query**: Utilizes PostgreSQL B-Tree index scan on `idx_transactions_card_timestamp (card_token, timestamp)`:
  ```sql
  SELECT transaction_id, merchant_id, amount::float AS amount, timestamp
  FROM transactions
  WHERE card_token = %s
    AND timestamp <= %s
    AND (timestamp < %s OR transaction_id < %s)
  ORDER BY timestamp ASC, transaction_id ASC;
  ```
- **Execution Plan**: Index Scan using `idx_transactions_card_timestamp` followed by incremental sort (`Execution time < 0.2 ms`).
- **Batch Feature Time**: 500,000 transactions processed in ~1.8 seconds using two-pointer sliding window algorithm with zero data leakage.

---

## 2. Chronological Splitting & Training Strategy

Datasets are partitioned strictly by `timestamp ASC, transaction_id ASC`:
- **Train (70%)**: Trains prevalence baseline, Logistic Regression, and candidate XGBoost models.
- **Validation A (7.5%)**: Compares `scale_pos_weight` options and fits the probability calibrator (`IsotonicRegression`).
- **Validation B (7.5%)**: Optimizes the decision threshold $T^*$ against the financial cost matrix.
- **Test (15%)**: Evaluated **once** for the final report with 1,000 bootstrap confidence intervals.

### Tripwire
If test PR-AUC exceeds `0.90`, the pipeline raises a `LeakageTripwireError` and prints a prominent `LEAKAGE WARNING` to halt execution.

---

## 3. Probability Calibration & Cost Matrix

### 3.1 Calibration
Raw gradient-boosted decision trees often output uncalibrated probabilities skewed toward the extreme margins. We fit an unconstrained `IsotonicRegression` on `val_a` predictions to ensure empirical risk aligns with predicted probability.

### 3.2 Decision Threshold Optimization
Threshold $T^*$ is selected on `val_b` to minimize asymmetric financial cost:
$$\text{Cost} = \sum_{i \in \text{FN}} \text{amount}_i + |\text{FP}| \times \text{FALSE\_ALARM\_COST}$$

> [!NOTE]
> **False Alarm Cost Assumption**: The default `FALSE_ALARM_COST = $25.00` is an operational assumption modeling customer friction, manual analyst review, and support cost, rather than an audited accounting figure.

---

## 4. SHAP Explanations in Log-Odds

TreeSHAP values are computed in the **margin space (log-odds)** of the base XGBoost tree ensemble:
- Positive SHAP: Increases the log-odds of fraud.
- Negative SHAP: Decreases the log-odds of fraud.
- Explanations report the original raw feature values (`amount`, `channel`, etc.) alongside signed log-odds contributions.

---

## 5. Model Integrity & Service API

### 5.1 Model Integrity Guard
On startup, `FraudModelService` reads `models/model_metadata.json` and computes the SHA-256 checksum of `models/fraud_model.joblib`. If any byte is corrupted or modified, `ModelIntegrityError` is raised, refusing to serve traffic.

### 5.2 API Endpoint: `POST /ml/score`
- **Request**:
  ```json
  {
    "txn_id": "txn_0000000000_c4addab5"
  }
  ```
- **Luhn PAN Guard**: Pre-checks request parameters with the Luhn checksum; any input matching a card number format is immediately rejected with HTTP 400.
- **Response**:
  ```json
  {
    "txn_id": "txn_0000000000_c4addab5",
    "score": 0.0076,
    "label": 0,
    "threshold": 0.0042,
    "top_features": [
      {"feature": "channel", "value": "ECOMMERCE", "shap": 0.4215},
      {"feature": "txn_count_24h", "value": 3, "shap": 0.3120},
      {"feature": "amount", "value": 78.50, "shap": -0.1542},
      {"feature": "merchant_risk_tier", "value": "MEDIUM", "shap": 0.1250},
      {"feature": "txn_hour", "value": 14, "shap": -0.0980}
    ],
    "model_version": "a4f8d91c2b5e7890"
  }
  ```
- **Audit Logging**: Emits only `txn_id`, `score`, and `model_version`. Strictly no raw features or secrets in server logs.

