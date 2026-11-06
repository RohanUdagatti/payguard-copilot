"""Feature extraction and leakage prevention for the PayGuard fraud model.

Strictly enforces feature allowlisting, forbidden-feature guards, and point-in-time
history feature extraction using rows strictly earlier than the scored transaction's
(timestamp, transaction_id).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

ALLOWED_FEATURES: list[str] = [
    "txn_hour",
    "day_of_week",
    "amount",
    "log_amount",
    "channel",
    "mcc",
    "merchant_risk_tier",
    "is_international",
    "country_mismatch",
    "txn_count_1h",
    "txn_count_24h",
    "distinct_merchants_24h",
    "is_new_merchant_for_card",
    "amount_past_mean_zscore",
]

CATEGORICAL_FEATURES: list[str] = [
    "channel",
    "mcc",
    "merchant_risk_tier",
]

NUMERIC_FEATURES: list[str] = [
    "txn_hour",
    "day_of_week",
    "amount",
    "log_amount",
    "is_international",
    "country_mismatch",
    "txn_count_1h",
    "txn_count_24h",
    "distinct_merchants_24h",
    "is_new_merchant_for_card",
    "amount_past_mean_zscore",
]

FORBIDDEN_FEATURES: list[str] = [
    "decline_code",
    "status",
    "is_fraud",
    "dispute_id",
    "reason_code",
    "filed_at",
    "evidence_due_date",
    "resolved_at",
    "disputes",
]


def validate_feature_columns(columns: list[str] | pd.Index) -> None:
    """Validate that no forbidden or target leakage columns are in the feature set.

    Raises:
        ValueError: If any forbidden column or unknown column is detected.
    """
    col_set = set(columns)
    for forbidden in FORBIDDEN_FEATURES:
        if forbidden in col_set:
            raise ValueError(
                f"Data leakage detected! Forbidden column '{forbidden}' "
                "found in feature list."
            )
    # Also ensure every column is explicitly allowlisted
    for col in columns:
        if col not in ALLOWED_FEATURES:
            raise ValueError(
                f"Unapproved column '{col}' is not in the allowed feature list."
            )


def compute_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Compute allowed features for a chronological batch of transactions.

    The input DataFrame MUST contain:
      - transaction_id
      - card_token
      - merchant_id
      - amount
      - timestamp (datetime with tz)
      - txn_hour
      - channel
      - mcc
      - merchant_risk_tier
      - is_international
      - country_mismatch

    It computes strictly-past history features per card using only transactions
    prior to each row's (timestamp, transaction_id).
    """
    # Ensure dataframe is sorted by (timestamp ASC, transaction_id ASC)
    df_sorted = df.sort_values(
        by=["timestamp", "transaction_id"], ascending=[True, True]
    ).copy()

    n = len(df_sorted)
    # Derived point-in-time features
    ts_series = pd.to_datetime(df_sorted["timestamp"])
    df_sorted["txn_hour"] = ts_series.dt.hour.astype(int)
    df_sorted["day_of_week"] = ts_series.dt.dayofweek.astype(int)
    amounts = df_sorted["amount"].to_numpy(dtype=float)
    df_sorted["log_amount"] = np.log1p(np.maximum(0.0, amounts))
    df_sorted["is_international"] = df_sorted["is_international"].astype(int)
    df_sorted["country_mismatch"] = df_sorted["country_mismatch"].astype(int)

    # History features using strictly earlier transactions
    ts_sec = (ts_series.astype("int64") // 10**9).to_numpy()
    card_tokens = df_sorted["card_token"].to_numpy()
    merchants = df_sorted["merchant_id"].to_numpy()

    # Index rows per card_token
    card_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, c in enumerate(card_tokens):
        card_to_indices[c].append(idx)

    txn_count_1h = np.zeros(n, dtype=np.int32)
    txn_count_24h = np.zeros(n, dtype=np.int32)
    distinct_merchants_24h = np.zeros(n, dtype=np.int32)
    is_new_merchant = np.zeros(n, dtype=np.int8)
    amount_zscore = np.zeros(n, dtype=np.float32)

    for _card, idxs in card_to_indices.items():
        m_seen: set[str] = set()
        cum_sum = 0.0
        cum_sq_sum = 0.0
        k = len(idxs)

        p_1h = 0
        p_24h = 0
        m_counts_24h: dict[str, int] = defaultdict(int)

        for i in range(k):
            curr_idx = idxs[i]
            curr_t = ts_sec[curr_idx]
            curr_m = merchants[curr_idx]
            curr_amt = amounts[curr_idx]

            # Slide 1h window: remove transactions strictly older than (curr_t - 3600)
            while p_1h < i and ts_sec[idxs[p_1h]] < curr_t - 3600:
                p_1h += 1

            # Slide 24h window: remove transactions strictly older than (curr_t - 86400)
            while p_24h < i and ts_sec[idxs[p_24h]] < curr_t - 86400:
                old_m = merchants[idxs[p_24h]]
                m_counts_24h[old_m] -= 1
                if m_counts_24h[old_m] == 0:
                    del m_counts_24h[old_m]
                p_24h += 1

            # Strictly past counts in the window [curr_t - window, curr_t)
            txn_count_1h[curr_idx] = i - p_1h
            txn_count_24h[curr_idx] = i - p_24h
            distinct_merchants_24h[curr_idx] = len(m_counts_24h)

            # Add current merchant to sliding window for downstream rows
            m_counts_24h[curr_m] += 1

            # is_new_merchant_for_card (strictly before this transaction)
            is_new_merchant[curr_idx] = 1 if curr_m not in m_seen else 0
            m_seen.add(curr_m)

            # amount_past_mean_zscore: requires >= 2 past transactions to compute std
            if i < 2:
                amount_zscore[curr_idx] = 0.0
            else:
                past_mean = cum_sum / i
                past_var = (cum_sq_sum / i) - (past_mean * past_mean)
                past_std = np.sqrt(max(0.0, past_var))
                if past_std < 1e-6:
                    amount_zscore[curr_idx] = 0.0
                else:
                    amount_zscore[curr_idx] = float((curr_amt - past_mean) / past_std)

            cum_sum += curr_amt
            cum_sq_sum += curr_amt * curr_amt

    df_sorted["txn_count_1h"] = txn_count_1h
    df_sorted["txn_count_24h"] = txn_count_24h
    df_sorted["distinct_merchants_24h"] = distinct_merchants_24h
    df_sorted["is_new_merchant_for_card"] = is_new_merchant
    df_sorted["amount_past_mean_zscore"] = amount_zscore

    return df_sorted


def compute_features_single(
    target_txn: dict[str, Any],
    history_txns: list[dict[str, Any]],
) -> pd.DataFrame:
    """Compute features for a single transaction at serving time using past history.

    Parameters:
        target_txn: Target transaction dict containing:
            - transaction_id
            - card_token
            - merchant_id
            - amount
            - timestamp (datetime)
            - txn_hour
            - channel
            - mcc
            - merchant_risk_tier
            - is_international
            - country_mismatch
        history_txns: List of strictly earlier transaction dicts for this card:
            - transaction_id
            - merchant_id
            - amount
            - timestamp (datetime)
            (MUST already be strictly earlier than target transaction)

    Returns:
        pd.DataFrame: A 1-row DataFrame containing ALLOWED_FEATURES.
    """
    ts = pd.to_datetime(target_txn["timestamp"])
    target_ts_sec = int(ts.timestamp())
    target_amt = float(target_txn["amount"])
    target_merchant = str(target_txn["merchant_id"])

    # Filter strictly earlier history
    valid_history: list[dict[str, Any]] = []
    for h in history_txns:
        h_ts = pd.to_datetime(h["timestamp"])
        h_ts_sec = int(h_ts.timestamp())
        h_id = str(h["transaction_id"])
        # Strict point-in-time check
        if (h_ts_sec < target_ts_sec) or (
            h_ts_sec == target_ts_sec and h_id < str(target_txn["transaction_id"])
        ):
            valid_history.append(h)

    # Sort history chronologically
    valid_history.sort(
        key=lambda x: (
            int(pd.to_datetime(x["timestamp"]).timestamp()),
            str(x["transaction_id"]),
        )
    )

    t_1h_thresh = target_ts_sec - 3600
    t_24h_thresh = target_ts_sec - 86400

    past_count_1h = 0
    past_count_24h = 0
    merchants_24h: set[str] = set()
    merchants_all_past: set[str] = set()
    past_amounts: list[float] = []

    for h in valid_history:
        h_sec = int(pd.to_datetime(h["timestamp"]).timestamp())
        h_amt = float(h["amount"])
        h_m = str(h["merchant_id"])

        if h_sec >= t_1h_thresh:
            past_count_1h += 1
        if h_sec >= t_24h_thresh:
            past_count_24h += 1
            merchants_24h.add(h_m)

        merchants_all_past.add(h_m)
        past_amounts.append(h_amt)

    is_new_merchant = 1 if target_merchant not in merchants_all_past else 0

    if len(past_amounts) < 2:
        amount_zscore = 0.0
    else:
        past_mean = float(np.mean(past_amounts))
        past_std = float(np.std(past_amounts))  # population std matches batch var
        if past_std < 1e-6:
            amount_zscore = 0.0
        else:
            amount_zscore = float((target_amt - past_mean) / past_std)

    row = {
        "txn_hour": int(target_txn.get("txn_hour", ts.hour)),
        "day_of_week": int(ts.dayofweek),
        "amount": target_amt,
        "log_amount": float(np.log1p(max(0.0, target_amt))),
        "channel": str(target_txn["channel"]),
        "mcc": str(target_txn["mcc"]),
        "merchant_risk_tier": str(target_txn["merchant_risk_tier"]),
        "is_international": int(target_txn["is_international"]),
        "country_mismatch": int(target_txn["country_mismatch"]),
        "txn_count_1h": int(past_count_1h),
        "txn_count_24h": int(past_count_24h),
        "distinct_merchants_24h": int(len(merchants_24h)),
        "is_new_merchant_for_card": int(is_new_merchant),
        "amount_past_mean_zscore": float(amount_zscore),
    }

    validate_feature_columns(list(row.keys()))
    return pd.DataFrame([row])[ALLOWED_FEATURES]
