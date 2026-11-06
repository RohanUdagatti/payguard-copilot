#!/usr/bin/env python3
"""Audit and diagnostic script for PayGuard synthetic payment data.

Writes comprehensive card activity, velocity distributions, and full-split lift tables
directly to eval/results/fraud_data_audit.txt.
"""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np
import pandas as pd
import psycopg

from payguard.config import get_settings
from payguard.data.seed import GENERATOR_PARAMS_HASH
from payguard.ml.pipeline import load_dataset_from_db, prepare_dataset_splits


def compute_lift_table(
    df: pd.DataFrame,
    col_name: str,
    bucket_series: pd.Series,
    split_name: str,
) -> tuple[str, int]:
    """Compute lift table for a categorized bucket series."""
    total_rows = len(df)
    total_frauds = int(df["is_fraud"].sum())
    base_rate = total_frauds / total_rows if total_rows > 0 else 0.0

    lines = []
    lines.append(f"#### Feature: {col_name} ({split_name})")
    base_info = (
        f"Base Rate: {base_rate * 100:.3f}% "
        f"({total_frauds:,} frauds / {total_rows:,} rows)"
    )
    lines.append(base_info)
    header = (
        "| Bucket | Row Count | Share (%) | Fraud Count "
        "| Fraud Rate (%) | Lift vs Base |"
    )
    lines.append(header)
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    grouped = (
        df.assign(_b=bucket_series)
        .groupby("_b", observed=False)
        .agg(
            row_count=("is_fraud", "count"),
            fraud_count=("is_fraud", "sum"),
        )
    )

    sum_rows = 0
    for bucket_val, row in grouped.iterrows():
        cnt = int(row["row_count"])
        f_cnt = int(row["fraud_count"])
        sum_rows += cnt
        share = cnt / total_rows * 100 if total_rows > 0 else 0.0
        f_rate = f_cnt / cnt * 100 if cnt > 0 else 0.0
        lift = (f_rate / (base_rate * 100)) if base_rate > 0 else 0.0
        row_str = (
            f"| {bucket_val} | {cnt:,} | {share:.2f}% | {f_cnt:,} | "
            f"{f_rate:.3f}% | {lift:.2f}x |"
        )
        lines.append(row_str)

    # Strict assertion: bucket counts must sum exactly to the split size
    assert sum_rows == total_rows, (
        f"Bucket count sum ({sum_rows}) does not match "
        f"split size ({total_rows}) for {col_name}!"
    )

    lines.append("")
    return "\n".join(lines), sum_rows


def run_audit(output_path: str = "eval/results/fraud_data_audit.txt") -> None:
    """Run full fraud data diagnostic audit and write to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    settings = get_settings()

    out_lines: list[str] = []
    out_lines.append("=" * 80)
    out_lines.append("PAYGUARD FRAUD DATA AUDIT & TYPOLOGY DIAGNOSTICS")
    out_lines.append(f"Generated: {datetime.datetime.now(datetime.UTC).isoformat()}")
    out_lines.append(f"Generator V2 Parameter Hash: {GENERATOR_PARAMS_HASH}")
    out_lines.append("=" * 80)
    out_lines.append("")

    # 1. Connect to DB and audit table counts
    conn_str = (
        f"host={settings.postgres_host} port={settings.postgres_port} "
        f"dbname={settings.postgres_db} user={settings.postgres_user} "
        f"password={settings.postgres_password}"
    )

    out_lines.append("## 1. Relational Table Summary")
    out_lines.append("-" * 80)
    query_counts = """
    SELECT 'cards' AS tbl, count(*) AS cnt FROM cards
    UNION ALL SELECT 'merchants', count(*) FROM merchants
    UNION ALL SELECT 'reason_codes', count(*) FROM reason_codes
    UNION ALL SELECT 'transactions', count(*) FROM transactions
    UNION ALL SELECT 'disputes', count(*) FROM disputes;
    """
    with psycopg.connect(conn_str) as conn, conn.cursor() as cur:
        cur.execute(query_counts)
        table_counts = dict(cur.fetchall())

    for tbl, cnt in table_counts.items():
        out_lines.append(f"- {tbl:<15}: {cnt:>12,}")
    out_lines.append("")

    # 2. Card Activity Distributions (Raw DB Query)
    out_lines.append("## 2. Card Activity Analysis (Transactions Per Card)")
    out_lines.append("-" * 80)
    with psycopg.connect(conn_str) as conn:
        card_activity_df = pd.read_sql(
            """
            SELECT card_token, count(*) AS txn_count
            FROM transactions
            GROUP BY card_token;
            """,
            conn,
        )
        legit_daily_df = pd.read_sql(
            """
            SELECT card_token, timestamp::date AS txn_date, count(*) AS daily_count
            FROM transactions
            WHERE is_fraud = false
            GROUP BY card_token, timestamp::date;
            """,
            conn,
        )

    counts_arr = card_activity_df["txn_count"].to_numpy()
    daily_arr = legit_daily_df["daily_count"].to_numpy()

    out_lines.append("### 2.1 Total Transactions per Card (90-Day Window)")
    out_lines.append(f"- Total Distinct Cards with Transactions: {len(counts_arr):,}")
    out_lines.append(f"- Mean Transactions per Card : {np.mean(counts_arr):.2f}")
    out_lines.append(f"- Std Deviation               : {np.std(counts_arr):.2f}")
    out_lines.append(f"- Median (p50)                : {np.median(counts_arr):.1f}")
    out_lines.append(
        f"- 75th Percentile (p75)       : {np.percentile(counts_arr, 75):.1f}"
    )
    out_lines.append(
        f"- 90th Percentile (p90)       : {np.percentile(counts_arr, 90):.1f}"
    )
    out_lines.append(
        f"- 95th Percentile (p95)       : {np.percentile(counts_arr, 95):.1f}"
    )
    out_lines.append(
        f"- 99th Percentile (p99)       : {np.percentile(counts_arr, 99):.1f}"
    )
    out_lines.append(f"- Max Transactions per Card   : {np.max(counts_arr):,}")
    out_lines.append("")

    out_lines.append("### 2.2 Daily Activity per Card (Legitimate Transactions Only)")
    out_lines.append(
        f"- Mean Legit Transactions / Active Day  : {np.mean(daily_arr):.2f}"
    )
    out_lines.append(
        f"- Median Legit Transactions / Active Day: {np.median(daily_arr):.1f}"
    )
    out_lines.append(
        f"- 95th Percentile (p95)                 : {np.percentile(daily_arr, 95):.1f}"
    )
    out_lines.append(
        f"- 99th Percentile (p99)                 : {np.percentile(daily_arr, 99):.1f}"
    )
    max_daily = int(np.max(daily_arr))
    out_lines.append(
        f"- Max Transactions per Card per Day     : {max_daily} (Hard Cap: <= 8)"
    )
    assert (
        max_daily <= 8
    ), f"Hard cap violated: found {max_daily} legit txns for a card on one day!"
    out_lines.append(
        "  [PASS] Verified: Zero legitimate cards exceed "
        "the 8 transactions/day hard cap."
    )
    out_lines.append("")

    # 3. Load Dataset & Point-in-Time History Features
    out_lines.append("## 3. Dataset Splits & Chronological Velocity Distributions")
    out_lines.append("-" * 80)
    raw_df = load_dataset_from_db(settings)
    splits = prepare_dataset_splits(raw_df)

    split_dict = {
        "Train (70%)": splits.train_df,
        "Val A (7.5%)": splits.val_a_df,
        "Val B (7.5%)": splits.val_b_df,
        "Test (15%)": splits.test_df,
    }

    hdr = (
        f"{'Split Name':<15} | {'Row Count':>10} | "
        f"{'Positives':>10} | {'Fraud Rate':>12}"
    )
    out_lines.append(hdr)
    out_lines.append("-" * 55)
    for s_name, s_df in split_dict.items():
        f_cnt = int(s_df["is_fraud"].sum())
        r_cnt = len(s_df)
        frate = f_cnt / r_cnt * 100 if r_cnt else 0.0
        line = f"{s_name:<15} | {r_cnt:>10,} | {f_cnt:>10,} | {frate:>11.3f}%"
        out_lines.append(line)
    out_lines.append("")

    # 4. Global Velocity Statistics
    all_v24 = np.concatenate(
        [s["txn_count_24h"].to_numpy() for s in split_dict.values()]
    )
    all_v1h = np.concatenate(
        [s["txn_count_1h"].to_numpy() for s in split_dict.values()]
    )

    out_lines.append("### 3.1 txn_count_24h Distribution (All 500,000 Rows)")
    out_lines.append(f"- Mean                : {np.mean(all_v24):.2f}")
    out_lines.append(f"- Std                 : {np.std(all_v24):.2f}")
    out_lines.append(f"- Median (p50)        : {np.median(all_v24):.1f}")
    out_lines.append(f"- 75th Percentile (p75): {np.percentile(all_v24, 75):.1f}")
    out_lines.append(f"- 90th Percentile (p90): {np.percentile(all_v24, 90):.1f}")
    out_lines.append(f"- 95th Percentile (p95): {np.percentile(all_v24, 95):.1f}")
    out_lines.append(f"- 99th Percentile (p99): {np.percentile(all_v24, 99):.1f}")
    out_lines.append(f"- Max                 : {np.max(all_v24):,}")
    out_lines.append(f"- Fraction > 1 txn/24h: {np.mean(all_v24 > 1) * 100:.2f}%")
    out_lines.append(f"- Fraction > 3 txns/24h: {np.mean(all_v24 > 3) * 100:.2f}%")
    out_lines.append(f"- Fraction > 5 txns/24h: {np.mean(all_v24 > 5) * 100:.2f}%")
    out_lines.append("")

    out_lines.append("### 3.2 txn_count_1h Distribution (All 500,000 Rows)")
    out_lines.append(f"- Mean                : {np.mean(all_v1h):.3f}")
    out_lines.append(f"- Median (p50)        : {np.median(all_v1h):.1f}")
    out_lines.append(f"- 99th Percentile (p99): {np.percentile(all_v1h, 99):.1f}")
    out_lines.append(f"- Max                 : {np.max(all_v1h):,}")
    v1h_pct = np.mean(all_v1h >= 3) * 100
    v1h_sum = np.sum(all_v1h >= 3)
    out_lines.append(f"- Fraction >= 3 txns/1h: {v1h_pct:.4f}% ({v1h_sum:,} rows)")
    out_lines.append("")

    # Coupling Test Analysis
    all_is_fraud = np.concatenate(
        [s["is_fraud"].to_numpy() for s in split_dict.values()]
    )
    high_vel_mask = all_v1h >= 3
    base_frate = np.mean(all_is_fraud) * 100
    if np.sum(high_vel_mask) > 0:
        high_vel_frate = np.mean(all_is_fraud[high_vel_mask]) * 100
        coupling_lift = high_vel_frate / base_frate if base_frate > 0 else 0.0
    else:
        high_vel_frate = 0.0
        coupling_lift = 0.0

    out_lines.append("### 3.3 Historical Velocity Coupling Metric")
    out_lines.append(f"- Base Fraud Rate (All Rows)      : {base_frate:.3f}%")
    out_lines.append(f"- Rows with txn_count_1h >= 3     : {np.sum(high_vel_mask):,}")
    hv_sum = np.sum(all_is_fraud[high_vel_mask])
    out_lines.append(
        f"- Fraud Rate (txn_count_1h >= 3)  : {high_vel_frate:.3f}% ({hv_sum:,} frauds)"
    )
    out_lines.append(f"- Measured Velocity Coupling Lift : {coupling_lift:.2f}x")
    out_lines.append("")

    # 5. Lift Tables Covering ALL Rows of Each Split (No Truncation)
    out_lines.append(
        "## 4. Full Lift Tables by Split (No Truncation, Strict Sum Verification)"
    )
    out_lines.append("-" * 80)
    out_lines.append(
        "Every row of every split is accounted for. "
        "Bucket counts sum strictly to split size."
    )
    out_lines.append("")

    for s_name, s_df in split_dict.items():
        out_lines.append("=" * 80)
        out_lines.append(f"### Split: {s_name} (Total Rows: {len(s_df):,})")
        out_lines.append("=" * 80)

        # 1. merchant_risk_tier
        t_md, _ = compute_lift_table(
            s_df, "merchant_risk_tier", s_df["merchant_risk_tier"], s_name
        )
        out_lines.append(t_md)

        # 2. channel
        t_md, _ = compute_lift_table(s_df, "channel", s_df["channel"], s_name)
        out_lines.append(t_md)

        # 3. txn_hour (Night 00-05 vs Day 06-23)
        hour_buckets = pd.Series(
            np.where(
                s_df["txn_hour"].isin([0, 1, 2, 3, 4, 5]),
                "Night (00-05)",
                "Day (06-23)",
            ),
            index=s_df.index,
        )
        t_md, _ = compute_lift_table(s_df, "hour_period", hour_buckets, s_name)
        out_lines.append(t_md)

        # 4. is_international
        intl_buckets = pd.Series(
            np.where(s_df["is_international"], "International", "Domestic"),
            index=s_df.index,
        )
        t_md, _ = compute_lift_table(s_df, "is_international", intl_buckets, s_name)
        out_lines.append(t_md)

        # 5. txn_count_1h
        v1h_raw = s_df["txn_count_1h"]
        v1h_b = pd.Series(
            np.where(
                v1h_raw == 0,
                "0 txns in 1h",
                np.where(
                    v1h_raw == 1,
                    "1 txn in 1h",
                    np.where(v1h_raw == 2, "2 txns in 1h", "3+ txns in 1h"),
                ),
            ),
            index=s_df.index,
        )
        t_md, _ = compute_lift_table(s_df, "txn_count_1h_buckets", v1h_b, s_name)
        out_lines.append(t_md)

        # 6. txn_count_24h
        v24_raw = s_df["txn_count_24h"]
        v24_b = pd.Series(
            np.where(
                v24_raw == 0,
                "0 txns in 24h",
                np.where(
                    v24_raw == 1,
                    "1 txn in 24h",
                    np.where(
                        v24_raw == 2,
                        "2 txns in 24h",
                        np.where(v24_raw <= 5, "3-5 txns in 24h", "6+ txns in 24h"),
                    ),
                ),
            ),
            index=s_df.index,
        )
        t_md, _ = compute_lift_table(s_df, "txn_count_24h_buckets", v24_b, s_name)
        out_lines.append(t_md)

        # 7. is_new_merchant_for_card
        new_m_b = pd.Series(
            np.where(
                s_df["is_new_merchant_for_card"] == 1,
                "New Merchant",
                "Seen Merchant",
            ),
            index=s_df.index,
        )
        t_md, _ = compute_lift_table(s_df, "is_new_merchant_for_card", new_m_b, s_name)
        out_lines.append(t_md)

        # 8. amount_bucket
        amt_raw = s_df["amount"]
        amt_b = pd.Series(
            np.where(
                amt_raw < 10.0,
                "Under $10",
                np.where(
                    amt_raw < 50.0,
                    "$10 - $50",
                    np.where(
                        amt_raw < 250.0,
                        "$50 - $250",
                        np.where(amt_raw < 1000.0, "$250 - $1,000", "$1,000+"),
                    ),
                ),
            ),
            index=s_df.index,
        )
        t_md, _ = compute_lift_table(s_df, "amount_bucket", amt_b, s_name)
        out_lines.append(t_md)

    content = "\n".join(out_lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(
        f"\n[DONE] Fraud data diagnostic audit successfully "
        f"written to: {output_path}"
    )
    print(f"File Size: {len(content):,} bytes, {len(out_lines):,} lines.")


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "eval/results/fraud_data_audit.txt"
    run_audit(out_file)
