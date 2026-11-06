#!/usr/bin/env python3
"""CLI script to seed PostgreSQL database with high-performance synthetic data."""

from __future__ import annotations

import argparse
import sys
import time

import psycopg

from payguard.data.seed import _get_connection_string, run_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed PostgreSQL database with high-performance synthetic payment data."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500000,
        help="Number of transactions to seed (default: 500000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42).",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not truncate existing table data prior to seeding.",
    )
    args = parser.parse_args()

    print(
        f"Seeding database with {args.count:,} transactions (seed={args.seed})..."
    )
    t0 = time.time()
    metrics = run_seed(
        num_transactions=args.count,
        seed=args.seed,
        reset=not args.no_reset,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 65)
    print("=== SEEDING EXECUTION METRICS ===")
    print("=" * 65)
    print(f"Elapsed Time       : {elapsed:.2f} seconds")
    print(f"Transactions Seeded: {metrics['transactions']:,}")
    print(f"Disputes Generated : {metrics['disputes']:,}")
    print(
        f"Measured Fraud Rate: {metrics['fraud_rate']*100:.3f}% ({int(metrics['transactions']*metrics['fraud_rate']):,} rows)"
    )
    print(f"Measured Disp. Rate: {metrics['dispute_rate']*100:.3f}%")
    print("=" * 65)

    # Verification SQL query
    conn_str = _get_connection_string()
    print("\nVerification SQL query: COUNT(*) per table:")
    print("-" * 65)
    query = """
    SELECT 'merchants' AS table_name, count(*) AS row_count FROM merchants
    UNION ALL SELECT 'cards', count(*) FROM cards
    UNION ALL SELECT 'reason_codes', count(*) FROM reason_codes
    UNION ALL SELECT 'transactions', count(*) FROM transactions
    UNION ALL SELECT 'disputes', count(*) FROM disputes;
    """
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            print(f"{'Table Name':<20} | {'Row Count':>15}")
            print("-" * 40)
            for tbl, cnt in rows:
                print(f"{tbl:<20} | {cnt:>15,}")
    print("-" * 65 + "\n")


if __name__ == "__main__":
    main()

