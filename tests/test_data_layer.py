"""Tests for Phase 1 Data Layer: migrations, determinism, rates, FKs, and Luhn."""

from __future__ import annotations

import hashlib
import re

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from payguard.config import Settings
from payguard.data.db import get_db_url, get_engine
from payguard.data.seed import _get_connection_string, run_seed


def is_luhn_valid(s: str) -> bool:
    """Return True if a 13-19 digit string satisfies the Luhn checksum."""
    if not (13 <= len(s) <= 19 and s.isdigit()):
        return False
    digits = [int(c) for c in s]
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            d = d * 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def test_db_url_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Database URL dynamically derives from discrete environment variables."""
    monkeypatch.setenv("POSTGRES_USER", "custom_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "custom_pwd")
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "custom_db")

    settings = Settings()
    expected_dsn = "postgresql://custom_user:custom_pwd@db.internal:5433/custom_db"
    assert settings.postgres_dsn == expected_dsn

    resolved = get_db_url(settings.postgres_dsn)
    expected_psycopg = (
        "postgresql+psycopg://custom_user:custom_pwd@db.internal:5433/custom_db"
    )
    assert resolved.startswith(expected_psycopg)


def test_migration_up_and_down() -> None:
    """Alembic migrations cleanly downgrade to base and upgrade to head."""
    alembic_cfg = Config("alembic.ini")

    # Upgrade to head
    command.upgrade(alembic_cfg, "head")

    engine = get_engine()
    with engine.connect() as conn:
        res = conn.execute(
            text(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
                """
            )
        ).fetchall()
        table_names = {row[0] for row in res}
        expected = {
            "cards",
            "disputes",
            "merchants",
            "reason_codes",
            "transactions",
            "alembic_version",
        }
        assert expected.issubset(table_names)

    # Downgrade to base
    command.downgrade(alembic_cfg, "base")

    with engine.connect() as conn:
        res = conn.execute(
            text(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
                """
            )
        ).fetchall()
        remaining_tables = {row[0] for row in res}
        app_tables = {
            "cards",
            "disputes",
            "merchants",
            "reason_codes",
            "transactions",
        }
        assert not remaining_tables.intersection(app_tables)

    # Restore head for subsequent tests
    command.upgrade(alembic_cfg, "head")


def test_seed_deterministic_checksums() -> None:
    """Seeding with seed=42 and 10,000 transactions produces identical checksums."""
    conn_str = _get_connection_string()

    def get_dataset_checksum() -> str:
        with psycopg.connect(conn_str) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT transaction_id, card_token, merchant_id,
                       amount, timestamp, is_fraud
                FROM transactions
                ORDER BY transaction_id;
                """
            )
            txn_rows = cur.fetchall()

            cur.execute(
                """
                SELECT dispute_id, transaction_id, amount, reason_code, status
                FROM disputes
                ORDER BY dispute_id;
                """
            )
            disp_rows = cur.fetchall()

        hasher = hashlib.sha256()
        for r in txn_rows:
            hasher.update(str(r).encode())
        for r in disp_rows:
            hasher.update(str(r).encode())
        return hasher.hexdigest()

    # Run 1
    run_seed(num_transactions=10000, seed=42, reset=True)
    checksum_1 = get_dataset_checksum()

    # Run 2 (with reset=True)
    run_seed(num_transactions=10000, seed=42, reset=True)
    checksum_2 = get_dataset_checksum()

    assert (
        checksum_1 == checksum_2
    ), "Seed generation is not deterministic across repeated runs with seed=42"


def test_fraud_and_dispute_rates_within_bounds() -> None:
    """Fraud rate is within 0.2-0.5% and dispute rate is ~1% within bounds."""
    conn_str = _get_connection_string()
    with psycopg.connect(conn_str) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                count(*) AS total_txns,
                count(*) FILTER (WHERE is_fraud) AS fraud_txns
            FROM transactions;
            """
        )
        row = cur.fetchone()
        assert row is not None
        total_txns, fraud_txns = row

        cur.execute("SELECT count(*) FROM disputes;")
        disp_row = cur.fetchone()
        assert disp_row is not None
        total_disp = disp_row[0]

    fraud_rate = fraud_txns / total_txns
    dispute_rate = total_disp / total_txns

    # Assert fraud rate strictly within 0.2% - 0.5% (0.002 - 0.005)
    assert (
        0.002 <= fraud_rate <= 0.005
    ), f"Fraud rate {fraud_rate*100:.3f}% is out of bounds [0.2%, 0.5%]"

    # Assert dispute rate strictly around 1% (e.g. 0.007 - 0.015)
    assert (
        0.007 <= dispute_rate <= 0.015
    ), f"Dispute rate {dispute_rate*100:.3f}% is out of bounds around 1%"


def test_no_legit_card_exceeds_daily_cap() -> None:
    """No legitimate card account exceeds the hard cap of 8 transactions per day."""
    conn_str = _get_connection_string()
    with psycopg.connect(conn_str) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT card_token, timestamp::date AS txn_date, count(*) AS daily_count
            FROM transactions
            WHERE is_fraud = false
            GROUP BY card_token, timestamp::date
            HAVING count(*) > 8;
            """
        )
        violations = cur.fetchall()
        assert (
            not violations
        ), f"Found legit cards exceeding 8 transactions/day: {violations[:5]}"


def test_fraud_bursts_and_typologies_exist() -> None:
    """Bursts and micro-testing fraud typologies are present in the dataset."""
    conn_str = _get_connection_string()
    with psycopg.connect(conn_str) as conn, conn.cursor() as cur:
        # Verify card bursts: cards with >= 2 frauds
        cur.execute(
            """
            SELECT card_token, count(*) AS fraud_count
            FROM transactions
            WHERE is_fraud = true
            GROUP BY card_token
            HAVING count(*) >= 2;
            """
        )
        burst_cards = cur.fetchall()
        assert (
            len(burst_cards) >= 1
        ), "Expected at least one card burst with >= 2 frauds"

        # Verify micro-testing fraud: tiny fraud transactions under $5.00
        cur.execute(
            """
            SELECT count(*)
            FROM transactions
            WHERE is_fraud = true AND amount < 5.00;
            """
        )
        micro_row = cur.fetchone()
        assert micro_row is not None
        micro_count = micro_row[0]
        assert micro_count >= 1, "Expected at least one micro-testing fraud under $5.00"


def test_history_coupling_txn_count_1h() -> None:
    """Fraud rate for rows with txn_count_1h >= 3 is clearly above the base rate."""
    conn_str = _get_connection_string()
    with psycopg.connect(conn_str) as conn, conn.cursor() as cur:
        cur.execute(
            """
            WITH ordered_txns AS (
                SELECT
                    transaction_id,
                    card_token,
                    timestamp,
                    is_fraud,
                    count(*) OVER (
                        PARTITION BY card_token
                        ORDER BY timestamp
                        RANGE BETWEEN INTERVAL '1 hour' PRECEDING
                            AND INTERVAL '1 microsecond' PRECEDING
                    ) AS prior_count_1h
                FROM transactions
            )
            SELECT
                count(*) AS total_txns,
                count(*) FILTER (WHERE is_fraud) AS total_frauds,
                count(*) FILTER (WHERE prior_count_1h >= 3) AS high_vel_txns,
                count(*) FILTER (
                    WHERE prior_count_1h >= 3 AND is_fraud
                ) AS high_vel_frauds
            FROM ordered_txns;
            """
        )
        row = cur.fetchone()
        assert row is not None
        total_txns, total_frauds, high_vel_txns, high_vel_frauds = row

    base_rate = total_frauds / total_txns if total_txns else 0.0
    assert (
        high_vel_txns >= 1
    ), "Expected at least one transaction with prior_count_1h >= 3"
    high_vel_rate = high_vel_frauds / high_vel_txns
    lift = high_vel_rate / base_rate if base_rate else 0.0
    msg = (
        f"Expected velocity coupling lift >= 3.0x, but got {lift:.2f}x "
        f"(base={base_rate*100:.3f}%, high_vel={high_vel_rate*100:.3f}%)"
    )
    assert lift >= 3.0, msg


def test_foreign_key_integrity() -> None:
    """All foreign key references are valid, and invalid references are rejected."""
    conn_str = _get_connection_string()
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            # 1. No orphan transactions (cards)
            cur.execute(
                """
                SELECT count(*) FROM transactions t
                LEFT JOIN cards c ON t.card_token = c.card_token
                WHERE c.card_token IS NULL;
                """
            )
            assert cur.fetchone()[0] == 0

            # 2. No orphan transactions (merchants)
            cur.execute(
                """
                SELECT count(*) FROM transactions t
                LEFT JOIN merchants m ON t.merchant_id = m.merchant_id
                WHERE m.merchant_id IS NULL;
                """
            )
            assert cur.fetchone()[0] == 0

            # 3. No orphan disputes (transactions)
            cur.execute(
                """
                SELECT count(*) FROM disputes d
                LEFT JOIN transactions t ON d.transaction_id = t.transaction_id
                WHERE t.transaction_id IS NULL;
                """
            )
            assert cur.fetchone()[0] == 0

            # 4. No orphan disputes (reason codes)
            cur.execute(
                """
                SELECT count(*) FROM disputes d
                LEFT JOIN reason_codes r ON d.reason_code = r.reason_code
                WHERE r.reason_code IS NULL;
                """
            )
            assert cur.fetchone()[0] == 0

        # Attempt to insert orphan transaction to verify DB enforcement
        with pytest.raises(psycopg.IntegrityError), conn.cursor() as cur:
            bad_insert = """
                INSERT INTO transactions (
                    transaction_id, card_token, merchant_id, amount, currency,
                    timestamp, txn_hour, channel, mcc, risk_tier,
                    is_international, status, is_fraud, created_at
                ) VALUES (
                    'txn_invalid_fk', 'tok_nonexistent_card', 'm_nonexistent',
                    10.00, 'USD', NOW(), 12, 'POS', '5411', 'LOW', false,
                    'APPROVED', false, NOW()
                );
            """
            cur.execute(bad_insert)


def test_no_luhn_valid_values_in_any_column() -> None:
    """Zero 13-19 digit Luhn-valid values exist in any text column across all tables."""
    conn_str = _get_connection_string()
    luhn_digits_re = re.compile(r"\b\d{13,19}\b")
    luhn_violations: list[str] = []

    tables_and_cols = [
        (
            "cards",
            [
                "card_token",
                "user_id",
                "card_brand",
                "card_type",
                "issuer_country",
            ],
        ),
        ("merchants", ["merchant_id", "name", "mcc", "risk_tier", "country"]),
        ("reason_codes", ["reason_code", "network", "description", "category"]),
        (
            "transactions",
            [
                "transaction_id",
                "card_token",
                "merchant_id",
                "currency",
                "channel",
                "mcc",
                "risk_tier",
                "status",
            ],
        ),
        (
            "disputes",
            [
                "dispute_id",
                "transaction_id",
                "currency",
                "reason_code",
                "status",
            ],
        ),
    ]

    with psycopg.connect(conn_str) as conn, conn.cursor() as cur:
        for tbl, cols in tables_and_cols:
            col_expr = ", ".join(f"COALESCE({c}::text, '')" for c in cols)
            cur.execute(f"SELECT {col_expr} FROM {tbl};")
            rows = cur.fetchall()
            for row in rows:
                for val in row:
                    matches = luhn_digits_re.findall(val)
                    for m in matches:
                        if is_luhn_valid(m):
                            luhn_violations.append(f"{tbl}.val={val} matched={m}")

    assert (
        not luhn_violations
    ), f"Found Luhn-valid numbers in database: {luhn_violations[:5]}"
