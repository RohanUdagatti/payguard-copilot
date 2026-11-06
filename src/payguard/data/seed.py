"""High-performance synthetic data seeder for PayGuard Phase 1 schema."""

from __future__ import annotations

import datetime
import hashlib
import io
import logging
import time
from typing import Any

import numpy as np
import psycopg

from payguard.config import get_settings

logger = logging.getLogger("payguard.data.seed")

# Representative synthetic reason codes (Visa & Mastercard)
SYNTHETIC_REASON_CODES = [
    {
        "reason_code": "10.4",
        "network": "VISA",
        "description": "Other Fraud: Card-Absent Environment",
        "category": "FRAUD",
    },
    {
        "reason_code": "10.5",
        "network": "VISA",
        "description": "Visa Fraud Monitoring Program",
        "category": "FRAUD",
    },
    {
        "reason_code": "13.1",
        "network": "VISA",
        "description": "Merchandise / Services Not Received",
        "category": "CUSTOMER_DISPUTE",
    },
    {
        "reason_code": "13.3",
        "network": "VISA",
        "description": "Not as Described or Defective Merchandise",
        "category": "CUSTOMER_DISPUTE",
    },
    {
        "reason_code": "12.1",
        "network": "VISA",
        "description": "Late Presentment",
        "category": "PROCESSING_ERROR",
    },
    {
        "reason_code": "11.3",
        "network": "VISA",
        "description": "No Authorization",
        "category": "AUTHORIZATION",
    },
    {
        "reason_code": "4837",
        "network": "MASTERCARD",
        "description": "No Cardholder Authorization",
        "category": "FRAUD",
    },
    {
        "reason_code": "4853",
        "network": "MASTERCARD",
        "description": "Cardholder Dispute (Defective / Not as Described)",
        "category": "CUSTOMER_DISPUTE",
    },
    {
        "reason_code": "4834",
        "network": "MASTERCARD",
        "description": "Point-of-Interaction Error",
        "category": "PROCESSING_ERROR",
    },
    {
        "reason_code": "4808",
        "network": "MASTERCARD",
        "description": "Authorization-Related Chargeback",
        "category": "AUTHORIZATION",
    },
]

# Merchant category codes, risk tiers and weights
MCC_PRESETS = [
    ("5411", "Grocery Stores / Supermarkets", "LOW", 0.22),
    ("5812", "Eating Places / Restaurants", "LOW", 0.20),
    ("5732", "Electronic Sales", "MEDIUM", 0.12),
    ("5967", "Direct Marketing / Inbound Telemarketing", "HIGH", 0.05),
    ("7995", "Betting / Casino / Gambling", "HIGH", 0.04),
    ("4722", "Travel Agencies / Tour Operators", "MEDIUM", 0.08),
    ("5311", "Department Stores", "LOW", 0.14),
    ("6051", "Quasi Cash / Crypto / Money Orders", "HIGH", 0.03),
    ("5651", "Family Clothing Stores", "LOW", 0.07),
    ("5999", "Miscellaneous Retail", "MEDIUM", 0.05),
]


def _get_connection_string(url: str | None = None) -> str:
    """Return a psycopg-compatible connection string."""
    if url:
        return url.replace("postgresql+psycopg://", "postgresql://", 1)
    s = get_settings()
    return (
        f"host={s.postgres_host} port={s.postgres_port} "
        f"dbname={s.postgres_db} user={s.postgres_user} "
        f"password={s.postgres_password}"
    )


def truncate_all_tables(conn: psycopg.Connection[Any]) -> None:
    """Truncate tables with CASCADE in reverse dependency order."""
    query = (
        "TRUNCATE TABLE disputes, transactions, cards, merchants, "
        "reason_codes CASCADE;"
    )
    with conn.cursor() as cur:
        cur.execute(query)
    conn.commit()


def seed_reason_codes(conn: psycopg.Connection[Any]) -> int:
    """Seed representative synthetic reason codes."""
    sql = """
        INSERT INTO reason_codes (reason_code, network, description, category)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (reason_code) DO UPDATE
        SET network = EXCLUDED.network,
            description = EXCLUDED.description,
            category = EXCLUDED.category;
    """
    with conn.cursor() as cur:
        for rc in SYNTHETIC_REASON_CODES:
            cur.execute(
                sql,
                (
                    rc["reason_code"],
                    rc["network"],
                    rc["description"],
                    rc["category"],
                ),
            )
    conn.commit()
    return len(SYNTHETIC_REASON_CODES)


def seed_merchants(
    conn: psycopg.Connection[Any], num_merchants: int, seed: int = 42
) -> list[dict[str, Any]]:
    """Seed synthetic merchants using streaming COPY."""
    rng = np.random.default_rng(seed)
    mcc_codes = [m[0] for m in MCC_PRESETS]
    mcc_probs = [m[3] for m in MCC_PRESETS]
    mcc_to_risk = {m[0]: m[2] for m in MCC_PRESETS}
    mcc_to_name = {m[0]: m[1] for m in MCC_PRESETS}

    countries = ["US", "GB", "CA", "DE", "FR", "AU", "JP"]
    country_probs = [0.85, 0.04, 0.04, 0.02, 0.02, 0.015, 0.015]

    chosen_mccs = rng.choice(mcc_codes, p=mcc_probs, size=num_merchants)
    chosen_countries = rng.choice(countries, p=country_probs, size=num_merchants)

    base_time = datetime.datetime(2025, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)
    merchants: list[dict[str, Any]] = []

    buf = io.StringIO()
    for i in range(num_merchants):
        m_id = f"m_{chosen_mccs[i]}_{i:06d}"
        risk = mcc_to_risk[chosen_mccs[i]]
        name = f"{mcc_to_name[chosen_mccs[i]]} #{i:04d}"
        country = chosen_countries[i]
        created = base_time + datetime.timedelta(days=int(rng.integers(0, 365)))
        created_str = created.isoformat()

        merchants.append(
            {
                "merchant_id": m_id,
                "name": name,
                "mcc": chosen_mccs[i],
                "risk_tier": risk,
                "country": country,
                "created_at": created_str,
            }
        )
        buf.write(
            f"{m_id}\t{name}\t{chosen_mccs[i]}\t{risk}\t{country}\t{created_str}\n"
        )

    buf.seek(0)
    copy_sql = (
        "COPY merchants (merchant_id, name, mcc, risk_tier, country, created_at) "
        "FROM STDIN"
    )
    with conn.cursor() as cur, cur.copy(copy_sql) as copy:
        copy.write(buf.read())
    conn.commit()
    return merchants


def seed_cards(
    conn: psycopg.Connection[Any], num_cards: int, seed: int = 42
) -> list[dict[str, Any]]:
    """Seed synthetic card accounts with zero raw PAN."""
    rng = np.random.default_rng(seed + 1)
    brands = ["VISA", "MASTERCARD", "AMEX", "DISCOVER"]
    brand_probs = [0.50, 0.35, 0.10, 0.05]

    types = ["CREDIT", "DEBIT", "PREPAID"]
    type_probs = [0.60, 0.35, 0.05]

    countries = ["US", "GB", "CA", "DE", "FR"]
    country_probs = [0.90, 0.03, 0.03, 0.02, 0.02]

    chosen_brands = rng.choice(brands, p=brand_probs, size=num_cards)
    chosen_types = rng.choice(types, p=type_probs, size=num_cards)
    chosen_countries = rng.choice(countries, p=country_probs, size=num_cards)

    base_time = datetime.datetime(2025, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)
    cards: list[dict[str, Any]] = []

    buf = io.StringIO()
    for i in range(num_cards):
        raw_hash = hashlib.sha256(f"salt_{seed}_{i}".encode()).hexdigest()[:20]
        token = f"tok_{chosen_brands[i].lower()[:4]}_{raw_hash}"
        user_id = f"usr_{i:08d}"
        created = base_time + datetime.timedelta(days=int(rng.integers(0, 365)))
        created_str = created.isoformat()

        cards.append(
            {
                "card_token": token,
                "user_id": user_id,
                "card_brand": chosen_brands[i],
                "card_type": chosen_types[i],
                "issuer_country": chosen_countries[i],
                "created_at": created_str,
            }
        )
        row_str = (
            f"{token}\t{user_id}\t{chosen_brands[i]}\t"
            f"{chosen_types[i]}\t{chosen_countries[i]}\t{created_str}\n"
        )
        buf.write(row_str)

    buf.seek(0)
    copy_sql = (
        "COPY cards (card_token, user_id, card_brand, card_type, "
        "issuer_country, created_at) FROM STDIN"
    )
    with conn.cursor() as cur, cur.copy(copy_sql) as copy:
        copy.write(buf.read())
    conn.commit()
    return cards


def seed_transactions_and_disputes(
    conn: psycopg.Connection[Any],
    num_transactions: int,
    merchants: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    seed: int = 42,
    batch_size: int = 50000,
) -> tuple[int, int, float, float]:
    """Generate and streaming-copy transactions and disputes."""
    rng = np.random.default_rng(seed + 2)
    num_cards = len(cards)
    num_merchants = len(merchants)

    # Power-law card weights to simulate real card activity
    alpha = 1.3
    weights = 1.0 / (np.arange(1, num_cards + 1) ** alpha)
    weights /= weights.sum()

    merchant_ids = [m["merchant_id"] for m in merchants]
    merchant_mccs = [m["mcc"] for m in merchants]
    merchant_risks = [m["risk_tier"] for m in merchants]
    merchant_countries = [m["country"] for m in merchants]

    card_tokens = [c["card_token"] for c in cards]
    card_countries = [c["issuer_country"] for c in cards]

    channels = ["POS", "ECOMMERCE", "ATM", "MOTO"]
    channel_probs = [0.60, 0.32, 0.05, 0.03]

    statuses = ["APPROVED", "DECLINED", "REFUNDED"]
    status_probs = [0.95, 0.04, 0.01]

    disp_statuses = ["OPEN", "UNDER_REVIEW", "WON", "LOST"]
    disp_status_probs = [0.40, 0.30, 0.15, 0.15]

    fraud_reasons = ["10.4", "10.5", "4837"]
    non_fraud_reasons = ["13.1", "13.3", "12.1", "11.3", "4853", "4834"]

    start_date = datetime.datetime(2026, 8, 1, 0, 0, 0, tzinfo=datetime.UTC)
    total_seconds = 90 * 86400  # 90-day window

    total_fraud_count = 0
    total_dispute_count = 0

    num_batches = (num_transactions + batch_size - 1) // batch_size

    for b in range(num_batches):
        cur_batch_size = min(batch_size, num_transactions - b * batch_size)
        batch_offset = b * batch_size

        c_indices = rng.choice(num_cards, p=weights, size=cur_batch_size)
        m_indices = rng.integers(0, num_merchants, size=cur_batch_size)

        time_offsets = rng.integers(0, total_seconds, size=cur_batch_size)
        time_offsets.sort()

        batch_channels = rng.choice(channels, p=channel_probs, size=cur_batch_size)
        batch_statuses = rng.choice(statuses, p=status_probs, size=cur_batch_size)
        batch_amounts = np.round(
            rng.exponential(scale=65.0, size=cur_batch_size) + 5.0, 2
        )
        outlier_mask = rng.random(size=cur_batch_size) < 0.02
        if outlier_mask.any():
            add_vals = rng.uniform(400, 2500, size=outlier_mask.sum())
            batch_amounts[outlier_mask] = np.round(
                batch_amounts[outlier_mask] + add_vals, 2
            )

        batch_velocities = rng.poisson(1.5, size=cur_batch_size)
        batch_is_new_merch = rng.random(size=cur_batch_size) < 0.20

        txn_buf = io.StringIO()
        disp_buf = io.StringIO()

        for i in range(cur_batch_size):
            global_idx = batch_offset + i
            c_idx = c_indices[i]
            m_idx = m_indices[i]

            t_token = card_tokens[c_idx]
            t_m_id = merchant_ids[m_idx]
            t_mcc = merchant_mccs[m_idx]
            t_risk = merchant_risks[m_idx]
            t_is_intl = card_countries[c_idx] != merchant_countries[m_idx]

            txn_dt = start_date + datetime.timedelta(seconds=int(time_offsets[i]))
            txn_hour = txn_dt.hour
            txn_dt_str = txn_dt.isoformat()

            amt = batch_amounts[i]
            chan = batch_channels[i]
            stat = batch_statuses[i]

            night_sig = 1.3 if txn_hour in (0, 1, 2, 3, 4, 5) else -0.3
            risk_sig = (
                1.6 if t_risk == "HIGH" else (0.4 if t_risk == "MEDIUM" else -0.6)
            )
            chan_sig = 1.1 if chan in ("ECOMMERCE", "MOTO") else -0.7
            intl_sig = 1.4 if t_is_intl else -0.2
            new_merch_sig = 0.8 if batch_is_new_merch[i] else -0.2
            vel_sig = 1.3 if batch_velocities[i] > 4 else -0.2
            amt_sig = 1.2 if amt > 500 else -0.3

            feature_score = (
                night_sig
                + risk_sig
                + chan_sig
                + intl_sig
                + new_merch_sig
                + vel_sig
                + amt_sig
            )
            noise = float(rng.normal(0, 1.2))
            logit = -6.8 + 0.55 * feature_score + noise
            p_feature = 1.0 / (1.0 + np.exp(-logit))

            is_feature_fraud = bool(rng.random() < p_feature)
            is_stealth_fraud = bool(rng.random() < 0.0008)
            is_fraud = is_feature_fraud or is_stealth_fraud

            idx_hash = hashlib.md5(f"{global_idx}_{seed}".encode()).hexdigest()[:8]
            txn_id = f"txn_{global_idx:010d}_{idx_hash}"

            if is_fraud:
                total_fraud_count += 1

            txn_row = (
                f"{txn_id}\t{t_token}\t{t_m_id}\t{amt:.2f}\tUSD\t"
                f"{txn_dt_str}\t{txn_hour}\t{chan}\t{t_mcc}\t{t_risk}\t"
                f"{t_is_intl}\t{stat}\t{is_fraud}\t{txn_dt_str}\n"
            )
            txn_buf.write(txn_row)

            p_disp = 0.65 if is_fraud else 0.008
            if rng.random() < p_disp:
                total_dispute_count += 1
                dsp_hash = hashlib.md5(f"dsp_{global_idx}_{seed}".encode()).hexdigest()[
                    :8
                ]
                disp_id = f"dsp_{global_idx:010d}_{dsp_hash}"
                reason = (
                    rng.choice(fraud_reasons)
                    if is_fraud
                    else rng.choice(non_fraud_reasons)
                )
                d_status = rng.choice(disp_statuses, p=disp_status_probs)

                filed_dt = txn_dt + datetime.timedelta(days=int(rng.integers(1, 15)))
                due_dt = filed_dt + datetime.timedelta(days=30)
                resolved_dt_str = "\\N"
                if d_status in ("WON", "LOST"):
                    res_dt = filed_dt + datetime.timedelta(
                        days=int(rng.integers(10, 40))
                    )
                    resolved_dt_str = res_dt.isoformat()

                disp_row = (
                    f"{disp_id}\t{txn_id}\t{amt:.2f}\tUSD\t{reason}\t"
                    f"{d_status}\t{due_dt.isoformat()}\t{filed_dt.isoformat()}\t"
                    f"{resolved_dt_str}\t{filed_dt.isoformat()}\n"
                )
                disp_buf.write(disp_row)

        txn_buf.seek(0)
        copy_txn_sql = (
            "COPY transactions (transaction_id, card_token, merchant_id, "
            "amount, currency, timestamp, txn_hour, channel, mcc, risk_tier, "
            "is_international, status, is_fraud, created_at) FROM STDIN"
        )
        with conn.cursor() as cur, cur.copy(copy_txn_sql) as copy:
            copy.write(txn_buf.read())

        disp_buf.seek(0)
        if disp_buf.getvalue():
            copy_disp_sql = (
                "COPY disputes (dispute_id, transaction_id, amount, currency, "
                "reason_code, status, evidence_due_date, filed_at, resolved_at, "
                "created_at) FROM STDIN (NULL '\\N')"
            )
            with conn.cursor() as cur, cur.copy(copy_disp_sql) as copy:
                copy.write(disp_buf.read())

        conn.commit()

    fraud_rate = total_fraud_count / num_transactions if num_transactions else 0.0
    dispute_rate = total_dispute_count / num_transactions if num_transactions else 0.0

    return (
        num_transactions,
        total_dispute_count,
        fraud_rate,
        dispute_rate,
    )


def run_seed(
    num_transactions: int = 500000,
    seed: int = 42,
    db_url: str | None = None,
    reset: bool = True,
) -> dict[str, Any]:
    """Execute complete deterministic database seed and report metrics."""
    conn_str = _get_connection_string(db_url)
    start_time = time.time()

    with psycopg.connect(conn_str) as conn:
        if reset:
            truncate_all_tables(conn)

        rc_count = seed_reason_codes(conn)
        num_merchants = min(1000, max(50, num_transactions // 100))
        num_cards = min(25000, max(500, num_transactions // 20))

        merchants = seed_merchants(conn, num_merchants=num_merchants, seed=seed)
        cards = seed_cards(conn, num_cards=num_cards, seed=seed)

        n_txn, n_disp, fraud_rate, dispute_rate = seed_transactions_and_disputes(
            conn=conn,
            num_transactions=num_transactions,
            merchants=merchants,
            cards=cards,
            seed=seed,
        )

        elapsed = time.time() - start_time

        return {
            "elapsed_seconds": elapsed,
            "reason_codes": rc_count,
            "merchants": len(merchants),
            "cards": len(cards),
            "transactions": n_txn,
            "disputes": n_disp,
            "fraud_rate": fraud_rate,
            "dispute_rate": dispute_rate,
        }
