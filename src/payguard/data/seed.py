"""High-performance synthetic data seeder for PayGuard Phase 1 schema (Generator V2)."""

from __future__ import annotations

import datetime
import hashlib
import io
import json
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

# Immutable parameter specification for Generator V2
GENERATOR_V2_CONFIG: dict[str, Any] = {
    "version": "2.0.0",
    "legit_activity": {
        "lognormal_mu": -1.35,
        "lognormal_sigma": 0.65,
        "daily_hard_cap": 8,
        "min_daily_rate": 0.01,
        "max_daily_rate": 7.5,
        "diurnal_standard": {
            "night_weight": 0.005,  # hours 00-05 (~3% total)
            "morning_weight": 0.035,  # hours 06-09 (~14% total)
            "day_weight": 0.065,  # hours 10-19 (~65% total)
            "evening_weight": 0.045,  # hours 20-23 (~18% total)
        },
        "night_owl_fraction": 0.04,
        "traveller_fraction": 0.03,
        "traveller_duration_days": [3, 7],
        "favorite_merchants_per_card": [5, 15],
        "favorite_merchant_probability": 0.75,
        "channels": {
            "POS": 0.60,
            "ECOMMERCE": 0.32,
            "ATM": 0.05,
            "MOTO": 0.03,
        },
        "amount_base_scale": 50.0,
        "amount_base_min": 5.0,
        "amount_outlier_prob": 0.02,
        "amount_outlier_range": [200.0, 1000.0],
    },
    "fraud_typologies": {
        "compromised_bursts": {
            "target_events": 120,
            "burst_size": [3, 7],
            "duration_hours": [1.0, 5.0],
            "channels": {"ECOMMERCE": 0.60, "MOTO": 0.25, "POS": 0.15},
            "amount_range": [35.0, 1200.0],
            "prefer_non_favorite": True,
        },
        "card_testing": {
            "target_events": 60,
            "micro_count": [4, 8],
            "duration_seconds": [10, 400],
            "micro_amount_range": [0.50, 4.99],
            "followup_large_prob": 0.50,
            "followup_large_amount": [500.0, 2200.0],
            "channel": "ECOMMERCE",
            "merchant_risk": "HIGH",
        },
        "geographic_jump": {
            "target_events": 260,
            "time_gap_seconds": [1800, 4800],
            "amount_range": [150.0, 1100.0],
            "channels": {"POS": 0.70, "ECOMMERCE": 0.30},
        },
        "opportunistic_high_value": {
            "target_events": 180,
            "night_hours": [0, 5],
            "amount_range": [600.0, 3500.0],
            "channels": {"ECOMMERCE": 0.60, "MOTO": 0.40},
            "prefer_non_favorite": True,
        },
        "stealth_fraud": {
            "target_events": 360,
            "daytime_hours": [10, 18],
            "amount_range": [15.0, 85.0],
            "channels": {"POS": 0.75, "ECOMMERCE": 0.25},
            "merchant_risks": ["LOW", "MEDIUM"],
        },
    },
    "disputes": {
        "fraud_dispute_prob": 0.65,
        "legit_dispute_prob": 0.0077,
        "fraud_reasons": ["10.4", "10.5", "4837"],
        "non_fraud_reasons": [
            "13.1",
            "13.3",
            "12.1",
            "11.3",
            "4853",
            "4834",
        ],
        "status_distribution": {
            "OPEN": 0.40,
            "UNDER_REVIEW": 0.30,
            "WON": 0.15,
            "LOST": 0.15,
        },
    },
}

GENERATOR_PARAMS_HASH = hashlib.sha256(
    json.dumps(GENERATOR_V2_CONFIG, sort_keys=True).encode("utf-8")
).hexdigest()


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
    conn: psycopg.Connection[Any], num_merchants: int = 1000, seed: int = 42
) -> list[dict[str, Any]]:
    """Seed synthetic merchant accounts with category codes and risk tiers."""
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
    """Generate and streaming-copy transactions and disputes using Generator V2."""
    rng = np.random.default_rng(seed + 2)
    num_cards = len(cards)
    num_merchants = len(merchants)
    days = 90
    total_seconds = days * 86400
    start_date = datetime.datetime(2026, 8, 1, 0, 0, 0, tzinfo=datetime.UTC)

    merchant_ids = [m["merchant_id"] for m in merchants]
    merchant_mccs = [m["mcc"] for m in merchants]
    merchant_risks = [m["risk_tier"] for m in merchants]
    merchant_countries = [m["country"] for m in merchants]

    card_tokens = [c["card_token"] for c in cards]
    card_countries = [c["issuer_country"] for c in cards]

    # Pre-index high-risk online merchants for card testing
    high_risk_online = [m for m in range(num_merchants) if merchant_risks[m] == "HIGH"]
    if not high_risk_online:
        high_risk_online = list(range(num_merchants))

    low_med_merchants = [
        m for m in range(num_merchants) if merchant_risks[m] in ("LOW", "MEDIUM")
    ]
    if not low_med_merchants:
        low_med_merchants = list(range(num_merchants))

    # 1. Card Personas and Favorites
    is_night_owl = rng.random(size=num_cards) < 0.04
    is_traveller = rng.random(size=num_cards) < 0.03
    travel_start_day = rng.integers(10, 80, size=num_cards)
    travel_dur = rng.integers(3, 8, size=num_cards)
    travel_countries = ["GB", "FR", "DE", "JP", "AU"]
    chosen_travel_country = rng.choice(travel_countries, size=num_cards)

    card_favs: list[list[int]] = []
    card_favs_set: list[set[int]] = []
    for _ in range(num_cards):
        k_fav = int(rng.integers(5, 16))
        k_fav = min(k_fav, num_merchants)
        favs = list(rng.choice(num_merchants, size=k_fav, replace=False))
        card_favs.append(favs)
        card_favs_set.append(set(favs))

    # Scale fraud events proportionally with requested transaction count
    scale = num_transactions / 500000.0
    n_burst_events = max(1, int(round(120 * scale)))
    n_testing_events = max(1, int(round(60 * scale)))
    n_geo_events = max(1, int(round(260 * scale)))
    n_opp_events = max(1, int(round(180 * scale)))
    n_stealth_events = max(1, int(round(360 * scale)))

    fraud_events: list[tuple[int, int, int, float, str, bool]] = []

    # Typology 1: Compromised-card bursts
    # (2-8 transactions in 1-6 hours, escalating amounts)
    burst_cards = rng.choice(num_cards, size=n_burst_events, replace=False)
    for c in burst_cards:
        k_burst = int(rng.integers(3, 8))
        t_start = int(rng.integers(0, total_seconds - 21600))
        t_span = int(rng.integers(3600, 18000))
        t_offsets = np.sort(rng.integers(0, t_span, size=k_burst))
        amts = np.sort(np.round(rng.uniform(35.0, 1200.0, size=k_burst), 2))
        non_favs = [m for m in range(num_merchants) if m not in card_favs_set[c]]
        if not non_favs:
            non_favs = list(range(num_merchants))
        m_chosen = rng.choice(non_favs, size=k_burst)
        chans = rng.choice(
            ["ECOMMERCE", "MOTO", "POS"], p=[0.60, 0.25, 0.15], size=k_burst
        )
        for j in range(k_burst):
            fraud_events.append(
                (
                    int(c),
                    int(m_chosen[j]),
                    int(t_start + t_offsets[j]),
                    float(amts[j]),
                    str(chans[j]),
                    True,
                )
            )

    # Typology 2: Card testing (rapid micro-transactions, then sometimes a large hit)
    test_cards = rng.choice(num_cards, size=n_testing_events, replace=False)
    for c in test_cards:
        k_test = int(rng.integers(4, 9))
        t_start = int(rng.integers(0, total_seconds - 3600))
        t_offsets = np.sort(rng.integers(10, 400, size=k_test))
        amts = np.round(rng.uniform(0.50, 4.99, size=k_test), 2)
        m_chosen = rng.choice(high_risk_online, size=k_test)
        for j in range(k_test):
            fraud_events.append(
                (
                    int(c),
                    int(m_chosen[j]),
                    int(t_start + t_offsets[j]),
                    float(amts[j]),
                    "ECOMMERCE",
                    True,
                )
            )
        if rng.random() < 0.50:
            big_amt = round(float(rng.uniform(500.0, 2200.0)), 2)
            fraud_events.append(
                (
                    int(c),
                    int(rng.choice(high_risk_online)),
                    int(t_start + t_offsets[-1] + rng.integers(600, 1800)),
                    big_amt,
                    "ECOMMERCE",
                    True,
                )
            )

    # Typology 3: Geographic jump (foreign transaction shortly following a domestic one)
    geo_cards = rng.choice(num_cards, size=n_geo_events, replace=False)
    for c in geo_cards:
        d_geo = int(rng.integers(0, days))
        h_geo = int(rng.integers(8, 20))
        sec_geo = d_geo * 86400 + h_geo * 3600 + int(rng.integers(0, 3600))
        amt_geo = round(float(rng.uniform(150.0, 1100.0)), 2)
        foreign_m = [
            m
            for m in range(num_merchants)
            if merchant_countries[m] != card_countries[c]
        ]
        if not foreign_m:
            foreign_m = list(range(num_merchants))
        fraud_events.append(
            (
                int(c),
                int(rng.choice(foreign_m)),
                sec_geo,
                amt_geo,
                str(rng.choice(["POS", "ECOMMERCE"], p=[0.70, 0.30])),
                True,
            )
        )

    # Typology 4: Opportunistic single high-value fraud
    opp_cards = rng.choice(num_cards, size=n_opp_events, replace=False)
    for c in opp_cards:
        d_opp = int(rng.integers(0, days))
        h_opp = int(rng.integers(0, 6))
        sec_opp = d_opp * 86400 + h_opp * 3600 + int(rng.integers(0, 3600))
        amt_opp = round(float(rng.uniform(600.0, 3500.0)), 2)
        non_favs = [m for m in range(num_merchants) if m not in card_favs_set[c]]
        if not non_favs:
            non_favs = list(range(num_merchants))
        fraud_events.append(
            (
                int(c),
                int(rng.choice(non_favs)),
                sec_opp,
                amt_opp,
                str(rng.choice(["ECOMMERCE", "MOTO"], p=[0.60, 0.40])),
                True,
            )
        )

    # Typology 5: Stealth fraud (looks normal on observable features)
    stealth_cards = rng.choice(num_cards, size=n_stealth_events, replace=False)
    for c in stealth_cards:
        d_stl = int(rng.integers(0, days))
        h_stl = int(rng.integers(10, 19))
        sec_stl = d_stl * 86400 + h_stl * 3600 + int(rng.integers(0, 3600))
        amt_stl = round(float(rng.uniform(15.0, 85.0)), 2)
        fraud_events.append(
            (
                int(c),
                int(rng.choice(low_med_merchants)),
                sec_stl,
                amt_stl,
                str(rng.choice(["POS", "ECOMMERCE"], p=[0.75, 0.25])),
                True,
            )
        )

    n_fraud = len(fraud_events)
    target_legit = max(0, num_transactions - n_fraud)

    # 2. Legit activity generation with lognormal rate and hard cap = 8
    mu = -1.35
    sigma = 0.65
    raw_rates = np.clip(rng.lognormal(mu, sigma, size=num_cards), 0.01, 7.5)
    day_weights_pattern = np.array(
        [0.98, 0.98, 0.98, 1.0, 1.02, 1.05, 1.02], dtype=float
    )
    day_weights = np.array(
        [day_weights_pattern[d % 7] for d in range(days)], dtype=float
    )

    daily_counts = np.minimum(rng.poisson(raw_rates[:, None] * day_weights[None, :]), 8)

    diff = int(daily_counts.sum()) - target_legit
    flat_counts = daily_counts.ravel()
    if diff > 0:
        flat_active = np.where(flat_counts > 0)[0]
        trim_indices = rng.choice(flat_active, size=diff, replace=False)
        np.subtract.at(flat_counts, trim_indices, 1)
    elif diff < 0:
        flat_can_add = np.where(flat_counts < 8)[0]
        add_indices = rng.choice(flat_can_add, size=abs(diff), replace=False)
        np.add.at(flat_counts, add_indices, 1)
    daily_counts = flat_counts.reshape((num_cards, days))

    card_indices, day_indices = np.where(daily_counts > 0)
    repeats = daily_counts[card_indices, day_indices]
    c_legit = np.repeat(card_indices, repeats)
    d_legit = np.repeat(day_indices, repeats)
    n_l = len(c_legit)

    std_hour_probs = np.array(
        [0.005] * 6 + [0.035] * 4 + [0.065] * 10 + [0.045] * 4, dtype=float
    )
    std_hour_probs /= std_hour_probs.sum()
    owl_hour_probs = np.array([0.09] * 6 + [0.0175] * 16 + [0.09] * 2, dtype=float)
    owl_hour_probs /= owl_hour_probs.sum()

    h_std = rng.choice(24, size=n_l, p=std_hour_probs)
    h_owl = rng.choice(24, size=n_l, p=owl_hour_probs)
    h_legit = np.where(is_night_owl[c_legit], h_owl, h_std)

    m_legit = rng.integers(0, 60, size=n_l)
    s_legit = rng.integers(0, 60, size=n_l)
    time_offsets_legit = d_legit * 86400 + h_legit * 3600 + m_legit * 60 + s_legit

    use_fav = rng.random(size=n_l) < 0.75
    fav_slot = rng.integers(0, 5, size=n_l)
    general_m = rng.integers(0, num_merchants, size=n_l)

    m_legit_arr = np.empty(n_l, dtype=np.int32)
    for i in range(n_l):
        c_i = c_legit[i]
        d_i = d_legit[i]
        # Check traveller persona
        if (
            is_traveller[c_i]
            and travel_start_day[c_i] <= d_i < travel_start_day[c_i] + travel_dur[c_i]
        ):
            dest = chosen_travel_country[c_i]
            dest_merchants = [
                m for m in range(num_merchants) if merchant_countries[m] == dest
            ]
            if dest_merchants:
                m_legit_arr[i] = dest_merchants[fav_slot[i] % len(dest_merchants)]
            else:
                m_legit_arr[i] = general_m[i]
        elif use_fav[i]:
            favs = card_favs[c_i]
            m_legit_arr[i] = favs[fav_slot[i] % len(favs)]
        else:
            m_legit_arr[i] = general_m[i]

    amt_legit = np.round(rng.exponential(scale=50.0, size=n_l) + 5.0, 2)
    # Legit shopping outliers (2% of txns)
    outlier_mask = rng.random(size=n_l) < 0.02
    if outlier_mask.any():
        amt_legit[outlier_mask] = np.round(
            amt_legit[outlier_mask]
            + rng.uniform(200.0, 1000.0, size=int(outlier_mask.sum())),
            2,
        )

    chan_legit = rng.choice(
        ["POS", "ECOMMERCE", "ATM", "MOTO"],
        p=[0.60, 0.32, 0.05, 0.03],
        size=n_l,
    )

    # 3. Combine Legit and Fraud Arrays
    all_cards = np.concatenate([c_legit, [r[0] for r in fraud_events]])
    all_merchants = np.concatenate([m_legit_arr, [r[1] for r in fraud_events]])
    all_times = np.concatenate([time_offsets_legit, [r[2] for r in fraud_events]])
    all_amts = np.concatenate([amt_legit, [r[3] for r in fraud_events]])
    all_chans = np.concatenate([chan_legit, [r[4] for r in fraud_events]])
    all_fraud = np.concatenate(
        [np.zeros(n_l, dtype=bool), [r[5] for r in fraud_events]]
    )

    # Sort strictly chronologically
    sort_order = np.argsort(all_times, kind="stable")
    all_cards = all_cards[sort_order]
    all_merchants = all_merchants[sort_order]
    all_times = all_times[sort_order]
    all_amts = all_amts[sort_order]
    all_chans = all_chans[sort_order]
    all_fraud = all_fraud[sort_order]

    # Pre-draw statuses and dispute attributes
    statuses = rng.choice(
        ["APPROVED", "DECLINED", "REFUNDED"],
        p=[0.95, 0.04, 0.01],
        size=num_transactions,
    )
    disp_rand = rng.random(size=num_transactions)
    disp_p = np.where(all_fraud, 0.65, 0.0077)
    has_dispute = disp_rand < disp_p

    fraud_reasons = ["10.4", "10.5", "4837"]
    non_fraud_reasons = ["13.1", "13.3", "12.1", "11.3", "4853", "4834"]
    disp_status_choices = ["OPEN", "UNDER_REVIEW", "WON", "LOST"]
    disp_status_probs = [0.40, 0.30, 0.15, 0.15]

    total_fraud_count = int(all_fraud.sum())
    total_dispute_count = int(has_dispute.sum())

    # Streaming COPY into PostgreSQL in chunks
    num_batches = (num_transactions + batch_size - 1) // batch_size
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min(num_transactions, (b + 1) * batch_size)
        cur_batch_len = end_idx - start_idx

        txn_buf = io.StringIO()
        disp_buf = io.StringIO()

        for i in range(cur_batch_len):
            global_idx = start_idx + i
            c_idx = all_cards[global_idx]
            m_idx = all_merchants[global_idx]

            t_token = card_tokens[c_idx]
            t_m_id = merchant_ids[m_idx]
            t_mcc = merchant_mccs[m_idx]
            t_risk = merchant_risks[m_idx]
            t_is_intl = card_countries[c_idx] != merchant_countries[m_idx]

            txn_dt = start_date + datetime.timedelta(seconds=int(all_times[global_idx]))
            txn_hour = txn_dt.hour
            txn_dt_str = txn_dt.isoformat()

            amt = all_amts[global_idx]
            chan = all_chans[global_idx]
            stat = statuses[global_idx]
            is_frd = bool(all_fraud[global_idx])

            idx_hash = hashlib.md5(f"{global_idx}_{seed}".encode()).hexdigest()[:8]
            txn_id = f"txn_{global_idx:010d}_{idx_hash}"

            txn_row = (
                f"{txn_id}\t{t_token}\t{t_m_id}\t{amt:.2f}\tUSD\t"
                f"{txn_dt_str}\t{txn_hour}\t{chan}\t{t_mcc}\t{t_risk}\t"
                f"{t_is_intl}\t{stat}\t{is_frd}\t{txn_dt_str}\n"
            )
            txn_buf.write(txn_row)

            if has_dispute[global_idx]:
                dsp_hash = hashlib.md5(f"dsp_{global_idx}_{seed}".encode()).hexdigest()[
                    :8
                ]
                disp_id = f"dsp_{global_idx:010d}_{dsp_hash}"
                reason = (
                    rng.choice(fraud_reasons)
                    if is_frd
                    else rng.choice(non_fraud_reasons)
                )
                d_status = rng.choice(disp_status_choices, p=disp_status_probs)
                filed_dt = txn_dt + datetime.timedelta(days=int(rng.integers(1, 15)))
                due_dt = filed_dt + datetime.timedelta(days=30)
                resolved_dt_str = "\\N"
                if d_status in ("WON", "LOST"):
                    res_dt = filed_dt + datetime.timedelta(
                        days=int(rng.integers(5, 30))
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

        logger.info(
            "Generator V2 Parameter Hash: %s",
            GENERATOR_PARAMS_HASH,
        )

        return {
            "elapsed_seconds": elapsed,
            "generator_params_hash": GENERATOR_PARAMS_HASH,
            "reason_codes": rc_count,
            "merchants": len(merchants),
            "cards": len(cards),
            "transactions": n_txn,
            "disputes": n_disp,
            "fraud_rate": fraud_rate,
            "dispute_rate": dispute_rate,
        }
