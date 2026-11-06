# PayGuard Copilot Data Dictionary & Synthetic Generator Specification

> [!IMPORTANT]
> **Synthetic Benchmark Notice**: This dataset is synthetically generated for testing, database benchmarking, and integration pipelines (RAG, text-to-SQL, and fraud model prototyping). Metrics, precision, recall, and ROC-AUC measured on this synthetic data do NOT represent real-world performance on production payment networks.

---

## 1. Relational Schema Overview

The database consists of 5 core tables hosted in PostgreSQL 16:
- `merchants`: Merchant accounts, category codes (MCC), and risk tiers.
- `cards`: Tokenized card accounts (strictly tokenized, no raw PAN stored).
- `reason_codes`: Standard card network chargeback reason codes (Visa and Mastercard), labelled synthetic.
- `transactions`: Core payment transaction log with normalized and denormalized risk features.
- `disputes`: Chargebacks and disputes linked to transactions and reason codes.

---

## 2. Table Specifications

### 2.1. `merchants`
| Column | Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `merchant_id` | VARCHAR(64) | NO (PK) | Unique merchant identifier (e.g. `m_5411_000123`). |
| `name` | VARCHAR(255) | NO | Business display / legal name. |
| `mcc` | VARCHAR(4) | NO | 4-digit Merchant Category Code (ISO 18245). |
| `risk_tier` | VARCHAR(16) | NO | Merchant baseline risk tier: `LOW`, `MEDIUM`, `HIGH`. |
| `country` | VARCHAR(2) | NO | ISO 3166-1 alpha-2 country code. |
| `created_at` | TIMESTAMPTZ | NO | Merchant onboarding timestamp. |

### 2.2. `cards`
| Column | Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `card_token` | VARCHAR(64) | NO (PK) | Opaque card token (e.g. `tok_visa_...`). Strictly zero raw PAN. |
| `user_id` | VARCHAR(64) | NO | Customer account ID. |
| `card_brand` | VARCHAR(16) | NO | Payment card network: `VISA`, `MASTERCARD`, `AMEX`, `DISCOVER`. |
| `card_type` | VARCHAR(16) | NO | Card product type: `CREDIT`, `DEBIT`, `PREPAID`. |
| `issuer_country` | VARCHAR(2) | NO | Issuing bank country code (ISO 3166-1 alpha-2). |
| `created_at` | TIMESTAMPTZ | NO | Card account registration timestamp. |

### 2.3. `reason_codes`
| Column | Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `reason_code` | VARCHAR(16) | NO (PK) | Network chargeback reason code (e.g. `10.4`, `13.1`, `4837`). |
| `network` | VARCHAR(16) | NO | Card network: `VISA`, `MASTERCARD`. |
| `description` | VARCHAR(255) | NO | Official chargeback reason description. |
| `category` | VARCHAR(64) | NO | Classification: `FRAUD`, `CUSTOMER_DISPUTE`, `PROCESSING_ERROR`, `AUTHORIZATION`. |

*Note: Reason codes in this table are a representative set labelled synthetic.*

### 2.4. `transactions`
| Column | Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `transaction_id` | VARCHAR(64) | NO (PK) | Unique transaction ID (e.g. `txn_...`). |
| `card_token` | VARCHAR(64) | NO (FK) | Reference to `cards.card_token`. Usable for velocity calculations. |
| `merchant_id` | VARCHAR(64) | NO (FK) | Reference to `merchants.merchant_id`. |
| `amount` | NUMERIC(12, 2)| NO | Transaction settlement amount. |
| `currency` | VARCHAR(3) | NO | ISO 4217 currency code (default `'USD'`). |
| `timestamp` | TIMESTAMPTZ | NO | Authorization timestamp. |
| `txn_hour` | SMALLINT | NO | Hour of day (0–23) extracted from `timestamp`. |
| `channel` | VARCHAR(16) | NO | Transaction channel: `POS`, `ECOMMERCE`, `ATM`, `MOTO`. |
| `mcc` | VARCHAR(4) | NO | Merchant Category Code (denormalized for query efficiency). |
| `risk_tier` | VARCHAR(16) | NO | Merchant risk tier: `LOW`, `MEDIUM`, `HIGH`. |
| `is_international`| BOOLEAN | NO | `TRUE` if card `issuer_country != merchant.country`. |
| `status` | VARCHAR(16) | NO | Transaction status: `APPROVED`, `DECLINED`, `REFUNDED`. |
| `is_fraud` | BOOLEAN | NO | Ground truth fraud label for downstream Phase 5 model. |
| `created_at` | TIMESTAMPTZ | NO | Record ingestion timestamp. |

**Indexes**:
- `idx_transactions_card_timestamp`: `(card_token, timestamp)`
- `idx_transactions_merchant_timestamp`: `(merchant_id, timestamp)`
- `idx_transactions_timestamp_hour`: `(timestamp, txn_hour)`
- `idx_transactions_is_fraud`: `(is_fraud)`

### 2.5. `disputes`
| Column | Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `dispute_id` | VARCHAR(64) | NO (PK) | Unique dispute ID (e.g. `dsp_...`). |
| `transaction_id` | VARCHAR(64) | NO (FK) | Reference to `transactions.transaction_id` (1:1 relation). |
| `amount` | NUMERIC(12, 2)| NO | Disputed amount. |
| `currency` | VARCHAR(3) | NO | Currency code (`'USD'`). |
| `reason_code` | VARCHAR(16) | NO (FK) | Reference to `reason_codes.reason_code`. |
| `status` | VARCHAR(24) | NO | Dispute lifecycle status: `OPEN`, `UNDER_REVIEW`, `WON`, `LOST`. |
| `evidence_due_date`| TIMESTAMPTZ| NO | Network evidence submission cutoff date. |
| `filed_at` | TIMESTAMPTZ | NO | Date dispute was initiated. |
| `resolved_at` | TIMESTAMPTZ | YES | Settlement/resolution date (`NULL` if pending). |
| `created_at` | TIMESTAMPTZ | NO | Record creation timestamp. |

**Indexes**:
- `idx_disputes_transaction_id`: `(transaction_id)`
- `idx_disputes_status_filed_at`: `(status, filed_at)`
- `idx_disputes_reason_code`: `(reason_code)`

---

## 3. Synthetic Data Generator Logic (Generator V2)

The synthetic payment data generator was upgraded in Phase 5.2 to **Generator V2**. Generator V2 replaces per-row stochastic coin flips with structural event typologies grounded in payment fraud domain patterns.

### 3.1. Integrity & Parameter Immutability
Generator V2 parameters are immutable and defined in `GENERATOR_V2_CONFIG`. The SHA-256 fingerprint of the parameter specification is:
```
GENERATOR_PARAMS_HASH = 4cb99e8f5b9a7cafcd4a0e72e6f2ebdafeed7b98533634088a2c488623266332
```
This configuration is frozen and deterministic for `seed=42`. Parameters are not retroactively tuned to fit model metrics.

### 3.2. Legitimate Activity Modeling
1. **Per-Card Daily Rate**:
   - Drawn from a lognormal distribution: $\lambda_c \sim \text{Lognormal}(\mu=-1.35, \sigma=0.65)$, clipped to $[0.01, 7.5]$.
   - Realized daily activity: median $\approx 0.17$ to $0.30$ txns/day, 99th percentile $< 3.0$ txns/day.
   - **Hard Cap**: Strictly $\le 8$ transactions per card on any single calendar day.
2. **Card Favorite Merchants**:
   - Each card is assigned $K_c \in [5, 15]$ favorite merchants.
   - 75% of legitimate transactions choose from the card's favorites, while 25% sample from the general merchant pool.
   - *Result*: Establishes realistic recurring merchant patterns; first visits to outside merchants give meaningful signal to `is_new_merchant_for_card`.
3. **Diurnal and Persona Variations**:
   - Standard cards: activity peaks between 10:00 and 19:59 (only ~3% of transactions occurring during night hours 00:00–05:59).
   - Habitual Night Owls (~4% of cards): inverted diurnal distribution with 72% of transactions occurring between 22:00 and 05:59.
   - Natural Travellers (~3% of cards): assigned a 3–7 day foreign window where merchants are located in a destination country ($M_{\text{country}} \neq C_{\text{issuer}}$).
   - Legit Shopping Spikes: ~2% of transactions feature high-value shopping additions (\$200–\$1,000).

### 3.3. Event-Driven Fraud Typologies
Fraud transactions (calibrated to 0.2%–0.5% of total transactions, target ~0.35%) are generated as distinct event patterns:
1. **Compromised-Card Bursts** (~35% of fraud):
   - 2 to 8 transactions within a 1 to 6 hour window on the same card.
   - Escalating amounts: successive transactions strictly increase in value (\$35 $\to$ \$1,200).
   - Primarily new merchants (outside the card's favorite set), high/medium risk tiers, often ECOMMERCE or MOTO.
2. **Card Testing** (~20% of fraud):
   - 3 to 10 rapid micro-transactions (under \$5.00) within 1 to 15 minutes at online merchants (ECOMMERCE, HIGH risk tier).
   - In 50% of testing events, followed 10–30 minutes later by a single large unauthorized transaction (\$500–\$2,200).
3. **Geographic Jump / Velocity Anomaly** (~15% of fraud):
   - An international transaction occurring 30 to 80 minutes after a legitimate domestic transaction on the same card (impossible physical travel speed).
4. **Opportunistic High-Value Fraud** (~10% of fraud):
   - Single isolated transaction at an unvisited merchant during night hours (00:00–05:59) with high amount (\$600–\$3,500).
5. **Stealth Fraud** (~20% of fraud):
   - Transactions that mimic normal user behavior across all observable features: domestic, daytime, ordinary amounts (\$15–\$85), POS/ECOMMERCE, LOW/MEDIUM risk tiers.

### 3.4. Dispute Generation
- **Fraud Transactions**: Approximately 65% result in disputes (`10.4`, `10.5`, `4837`).
- **Legitimate Transactions**: Approximately 0.77% result in commercial disputes (`13.1`, `13.3`, `12.1`, `11.3`, `4853`, `4834`).
- **Target Rates**: Fraud rate calibrated to 0.2%–0.5% (~0.35%); dispute rate calibrated to approx 1.00%.
- **Lifecycle & Resolution**:
  - `filed_at`: 1–15 days after transaction timestamp.
  - `status`: `OPEN` (40%), `UNDER_REVIEW` (30%), `WON` (15%), `LOST` (15%).
  - `resolved_at`: 5–30 days after filing for resolved statuses, `NULL` for open disputes.

