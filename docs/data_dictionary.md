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

## 3. Synthetic Data Generator Logic

### 3.1. Stochastic, Non-Deterministic Fraud Model
Fraud generation is **non-deterministic** and incorporates latent stochastic noise:

1. **Feature Score ($S$)**:
   $$\begin{aligned}
   S = &\ \text{night\_signal}(\text{txn\_hour}) \\
   & + \text{risk\_tier\_signal}(\text{risk\_tier}) \\
   & + \text{channel\_signal}(\text{channel}) \\
   & + \text{intl\_signal}(\text{is\_international}) \\
   & + \text{new\_merch\_signal}(\text{is\_new\_merchant}) \\
   & + \text{velocity\_signal}(\text{velocity}_{24h}) \\
   & + \text{amount\_signal}(\text{amount})
   \end{aligned}$$

   - $\text{night\_signal} = +1.3$ if $\text{txn\_hour} \in \{0, 1, 2, 3, 4, 5\}$, else $-0.3$.
   - $\text{risk\_tier\_signal} = +1.6$ for `HIGH`, $+0.4$ for `MEDIUM`, $-0.6$ for `LOW`.
   - $\text{channel\_signal} = +1.1$ for `ECOMMERCE`/`MOTO`, $-0.7$ for `POS`/`ATM`.
   - $\text{intl\_signal} = +1.4$ for international, $-0.2$ for domestic.
   - $\text{new\_merch\_signal} = +0.8$ for first transaction at merchant, $-0.2$ otherwise.
   - $\text{velocity\_signal} = +1.3$ if 24h card velocity $> 4$, $-0.2$ otherwise.
   - $\text{amount\_signal} = +1.2$ if $\text{amount} > \$500$, $-0.3$ otherwise.

2. **Noisy Latent Logit & Probability**:
   $$\text{logit} = -6.8 + 0.55 \cdot S + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1.2^2)$$
   $$P_{\text{feature}} = \frac{1}{1 + e^{-\text{logit}}}$$

3. **Stealth Fraud & False Alarms**:
   - **Stealth / Normal Fraud (~20% of total fraud)**: An independent base probability triggers low-profile fraud on ordinary, low-risk transactions (e.g. small amount, domestic daytime POS).
   - **False Alarms**: Legitimate cardholders routinely travel abroad, make high-value purchases, or shop online at night, generating high feature scores without being flagged as fraud.

### 3.2. Dispute Generation
- **Fraud Transactions**: Approximately 65% result in disputes (`10.4`, `10.5`, `4837`).
- **Legitimate Transactions**: Approximately 0.8% result in commercial disputes (`13.1`, `13.3`, `12.1`, `11.3`, `4853`, `4834`).
- **Target Rates**: Fraud rate calibrated to 0.2%–0.5% (approx 0.30%); dispute rate calibrated to approx 1.0%.

