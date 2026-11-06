# PayGuard Copilot

Enterprise GenAI platform for payment disputes, fraud and compliance.

## Requirements

- Python 3.11
- Docker and Docker Compose
- Make
- `GEMINI_API_KEY` in local `.env` (for LLM answers)

## Setup

```bash
cp .env.example .env
# add GEMINI_API_KEY=... to .env
conda activate payguard   # or any Python 3.11 env
pip install -e ".[core,dev,rag,data,ml]"
```

## Run infrastructure + API

```bash
make up
curl http://localhost:8000/health
```

## Data Layer (Phase 1)

PostgreSQL schema (`merchants`, `cards`, `reason_codes`, `transactions`, `disputes`), Alembic migrations, and high-performance synthetic data generator.

```bash
# Run database migrations
make migrate

# Seed 500,000 synthetic transactions (~25 seconds)
make seed

# Reset database schema
make reset-db
```

Details on schema, stochastic fraud probability model, and data dictionary: [docs/data_dictionary.md](docs/data_dictionary.md).


## RAG baseline (Phase 2)

Put PDFs in `data/raw_docs/` (gitignored), then:

```bash
make ingest
# rebuild collection after changing embedder:
# make ingest INGEST_ARGS=--recreate

curl -s http://localhost:8000/rag/query \
  -H 'content-type: application/json' \
  -d '{"question":"What does the Federal Reserve payment system risk policy cover?"}'
```

Default embedding provider is **local** `BAAI/bge-small-en-v1.5`.
Default LLM provider is **gemini** (`LLM_MODEL` from env).

Design notes: [docs/rag_baseline.md](docs/rag_baseline.md).

## Test and quality

```bash
make test
make lint
make typecheck
```

## Phase 4: Evaluation Harness (Golden Set + RAGAS + CI Gate)

The Phase 4 evaluation pipeline benchmarks RAG configurations (`dense`, `hybrid`, `hybrid+parent-child`) across verified domain questions with LLM-as-a-judge scoring, confidence intervals, and automated CI regression gates.

### 1. Golden Question Dataset & Validation
Stored in `eval/golden_set.jsonl`. Only a human reviewer may set `verified=true`. Questions must follow the `GoldenQuestion` schema:
```bash
python scripts/validate_golden_set.py
```
Validates question types (`lookup`, `paraphrase`, `near_miss`, `multi_section`, `out_of_scope`), ensures out-of-scope items enforce strict abstention, and rejects questions containing Luhn-valid card numbers (PANs).

### 2. Proposing Synthetic Candidate Questions
Generate stratified candidates across compliance PDFs with automatic import from `eval/manual_questions.yaml`:
```bash
python scripts/generate_testset.py --dry-run
python scripts/generate_testset.py --limit 10 --mock
```
Candidates are marked with `verified=false` and `needs_ground_truth=true` in `eval/golden_candidates.jsonl` until reviewed by a domain expert.

### 3. Running RAGAS Evaluation
Evaluation is separated into Stage 1 (`generate`) and Stage 2 (`judge`) with rate limiting and checkpointing:
```bash
# Print dry-run API call estimates:
python scripts/run_ragas.py --dry-run

# Run offline smoke test without API calls:
python scripts/run_ragas.py --mock --yes --run-id smoke_test_run

# Live evaluation (requires JUDGE_MODEL in .env and --yes flag):
python scripts/run_ragas.py --yes
```

### 4. Generating Ablation Reports
Generates `eval/results/ablation.md` with 95% non-parametric bootstrap confidence intervals (1,000 resamples), retrieval metrics, latencies, and judge error accounting:
```bash
python scripts/make_report.py
```

### 5. CI Evaluation Gate & Baseline Management
The CI gate protects against quality regressions by comparing against `eval/baseline.json`:
```bash
# Snapshot approved evaluation run as baseline:
make eval-baseline

# Verify current run meets or exceeds baseline:
make eval-gate
```

> [!WARNING] Self-Preference Limitation
> When the generator and judge share the same model family, RAGAS evaluation scores may exhibit self-preference bias. Evaluator models should ideally be drawn from an independent model family when production budgets permit.

## Fraud Model Service (Phase 5)

Point-in-time XGBoost fraud model service with probability calibration, cost threshold optimization, and SHAP explanations:

```bash
# Train, calibrate, evaluate, and save fraud model:
make train

# Score a transaction via API gateway:
curl -s http://localhost:8000/ml/score \
  -H 'content-type: application/json' \
  -d '{"txn_id": "txn_0000000000_c4addab5"}'

# Optional Kaggle benchmark:
make ml-benchmark-kaggle
```

Details on feature engineering, zero-leakage constraints, cost matrix, and SHAP log-odds: [docs/fraud_model.md](docs/fraud_model.md).

## Layout

```text
src/payguard/{gateway,rag,data,ml,agents,security,eval,platform}
tests/
scripts/
docs/
configs/
eval/
```

## Notes

- Secrets stay in `.env` (never commit).
- `/data/` is gitignored so PDFs are not committed.
- Langfuse is pinned to v2.
