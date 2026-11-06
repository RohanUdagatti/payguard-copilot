# ⚠️ MOCK RUN - NOT REAL RESULTS ⚠️

> [!CAUTION] MOCK RUN
> This evaluation report was generated from a mock run. Scores and latencies are synthetic.

# Phase 4 RAGAS Ablation Report: `mock_phase4_run`

- **Timestamp**: `2026-11-05T11:49:07.758160+00:00`
- **Git Commit**: `5a8c7b1261c3c3c35c46cb2a960eedef6a7c6161`
- **Golden Set SHA-256**: `c040d7b06e52a02ae139340f9c04fb62dbd00558a2b9cbee65ba577887e1ef82`
- **Generator Model**: `gemini-flash-lite-latest` (temp=0.0)
- **Judge Model**: `gemini` / `mock-judge`
- **RAGAS Version**: `0.4.3` | **Bootstrap Resamples**: `1000` (seed=42)

> [!WARNING] Self-Preference Limitation
> When the generator and judge share the same model family, RAGAS evaluation scores may exhibit self-preference bias. Evaluator models should ideally be drawn from an independent model family when production budgets permit.

## Verified Golden Questions Distribution
- **Total Human-Verified Questions**: 0

| Question Type | Verified Count |
| :--- | :---: |
| *(none verified)* | 0 |

| Source Document | Verified Count |
| :--- | :---: |
| *(none verified)* | 0 |

## Evaluation Results & Ablation Matrix

| Configuration | n (scored/tot) | Faithfulness (95% CI) | Relevancy (95% CI) | Context Precision (95% CI) | Context Recall (95% CI) | Abstention Accuracy | Hit@1 | Hit@5 | p50 Total Latency | p95 Total Latency | NaNs/Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| dense | 7/7 | 0.900 [0.900, 0.900] | 0.880 [0.880, 0.880] | 0.850 [0.850, 0.850] | 0.920 [0.920, 0.920] | 1.000 [1.000, 1.000] | 100.0% | 100.0% | 45.0 ms | 45.0 ms | 0 |
| hybrid | 7/7 | 0.900 [0.900, 0.900] | 0.880 [0.880, 0.880] | 0.850 [0.850, 0.850] | 0.920 [0.920, 0.920] | 1.000 [1.000, 1.000] | 100.0% | 100.0% | 45.0 ms | 45.0 ms | 0 |
| hybrid+parent-child | 7/7 | 0.900 [0.900, 0.900] | 0.880 [0.880, 0.880] | 0.850 [0.850, 0.850] | 0.920 [0.920, 0.920] | 1.000 [1.000, 1.000] | 100.0% | 100.0% | 45.0 ms | 45.0 ms | 0 |
| hybrid+rerank *(Phase 3.1 negative result)* | 10/10 | — | — | — | — | — | 60.0% | 80.0% | 8049.3 ms | 8461.2 ms | 0 |

## Metric Definitions & Methodology
1. **Confidence Intervals**: 95% non-parametric bootstrap intervals (2.5th and 97.5th percentiles) derived from 1,000 resamples.
2. **Abstention Accuracy**: Percentage of out-of-scope compliance questions where the copilot successfully refrained from hallucinating and returned the exact refusal message.
3. **Hybrid + Reranker Row**: Included as a Phase 3.1 benchmark negative result. Cross-encoder reranking increased p50 retrieval latency from 53 ms to 8,049 ms while dropping Hit@1 from 70% to 60%, and was excluded from LLM generation stages.

## NaN & Error Log Summary
✓ **0 NaNs and 0 Judge Errors detected.** All verified items evaluated successfully.
