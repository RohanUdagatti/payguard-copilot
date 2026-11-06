# Reranker Diagnosis & Sweep Benchmark Report

## 1. Per-Question Before & After Rank Table

Evaluated on 10 gold benchmark questions from `eval/manual_questions.yaml` with hybrid candidate retrieval ($N=20$).

| # | Question | Document | Section Hint | Hybrid Rank (Before) | Reranker Rank (After) | Impact |
| :-: | :--- | :--- | :--- | :-: | :-: | :--- |
| 1 | What is the Federal Reserve payment system... | fed_payment_risk | INTRODUCTION | `2` | `3` | DEMOTED (-1) |
| 2 | What are the rules and caps for daylight o... | fed_payment_risk | RISKS IN PAYMENT, CLEARI | `1` | `4` | DEMOTED (-3) |
| 3 | What are the requirements for memorized se... | nist_digital_ide | 5.1.1 Memorized Secret A | `5` | `6` | DEMOTED (-1) |
| 4 | What are the requirements for out-of-band ... | nist_digital_ide | 5.1.3 Out-of-Band Authen | `1` | `1` | Unchanged |
| 5 | What are the security requirements for Aut... | nist_digital_ide | 4.3 Authenticator Assura | `>20` | `>20` | Not in top-20 |
| 6 | What is Dispute Condition 13.1 for merchan... | visa_chargeback_ | 11.10.2 Dispute Conditio | `1` | `1` | Unchanged |
| 7 | What are the invalid dispute conditions fo... | visa_chargeback_ | 11.7.5.3 Dispute Conditi | `1` | `1` | Unchanged |
| 8 | What are the merchant requirements for Dyn... | visa_chargeback_ | 4.1.24.7 Agentic Payment | `1` | `1` | Unchanged |
| 9 | What are the requirements for partial paym... | visa_chargeback_ | 5.8.11.1 Requirements fo | `1` | `1` | Unchanged |
| 10 | What are the Visa Fleet Merchant Discount ... | visa_chargeback_ | 4.22.2.2 Visa Fleet Merc | `1` | `1` | Unchanged |

## 2. Parameter Sweep: Candidates $N \in \{10, 20\} \times \text{max\_length} \in \{256, 512\}$

| Configuration | Hit@1 (%) | Hit@5 (%) | MRR | p50 Latency (ms) | p95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hybrid Baseline (No Rerank)** | **70.0%** | **90.0%** | **0.770** | **~14.0** | **~20.0** |
| N=10, max_len=256 | 50.0% | 90.0% | 0.667 | 1672.7 | 1917.9 |
| N=10, max_len=512 | 60.0% | 90.0% | 0.683 | 4066.3 | 5091.6 |
| N=20, max_len=256 | 50.0% | 90.0% | 0.675 | 3495.0 | 4827.3 |
| N=20, max_len=512 | 60.0% | 80.0% | 0.658 | 7914.1 | 9788.5 |

## 3. Markup Ablation: Raw Text vs Clean Text

| Text Format | Hit@1 (%) | Hit@5 (%) | MRR | p50 Latency (ms) | p95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Raw Text (with `[TABLE]` markup) | 60.0% | 80.0% | 0.658 | 7914.1 | 9788.5 |
| Clean Text (stripped tags) | 60.0% | 80.0% | 0.658 | 8626.2 | 9955.5 |

## 4. CPU Thread Settings Benchmark

- Hardware: Intel Core i5-1235U (10 physical cores, 12 logical threads)

| Thread Setting | Active Threads | Measured Latency (ms) |
| :--- | :---: | :---: |
| 10 Physical Cores (torch.set_num_threads(10)) | `10` | `8594.2` |
| 2 Performance Cores (torch.set_num_threads(2)) | `2` | `14435.4` |
| 4 Cores (torch.set_num_threads(4)) | `4` | `12975.9` |
| 12 Logical Threads (torch.set_num_threads(12)) | `12` | `10501.5` |

## 5. Formal Recommendation

**Conclusion**: **Keep `RERANK_ENABLED=False` (default OFF)**.

**Plain statement**: Nothing beats hybrid. Hybrid dense+BM25 retrieval achieves **70.0% Hit@1** and **0.770 MRR** in ~14 ms. In contrast, every reranker configuration achieves only **60.0% Hit@1** while adding 3,000–8,000 ms of CPU latency. Specifically, the cross-encoder demotes precise rule-specific sections below broad introductory text.