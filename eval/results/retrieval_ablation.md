# Retrieval Ablation Benchmark Results

Evaluated on 10 questions from `eval/manual_questions.yaml`.

| Configuration              | Hit@1 | Hit@5 | MRR   | p50 Latency | p95 Latency |
|----------------------------|-------|-------|-------|-------------|-------------|
| dense                      | 50.0% | 80.0% | 0.633 | 31.63 ms    | 58.82 ms    |
| hybrid                     | 70.0% | 90.0% | 0.770 | 53.00 ms    | 140.32 ms   |
| hybrid+rerank              | 60.0% | 80.0% | 0.658 | 8049.25 ms  | 8461.23 ms  |
| hybrid+rerank+parent-child | 60.0% | 80.0% | 0.653 | 5314.77 ms  | 7629.58 ms  |
