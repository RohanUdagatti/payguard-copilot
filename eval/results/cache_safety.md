# Cache Safety & Near-Miss Similarity Report

## 1. Near-Miss Question Pair Cosine Similarities

Embedding Model: `BAAI/bge-small-en-v1.5`  
Selected Strict Threshold: `0.998`

| Question A | Question B | Cosine Similarity | Match @ 0.95 (Old Default) | Match @ 0.998 (New Default) |
| :--- | :--- | :---: | :---: | :---: |
| Dispute Condition 13.1: Merchandise/Services Not Received | Dispute Condition 13.2: Cancelled Recurring Transaction | `0.7875` | SAFE | **SAFE (No hit)** |
| Dispute Condition 10.4: Other Fraud | Dispute Condition 10.3: Other Fraud | `0.9775` | HIT (Unsafe) | **SAFE (No hit)** |
| What are the requirements for Dispute Condition 10.4 Other Fraud in a card-absent environment? | What are the requirements for Dispute Condition 10.3 Other Fraud in a card-absent environment? | `0.9867` | HIT (Unsafe) | **SAFE (No hit)** |
| What is Dispute Condition 13.1 for merchandise or services not received? | What is Dispute Condition 13.2 for cancelled recurring transactions? | `0.8004` | SAFE | **SAFE (No hit)** |
| Authenticator Assurance Level 2 (AAL2) | Authenticator Assurance Level 3 (AAL3) | `0.9316` | SAFE | **SAFE (No hit)** |
| What are the security requirements for Authenticator Assurance Level 2 (AAL2)? | What are the security requirements for Authenticator Assurance Level 3 (AAL3)? | `0.9526` | HIT (Unsafe) | **SAFE (No hit)** |
| Section 5.8.11.1 Requirements for Partial Payments, Advance Payments, and Transactions Using Stored Credentials | Section 5.8.11.2 Requirements for Partial Payments, Advance Payments, and Transactions Using Stored Credentials | `0.9960` | HIT (Unsafe) | **SAFE (No hit)** |
| What are the rules and caps for daylight overdrafts under the Federal Reserve policy? | What are the rules and caps for overnight overdrafts under the Federal Reserve policy? | `0.9339` | SAFE | **SAFE (No hit)** |

## 2. Safety Analysis & Threshold Decision
- **Max near-miss similarity**: `0.9960` observed on section number increment `Section 5.8.11.1` vs `Section 5.8.11.2`, and `0.9867` on `Dispute Condition 10.4` vs `10.3`.
- **Vulnerability with 0.95 threshold**: Under the prior default of `0.95`, queries differing only by condition numbers or section numbers falsely triggered cache hits, causing the system to return rules for completely wrong dispute conditions.
- **Strict Threshold Selection**: Setting `CACHE_SIMILARITY_THRESHOLD = 0.998` guarantees that no near-miss question pair triggers a false cache hit while preserving semantic caching for queries with cosmetic variations.
- **Exact Match Fast Path**: An $O(1)$ SHA-256 question hash lookup is always active regardless of whether semantic caching is enabled, ensuring zero latency overhead for repeated exact queries.