# Controlled ablation (test split, 270 queries)

Dataset: authored by Claude (AI) on 2026-10-12; not human-validated. Exact search: False. Candidate pool: 20. Top-K for answers: 5.

## Retrieval (answerable queries)

| Arm | Recall@5 | Recall@10 | Hit@5 | MRR | nDCG@10 | Pool hit | Reranker recovery | P50 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vector | 31.8% | 32.0% | 40.1% | 0.369 | 0.300 | 40.1% | 100.0% | 29 | 121 |
| vector_reranker | 32.0% | 32.0% | 40.1% | 0.373 | 0.305 | 40.1% | 100.0% | 77 | 92 |
| vector_graph | 35.1% | 38.7% | 43.3% | 0.386 | 0.329 | 51.8% | 83.6% | 42 | 51 |
| vector_graph_reranker | 42.7% | 42.7% | 51.8% | 0.482 | 0.414 | 51.8% | 100.0% | 89 | 104 |
| hybrid_full | 57.5% | 57.5% | 66.4% | 0.621 | 0.549 | 66.4% | 100.0% | 96 | 109 |

## Answers and answerability (all queries, extractive generator)

| Arm | Correct (grounded) | Correct top-1 (answerable) | Answerable accepted | Unanswerable refused | Balanced | Context precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vector | 28.9% | 25.5% | 100.0% | 0.0% | 50.0% | 9.1% |
| vector_reranker | 28.9% | 27.1% | 100.0% | 0.0% | 50.0% | 9.1% |
| vector_graph | 31.9% | 24.7% | 100.0% | 0.0% | 50.0% | 10.1% |
| vector_graph_reranker | 39.3% | 37.2% | 100.0% | 0.0% | 50.0% | 12.2% |
| hybrid_full | 54.4% | 50.2% | 100.0% | 0.0% | 50.0% | 15.6% |

## Recall@5 by query type

| Type | vector | vector_reranker | vector_graph | vector_graph_reranker | hybrid_full |
| --- | ---: | ---: | ---: | ---: | ---: |
| semantic | 22.7% | 22.7% | 22.7% | 31.8% | 40.9% |
| episodic | 34.8% | 34.8% | 34.8% | 34.8% | 73.9% |
| temporal | 29.5% | 29.5% | 29.5% | 40.9% | 43.2% |
| historical | 21.7% | 21.7% | 52.2% | 73.9% | 87.0% |
| multi_hop | 22.7% | 22.7% | 22.7% | 40.9% | 63.6% |
| entity_relationship | 47.8% | 47.8% | 47.8% | 56.5% | 78.3% |
| conflict | 47.7% | 47.7% | 47.7% | 47.7% | 63.6% |
| preference | 41.3% | 41.3% | 41.3% | 41.3% | 45.7% |
| procedural | 31.8% | 31.8% | 31.8% | 31.8% | 36.4% |
| recency | 31.2% | 32.6% | 31.2% | 32.6% | 34.8% |
| negative_absence | 18.2% | 18.2% | 22.7% | 36.4% | 63.6% |
| unanswerable | — | — | — | — | — |

## Paired comparisons (right minus left)

- graph_contribution (vector → vector_graph): Recall@5 +0.032 [+0.012, +0.057], p=0.0058; grounded correctness +0.030, p=0.0078.
- reranker_on_vector (vector → vector_reranker): Recall@5 +0.001 [+0.000, +0.004], p=1.0000; grounded correctness +0.000, p=1.0000.
- reranker_on_graph (vector_graph → vector_graph_reranker): Recall@5 +0.076 [+0.047, +0.109], p=0.0002; grounded correctness +0.074, p=0.0002.
- keyword_contribution (vector_graph_reranker → hybrid_full): Recall@5 +0.148 [+0.095, +0.198], p=0.0002; grounded correctness +0.152, p=0.0002.

Labels are Claude-authored and not human-validated. Grounded correctness requires the answer to appear in a supporting memory that reached the context.
