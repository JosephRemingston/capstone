# Controlled ablation (dev split, 90 queries)

Dataset: authored by Claude (AI) on 2026-10-12; not human-validated. Exact search: True. Candidate pool per arm: {'vector': 20, 'vector_reranker': 100, 'vector_graph': 20, 'vector_graph_reranker': 100, 'hybrid_full': 100}. Top-K for answers: 5.

## Retrieval (answerable queries)

| Arm | Recall@5 | Recall@10 | Hit@5 | MRR | nDCG@10 | Pool hit | Reranker recovery | P50 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vector | 92.8% | 95.2% | 96.4% | 0.907 | 0.858 | 100.0% | 96.4% | 19 | 27 |
| vector_reranker | 95.2% | 98.8% | 98.8% | 0.905 | 0.878 | 100.0% | 98.8% | 193 | 230 |
| vector_graph | 97.6% | 98.8% | 98.8% | 0.881 | 0.871 | 100.0% | 98.8% | 34 | 41 |
| vector_graph_reranker | 99.4% | 100.0% | 100.0% | 0.950 | 0.935 | 100.0% | 100.0% | 208 | 242 |
| hybrid_full | 99.4% | 100.0% | 100.0% | 0.950 | 0.935 | 100.0% | 100.0% | 221 | 257 |

## Answers and answerability (all queries, extractive generator)

| Arm | Correct (grounded) | Correct top-1 (answerable) | First statement correct (answerable) | Answerable accepted | Unanswerable refused | Balanced | Context precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vector | 85.6% | 69.9% | 55.4% | 100.0% | 0.0% | 50.0% | 26.0% |
| vector_reranker | 93.3% | 73.5% | 73.5% | 100.0% | 42.9% | 71.4% | 26.7% |
| vector_graph | 90.0% | 63.9% | 60.2% | 100.0% | 0.0% | 50.0% | 27.5% |
| vector_graph_reranker | 95.6% | 81.9% | 81.9% | 100.0% | 42.9% | 71.4% | 28.2% |
| hybrid_full | 95.6% | 81.9% | 81.9% | 100.0% | 42.9% | 71.4% | 28.2% |

## Recall@5 by query type

| Type | vector | vector_reranker | vector_graph | vector_graph_reranker | hybrid_full |
| --- | ---: | ---: | ---: | ---: | ---: |
| semantic | 87.5% | 100.0% | 100.0% | 100.0% | 100.0% |
| episodic | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| temporal | 87.5% | 87.5% | 87.5% | 93.8% | 93.8% |
| historical | 100.0% | 85.7% | 100.0% | 100.0% | 100.0% |
| multi_hop | 75.0% | 75.0% | 100.0% | 100.0% | 100.0% |
| entity_relationship | 85.7% | 100.0% | 100.0% | 100.0% | 100.0% |
| conflict | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| preference | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| procedural | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| recency | 85.7% | 100.0% | 85.7% | 100.0% | 100.0% |
| negative_absence | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| unanswerable | — | — | — | — | — |

## Paired comparisons (right minus left)

- graph_contribution (vector → vector_graph): Recall@5 +0.048 [+0.012, +0.090], p=0.0370; grounded correctness +0.044, p=0.1244.
- reranker_on_vector (vector → vector_reranker): Recall@5 +0.024 [-0.030, +0.078], p=0.5297; grounded correctness +0.078, p=0.0670.
- reranker_on_graph (vector_graph → vector_graph_reranker): Recall@5 +0.018 [-0.012, +0.054], p=0.4983; grounded correctness +0.056, p=0.0680.
- keyword_contribution (vector_graph_reranker → hybrid_full): Recall@5 +0.000 [+0.000, +0.000], p=1.0000; grounded correctness +0.000, p=1.0000.

Labels are Claude-authored and not human-validated. Grounded correctness requires the answer to appear in a supporting memory that reached the context.
