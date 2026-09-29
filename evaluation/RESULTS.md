# CogniMem controlled graph and reranker ablation

## Research question

How much does adding temporal/relational graph retrieval improve vector-based
conversational memory retrieval and RAG performance?

The frozen development benchmark contains 240 queries and 240 memories, with 20
queries in each of 12 categories. All systems use the same local embedding model,
dataset, Top-K values, 20-candidate limit, context budget, answer policy, and
metrics. The optional XGBoost memory-importance model is disabled. These labels
are developer-authored controlled cases and are not independent human gold.

## Retrieval results

| System | Recall@5 | Recall@10 | Recall@20 | Hit@5 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector | 89.55% | 91.14% | 91.36% | 91.36% | 0.842 | 0.830 |
| Vector + reranker | 91.36% | 91.36% | 91.36% | 91.36% | 0.914 | 0.893 |
| Vector + graph | 91.36% | 91.36% | 91.36% | 91.36% | 0.884 | 0.891 |
| Vector + graph + reranker | 91.36% | 91.36% | 91.36% | 91.36% | 0.914 | 0.893 |

Adding the graph improves Recall@5 by 1.82 percentage points over vector-only
(95% paired-bootstrap interval 0.68–3.18 points; paired randomization p=0.0072).
The graph gain is concentrated in multi-hop queries, where Recall@5 rises from
80% to 100%. Graph-enabled systems recover a complete supporting evidence path
for 100% of multi-hop cases; vector-only systems recover no graph path by design.

Adding the reranker to vector retrieval also improves Recall@5 by 1.82 points and
raises MRR from 0.842 to 0.914. Adding it after graph fusion leaves Recall@5
unchanged and raises MRR from 0.884 to 0.914. This supports an ordering benefit,
not an additional coverage gain after graph fusion, on this dataset.

## Deterministic extractive RAG results

| System | Correctness | Faithfulness | Context precision | Context recall | Citation completeness |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector | 91.67% | 100% | 21.08% | 82.08% | 80.42% |
| Vector + reranker | 91.67% | 100% | 21.75% | 83.75% | 83.75% |
| Vector + graph | 91.67% | 100% | 21.75% | 83.75% | 83.75% |
| Full | 91.67% | 100% | 21.75% | 83.75% | 83.75% |

Faithfulness is mechanical quote grounding for the extractive generator. It is
not an independent judgment of answer quality. All four systems fail to abstain
on the unanswerable cases because retrieval still supplies unrelated context;
this is recorded as a failure rather than hidden.

## Latency

| System | End-to-end P50 | P95 | P99 |
| --- | ---: | ---: | ---: |
| Vector | 16.78 ms | 21.49 ms | 23.03 ms |
| Vector + reranker | 38.40 ms | 89.45 ms | 94.04 ms |
| Vector + graph | 25.29 ms | 31.45 ms | 34.13 ms |
| Full | 47.04 ms | 98.39 ms | 102.23 ms |

The graph adds about 8.6 ms at P50, while the reranker adds about 22–23 ms. The
full system costs roughly 31 ms more than vector-only at P50.

## Limits and pending work

The separate rule-based classification check reports 100% accuracy and macro F1
across its 48 developer cases, with per-class precision/recall/F1 and a confusion
matrix. The lifecycle audit passes all 14 checks covering tier assignment, task
expiry, completion, cancellation, rescheduling, recurrence, archive views, and
history preservation. Both are regression evidence rather than independent
accuracy estimates.

- Independent reviewers still need to complete the generated blind 60-question,
  four-system packet.
- Live Gemini comparison remains optional until `GOOGLE_API_KEY` is configured.
- `BAAI/bge-reranker-v2-m3` is unavailable in the pinned FastEmbed release. Both
  reranked arms use the integrated Apache-2.0
  `Xenova/ms-marco-MiniLM-L-6-v2`, preserving a controlled comparison.
- Procedural queries score only 5% Recall@5 because the templated procedure
  memories are highly similar. This failure remains visible in the report.
- Results establish behavior on a controlled development set. A capstone claim
  about general conversational performance needs a fresh independent dataset and
  completed human ratings.
