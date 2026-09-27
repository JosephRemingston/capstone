# Independent evaluation and cleanup

Execution status: **passed**. This means the evaluation ran, not that model quality passed a target.

Externally authored frozen datasets, no tuning in this run. Prior LoCoMo/bAbI and 108 LongMemEval cases were evaluated before; 392 additional LongMemEval examples extend coverage. This is not a fresh held-out test after any future tuning.

| Stage | Status | Seconds |
| --- | --- | ---: |
| locomo_retrieval | passed | 113.6 |
| longmemeval_retention | passed | 12.1 |
| babi_conflicts | passed | 4.7 |
| personalization | passed | 225.4 |

| Retriever | Reference recall@5 | Hit@5 |
| --- | ---: | ---: |
| recency | 0.89% | 1.06% |
| keyword | 37.65% | 41.31% |
| semantic | 43.26% | 47.85% |
| graph | 0.00% | 0.00% |
| hybrid | 48.15% | 53.07% |

LoCoMo reference retrieval includes adversarial cases; answerable-only and per-category metrics are in the detailed report. Retrieval recall is not answer correctness.

Excluded 43 LongMemEval cases whose sessions occur after the question; these cases do not enter retention/cleanup scoring.

Cleanup on isolated external replay stores removed 3,169 eligible memories. JSONL size changed from 24,660,289 to 16,307,078 bytes. Visible memory state was unchanged in 457/457 cases.

| Conversation type | Labeled cases | Retained gold evidence | Retain-all baseline |
| --- | ---: | ---: | ---: |
| knowledge-update | 69 | 10.14% | 100% |
| multi-session | 125 | 12.12% | 100% |
| single-session-assistant | 56 | 62.50% | 100% |
| single-session-preference | 30 | 8.33% | 100% |
| single-session-user | 64 | 21.88% | 100% |
| temporal-reasoning | 92 | 19.60% | 100% |

bAbI uses a deterministic movement-to-location grammar adapter to isolate state updates; it is not natural-language conflict understanding.

| State tracker | Accuracy |
| --- | ---: |
| memory_correct | 100.00% |
| graph_correct | 100.00% |
| first_assertion_correct | 39.60% |
| latest_assertion_correct | 100.00% |

Personalization/update answers use **extractive** generation. Human quality scores are pending, not zero or assumed correct.

| Answer context baseline | Evidence recall | Generation failures |
| --- | ---: | ---: |
| no_memory | 0.00% | 0 |
| keyword | 67.17% | 0 |
| hybrid | 79.46% | 0 |

Blind review packets, frozen answer bindings, and baseline mappings are written to `data/evaluation/personalization/`. Complete independent reviews can be scored with `python -m evaluation.personalization --reviews FILE`. Review assertions are supplied attestations; the software cannot certify reviewer independence.

No cleanup was applied to the real user store, and no background service was installed by this evaluation run. See [cleanup operations](../cleanup.md) and [evaluation instructions](../../evaluation/README.md).
