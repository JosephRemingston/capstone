# Retrieval, graph, and external evaluation report

Historical baseline report (September 27). The current expanded evaluation,
chronology exclusions, physical cleanup results, and answer-review workflow are
in [independent_evaluation.md](independent_evaluation.md). Detailed JSON reports
now reflect that newer run; the measurements below preserve the earlier baseline.

Implemented locally: BGE embeddings, persistent SQLite vectors, hybrid ranking, bounded source context, LangChain Gemini 2.5 Flash integration, checked citations, and a typed bitemporal graph. Configuration and usage: [retrieval/graph guide](../retrieval_graph.md).

## LoCoMo: retrieval baselines

All 10 external conversations were indexed as raw text turns with speaker/date metadata, without QA labels or generated summaries. No fitting or weight tuning was performed. Of 1,986 questions, 1,973 have resolvable text evidence; 4 lack evidence and 9 reference unavailable text evidence. This includes 446 category-5 adversarial questions: their evidence can be retrieved, but they are not answerable. The answerable-only column excludes category 5 (1,527 questions).

| Retriever | Overall reference recall@5 | Answerable recall@5 | Hit@5 | MRR@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| recency | 0.89% | 0.76% | 1.06% | 0.0041 | 0.0050 |
| keyword | 37.65% | 37.94% | 41.31% | 0.3023 | 0.3104 |
| semantic | 43.26% | 44.24% | 47.85% | 0.3280 | 0.3413 |
| graph | 0.00% | 0.00% | 0.00% | 0.0000 | 0.0000 |
| hybrid | 48.15% | 48.59% | 53.07% | 0.3730 | 0.3854 |

Hybrid overall recall improves by 10.50 percentage points over keyword search. Its conversation-bootstrap 95% interval is 46.18%–50.06% (1,000 resamples, seed 42; only 10 conversation groups, so uncertainty is substantial). These are evidence-retrieval metrics, not generated-answer accuracy.

The graph extracted no edges from this corpus: the automatic extractor requires supported complete assertions and rejects most long turns. Graph-only recall is therefore zero, and hybrid gains here come from lexical+dense fusion, not graph reasoning. Functional graph traversal on structured/atomic assertions is tested separately. This limitation is preserved in the report.

The fixed preference-word query subset has 121 questions. Evidence recall is 36.70% for hybrid and 37.76% for semantic retrieval. This subset does not measure personalized recommendation quality.

The corpus contains 5,882 text turns, all retained to isolate retrieval. Mean hybrid retrieved snippet length is 1009 characters per query (excluding prompt/ID overhead). Index sizes and source bytes are reported per conversation. Raw-turn retention ratio is 1.0; this benchmark does not demonstrate memory compression.

## LongMemEval: end-to-end evidence retention

All 78 knowledge-update and 30 single-session-preference oracle examples were replayed through the existing core/store with ingestion clocks set to their original timestamps. Gold `has_answer` annotations are used only for scoring. Six knowledge-update examples lack labeled answer turns. Oracle evidence sessions exclude full-corpus retrieval distractors.

| Type | Examples with evidence | Retain-everything baseline | Current memory evidence retention | Mean visible turn fraction |
| --- | ---: | ---: | ---: | ---: |
| knowledge-update | 72 | 100.00% | 11.81% | 14.36% |
| single-session-preference | 30 | 100.00% | 8.33% | 23.93% |

These poor retention results show that the existing rules/lifecycle often discard useful evidence in long conversations. Neither subset triggered supported conflict/consolidation decisions. Low visible-memory size is therefore not evidence of good compression. Original history remains in JSONL, and report rows include input/visible text bytes and physical history bytes. We did not tune the existing lifecycle rules on these examples.

## bAbI: independent synthetic state updates

Used all 200 English QA1 test stories and 1,000 questions from the published dataset. A deterministic adapter maps movement events to supported named-person location assertions; original gold answers are unchanged. This isolates conflict resolution from language extraction. There are 1,241 changed-location events.

| Method | Current-location accuracy |
| --- | ---: |
| Memory conflict resolver | 100.00% |
| Temporal graph | 100.00% |
| Keep first assertion | 39.60% |
| Latest assertion baseline | 100.00% |

The latest-assertion baseline also solves this simple task. This is external synthetic evidence for state tracking, not real conversational understanding, arbitrary contradiction handling, or independent human judgment of the conflict policy.

## Validation and limits

- Standard-library suite: 102 tests, 98 passed and 4 optional ML tests skipped. The separate ML track was not changed.
- All 19 new graph/RAG/evaluation tests also pass in the virtual environment with optional dependencies installed.
- Existing 48 developer-authored category/tier/importance checks still pass; these are regression cases, not independent evaluation.
- Tests cover valid/recorded time, historical restoration, changed source text, task status history, aliases, graph evidence paths, user isolation, stale vector removal, metadata validation, context budgets, CLI preview/extractive answers, citation rejection, environment configuration, and mocked hosted responses.
- Real local BGE embeddings were used in the LoCoMo run. LangChain accepts the Gemini configuration and output schema. No live Gemini request was made; the user will supply the key later.
- Exact citation quotes are checked against retrieved chunks. This does not verify logical entailment or guarantee prompt-injection resistance.
- Independent human ratings of conflict policy and personalization, live generated-answer quality, broader paragraph extraction, and deployed memory-system comparisons remain unmeasured.

## Reproduction and artifacts

```bash
uv pip install --python .venv/bin/python -r requirements-evaluation.txt
.venv/bin/python -m tests.evaluation.benchmark --download
.venv/bin/python -m tests.evaluation.longmemeval --download
.venv/bin/python -m tests.evaluation.conflicts --download
python3 -m unittest discover -s tests -q
```

Protocols, pinned source revisions, SHA-256 checksums, exclusions, per-query predictions, category breakdowns, and storage measurements are in [LoCoMo JSON](locomo_retrieval.json), [LongMemEval JSON](longmemeval_retention.json), and [bAbI JSON](babi_conflicts.json). Large downloaded data/models/indexes are ignored by Git.

Primary dataset sources: [LoCoMo](https://github.com/snap-research/locomo), [LongMemEval](https://github.com/xiaowu0162/LongMemEval), [bAbI QA](https://huggingface.co/datasets/facebook/babi_qa) (CC BY 3.0).
