# CogniMem TODO

## Completed features

- Memory creation, categorization, importance scoring, and lifecycle tiers. The default score is 0.25 × the trained csv5k
  ensemble + 0.75 × the rule-based heuristic; `COGNIMEM_SCORER=heuristic` opts out.
- Task management: deadlines, completion, cancellation, rescheduling, recurrence, and history.
- Conflict detection/resolution for supported facts and preferences.
- Duplicate-memory consolidation with source evidence and revision history.
- JSONL memory history with SQLite vector and temporal-graph indexes.
- Keyword, semantic, graph, and hybrid retrieval.
- RAG context building with source citations.
- LangChain integration for Gemini 2.5 Flash.
- Temporal knowledge graph: entities, aliases, relationships, valid time, recorded time, and bounded traversal.
- CLI commands for processing, searching, graph queries, indexing, cleanup, and RAG answers.
- Scheduled safe cleanup and history compaction.
- External evaluation/reporting framework with baseline comparisons.

## Improvements needed for completed features

- [x] Improve conversational retention: useful declarative facts receive semantic retention and balanced cleanup preserves useful expired records.
- [x] Improve natural-language graph extraction from conversational paragraphs through independent-assertion extraction.
- [x] Improve retrieval ranking and context selection using cross-channel agreement, weighted reciprocal-rank fusion, local cross-encoder reranking, and diverse evidence selection. On the frozen LoCoMo run, reranking raised recall@5 from 48.15% to 55.83%.
- [x] Handle supported indirect transitions and multi-sentence contradictions; ambiguous statements remain non-destructive.
- [x] Consolidate high-overlap paraphrases into provenance-backed related-observation summaries.
- [ ] Add `GOOGLE_API_KEY` and run live Gemini answer-quality evaluation.
- [ ] Obtain independent human reviews of answer usefulness, personalization, and conflict decisions. The blinded two-reviewer protocol and blank template are ready under `reviews/`; real reviewers are still required.
- [x] Calibrate cleanup policy with `balanced` (default) and `expiry` modes and explicit archived/deleted IDs.
- [x] Add pre-persistence secret/identifier redaction and privacy metadata.
- [x] Add the controlled 240-query vector/graph/reranker ablation runner with retrieval, RAG, citation, latency, query-type, statistical, and failure-case reports.
- [ ] Complete the generated blind human-review packet with independent evaluators; the software leaves all ratings blank.

## Features left to build

- [x] REST API.
- [x] Frontend or demo application.
- [x] Authentication and authorization.
- [x] User data export and deletion endpoints.
- [x] Monitoring, metrics, structured logging, and store health checks.
- [ ] Production database backend.
- [x] Approximate-nearest-neighbor vector search for larger workloads.
- [ ] Deployment setup.
- [ ] Background task/reminder delivery service.
- [ ] Broader entity extraction, coreference resolution, and relationship extraction.
- [ ] Human feedback loop for memory decisions.

## Separate ML research track

- [x] Ship the csv5k "Ensemble + thresholds" as 25 % of the default importance score (`python -m training.ensemble.run`);
  the XGBoost-only and neural scorers were removed. Measured on 16 Oct 2026: no change to LongMemEval retention
  (README, "Importance scoring (default)").
- [ ] Collect human-rated importance labels for conversation turns.
- [ ] Build in-domain conversational training data (csv5k sentences are synthetic and at most 13 words long).
- [ ] Calibrate or re-justify the 25 % model weight once the model is trained on in-domain labels.
