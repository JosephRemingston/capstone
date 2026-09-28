# CogniMem TODO

## Completed features

- Memory creation, categorization, heuristic importance scoring, and lifecycle tiers.
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

## Features left to build

- [ ] REST API.
- [ ] Frontend or demo application.
- [ ] Authentication and authorization.
- [ ] User data export and deletion endpoints.
- [ ] Monitoring, metrics, structured logging, and store health checks.
- [ ] Production database backend.
- [ ] Approximate-nearest-neighbor vector search for larger workloads.
- [ ] Deployment setup.
- [ ] Background task/reminder delivery service.
- [ ] Broader entity extraction, coreference resolution, and relationship extraction.
- [ ] Human feedback loop for memory decisions.

## Separate ML research track

- [ ] Improve and validate the optional ML importance model with representative conversational labels.
