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

- [ ] Improve conversational retention: useful memories are sometimes classified as temporary or expire too soon.
- [ ] Improve natural-language graph extraction from conversational paragraphs.
- [ ] Improve retrieval ranking and context selection using benchmark failures.
- [ ] Handle indirect, ambiguous, and multi-sentence contradictions.
- [ ] Consolidate related memories with different wording into useful summaries.
- [ ] Add `GOOGLE_API_KEY` and run live Gemini answer-quality evaluation.
- [ ] Obtain independent human reviews of answer usefulness, personalization, and conflict decisions.
- [ ] Calibrate cleanup policy: decide which expired memories should be deleted versus archived.
- [ ] Add sensitive-memory redaction and stronger privacy safeguards.

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
