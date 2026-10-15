# CogniMem System Design Document

Status: Draft

Owner: Joseph Remingston L

Last updated: October 16, 2026

Related docs:

- `README.md`
- `data/Capstone Project Proposal.pdf`
- `data/Capstone_Base_Paper_Analysis.docx`
- `data/RAG-Driven_Memory_Architectures_in_Conversational_LLMsA_Literature_Review_With_Insights_Into_Emerging_Agriculture_Data_Sharing.pdf`

Scope: This document describes the current implemented Memory Core, CLI, local JSONL store, and the planned complete CogniMem capstone architecture. It clearly separates implemented functionality from planned future components.

## 1. Abstract

CogniMem is a Cognitive Hybrid Memory Architecture for long-term personalized LLM agents. The project is based on the observation that standard vector-only RAG memory systems struggle with long-term personalization, temporal reasoning, contradictory information, lifecycle management, and structured memory types.

The repository currently implements the first foundation layer: a Memory Core with local CLI access and JSONL persistence. This layer accepts raw interaction content, classifies the content into a memory category, extracts scoring features, calculates an importance score (by default a hybrid of a trained model and the rule-based heuristic), assigns a lifecycle tier, and returns a serializable memory record that can be saved locally.

The core now includes local semantic/hybrid retrieval, cited Gemini RAG, SQLite temporal graph indexes, external evaluation, an authenticated FastAPI/demo surface, privacy endpoints, operational probes/metrics, and deterministic LSH ANN retrieval (opt-in, experimental; exact search is the default). Production database migration and deployment remain separate planned work. ML research remains separate. See [the retrieval and graph sections of DOCUMENTATION.md](DOCUMENTATION.md).

## 2. Repository Analysis

The current repository contains:

| Area | Status | Notes |
| --- | --- | --- |
| `main/` | Implemented | Python package containing the Memory Core, CLI entry point, and local JSONL store modules. |
| `tests/` | Implemented | Unit tests for classification, scoring, lifecycle assignment, serialization, local storage, CLI behavior, and graph isolation. |
| `README.md` | Implemented | Current implementation explanation and next-step roadmap. |
| `data/` | Source documentation | Capstone proposal, base paper analysis, and base literature review paper. |
| Local storage | Implemented | JSONL-backed local memory store at `memory_store/memories.jsonl` by default. |
| CLI | Implemented | `python3 -m main` supports process, list, search, get, and stats commands. |
| Production database layer | Not implemented | SQLite indexes exist; authoritative concurrent server storage is still planned. |
| RAG/vector layer | Implemented locally | BGE embeddings, SQLite vectors, hybrid retrieval, bounded context, and LangChain Gemini with checked citations. |
| Graph layer | Implemented locally | Typed SQLite graph with provenance, valid/recorded time, aliases, task edges, and traversal. |
| ML training | Implemented (default scorer) | csv5k importance ensemble (XGBoost plus a deep head on frozen BGE embeddings) whose model files are rebuilt bit for bit by `training/ensemble/`, served from `artifacts/importance_ensemble/` by `main/ensemble.py`; trained on synthetic data, test metrics in `evaluation/reports/retention_ensemble.md`. |
| REST API | Implemented | FastAPI authentication, memory, search, health, metrics, and privacy endpoints. |

## 3. Implemented, Partial, Planned, and Out of Scope

Implemented:

- `MemoryInput` model for raw incoming interactions.
- `MemoryRecord` model for normalized memory objects.
- `MemoryCategory` values: `semantic`, `episodic`, `procedural`, `preference`, `task`, and `temporary`.
- `MemoryTier` values: `working`, `short_term`, `long_term`, and `archive`.
- Rule-based memory classification.
- Feature extraction for scoring.
- Hybrid importance scoring (default): `0.25 × model_score + 0.75 × heuristic_score` from the trained csv5k ensemble and the rule-based heuristic; heuristic-only scoring as an opt-out.
- Lifecycle tier assignment.
- Serialization and deserialization.
- Local JSONL memory persistence through `LocalMemoryStore`.
- Ranked keyword retrieval through `MemoryRanker`, with configurable signal weights and recency half-life.
- CLI commands for processing, listing, searching, getting records, and store statistics.
- Tests for core behavior, local store behavior, CLI behavior, and graph isolation.
- Supported assertion conflict detection, automatic/explicit resolution, and durable-memory consolidation.
- Revision history and idempotent bulk reconciliation of existing records.

Partially implemented:

- Graph language coverage: temporal storage/traversal work; automatic extraction covers only supported complete assertions. Structured relationships support richer caller-provided data.
- Memory lifecycle: task deadline parsing, expiry filtering, resolved-task visibility, and inactivity archive views exist. Scheduled physical expiry cleanup and lossless compaction are implemented; archive migration remains planned.
- Importance modeling: hybrid default (trained csv5k ensemble at 25 %, heuristic at 75 %) with a heuristic-only opt-out; the model is trained on synthetic sentences, so conversational validation remains outstanding; the system-level measurement (16 Oct 2026) left LongMemEval user-fact retention unchanged at 90.55 % (see [README "Importance scoring (default)"](README.md#importance-scoring-default)).
- Retrieval quality: keyword/vector/graph/hybrid retrieval work; external results expose extraction and retention limitations.

Planned:

- In-domain (conversational) validation of the hybrid importance scorer; the csv5k ensemble is trained and tested only on synthetic memory sentences.
- Human-rated numeric importance and more representative task/greeting labels.
- Production storage backend.
- Broader graph entity/relation extraction from paragraphs.
- Controlled forgetting.
- Continuous learning from feedback/retrieval success.
- Evaluation against vector-only RAG, Mem0, MemGPT, LangMem, or similar baselines.

Out of scope for the current phase:

- Neo4j setup.
- Distributed graph deployment.
- Unbounded graph reasoning.
- Training custom embedding models.
- Fine-tuning generation models.
- Cloud deployment.

## 4. Current System Overview

The current implementation solves two narrow but important problems: converting raw conversational content into a structured memory object, and saving/inspecting those records through a local JSONL store and CLI.

It provides local append-only storage, ranked keyword/semantic/hybrid retrieval, cited RAG, and temporal graph traversal. Production concurrent storage remains planned.

Current objective:

- Accept a raw interaction.
- Classify the memory type.
- Extract useful memory features.
- Score memory importance.
- Assign a lifecycle tier.
- Return a normalized record.
- Save records locally as JSONL when requested through the CLI or `LocalMemoryStore`.
- Inspect stored records through list, get, search, and stats commands.

Current research-methodology alignment:

- The base paper discusses semantic, episodic, procedural, and emotional memory, plus hybrid memory systems.
- The capstone adapts that direction into implementable categories: semantic, episodic, procedural, preference, task, and temporary.
- The current repository implements classification and lifecycle foundations, but not the full hybrid memory architecture yet.

## 5. Current Architecture

```mermaid
flowchart TD
    A[Raw user or assistant message] --> B[MemoryInput]
    B --> C[MemoryClassifier]
    C --> D[MemoryRecord.from_input]
    D --> E[FeatureExtractor]
    E --> F[HybridImportanceScorer model plus heuristic]
    F --> G[LifecycleManager]
    G --> H[Final MemoryRecord]
    H --> I[to_dict serialization]
    I --> J[LocalMemoryStore JSONL]
    J --> K[CLI list search get stats]
```

Current architecture properties:

- Runs in-process as Python code.
- Uses no external services.
- Uses a local JSONL file store for development/demo persistence.
- Uses no graph database.
- Uses no vector database.
- Scores importance by default with a local trained ensemble (xgboost, onnxruntime, and the BGE encoder cached in `data/rag/models`, which the scorer never downloads); `python -m training.ensemble.run --csv /path/to/cognimem_memory_importance_5000.csv` rebuilds it offline (the flag is needed only on the first run of a fresh clone), and torch is needed only for training.
- Produces serializable Python objects and JSON CLI output.

Default conversational rules use whole-word matching, indirect preferences, explicit constraint/event recognition, conservative negation handling, and short-term handling of one-off tasks. The record's category comes from these default rules; the hybrid scorer's model computes its own inputs from the text with the legacy feature path (`MemoryClassifier(legacy=True)` plus `MLFeatureExtractor` in `main/ml.py`). See rule experiment for results and limitations.

## 6. Current Component Design

| Component | File | Responsibility | Input | Output |
| --- | --- | --- | --- | --- |
| Public API | `main/__init__.py` | Exposes the Memory Core classes and enums. | Imports from caller code. | Importable package API. |
| Memory models | `main/domain/models.py` | Defines memory input, memory record, categories, tiers, timestamps, and serialization. | Raw field values or dict payload. | `MemoryInput` and `MemoryRecord`. |
| Classifier | `main/domain/classification.py` | Categorizes content using deterministic keyword/pattern rules. | `MemoryInput`. | `MemoryCategory`. |
| Feature extractor | `main/domain/features.py` | Converts content and metadata into numeric scoring features. | `MemoryInput`, `MemoryRecord`. | `dict[str, float]`. |
| Heuristic importance scorer | `main/domain/scoring.py` | Applies heuristic weights to features; 75 % of the hybrid score, or the whole score when opted out. | Feature dictionary. | Float score from `0.0` to `1.0`. |
| Hybrid importance scorer | `main/ensemble.py` | Default scorer: loads the validated `EnsembleModel` bundle, predicts low/mid/high from the text, and blends the class value with the heuristic score. | Content and feature dictionary. | Float score from `0.0` to `1.0` plus provenance. |
| Lifecycle manager | `main/domain/lifecycle.py` | Assigns memory tier and expiry/archive hints. | `MemoryRecord`, score. | `MemoryTier`, timestamp hints. |
| Memory orchestrator | `main/application/core.py` | Runs the full processing pipeline. | `MemoryInput`. | Final `MemoryRecord`. |
| Local memory store | `main/storage/store.py` | Saves, loads, filters, gets, and keyword-searches JSONL memory records. | `MemoryRecord` or filter/query arguments. | Stored or retrieved `MemoryRecord` objects. |
| Retrieval ranker | `main/retrieval/ranker.py` | Ranks whole-word matches by keyword coverage, category, tier, recency, and importance. | Query and filtered records. | Ordered records. |
| CLI | `main/__main__.py` | Provides terminal commands for process, list, search, get, and stats. | Command-line arguments. | JSON output. |
| Graph contract | `main/graph/interfaces.py` | Defines the graph adapter protocol. | `MemoryRecord`. | Graph integration contract. |
| Tests | `tests/test_memory_core.py`, `tests/test_memory_store_cli.py` | Verifies current behavior. | Unit test examples. | Passing tests. |

Task completion/cancellation and multi-memory processing are implemented conservatively via `LocalMemoryStore.ingest()` and `MemoryCore.process_many()`. The JSONL store retains revisions, with the latest row per ID used for reads. Detailed policies and limitations: lifecycle changes.

## 7. Current Data Contracts

### MemoryInput

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `content` | `str` | Yes | Raw message text. Must not be empty. |
| `user_id` | `str` | Yes | User identifier. Must not be empty. |
| `session_id` | `str` | Yes | Session/conversation identifier. Must not be empty. |
| `role` | `str` | No | Source role. Defaults to `user`. |
| `timestamp` | `datetime` | No | Message timestamp. Defaults to current UTC time. |
| `metadata` | `dict[str, Any]` | No | Optional caller-provided metadata such as `interaction_score`. |

### MemoryRecord

| Field | Type | Description |
| --- | --- | --- |
| `id` | `str` | Generated UUID for the memory record. |
| `content` | `str` | Cleaned memory content. |
| `user_id` | `str` | User identifier. |
| `session_id` | `str` | Session identifier. |
| `category` | `MemoryCategory` | Classified memory category. |
| `tier` | `MemoryTier` | Assigned lifecycle tier. |
| `created_at` | `datetime` | Creation timestamp. |
| `updated_at` | `datetime` | Last update/access timestamp. |
| `confidence` | `float` | Current default is `1.0`. |
| `importance_score` | `float` | Importance score: hybrid by default, heuristic-only when opted out. |
| `access_count` | `int` | Access counter, incremented by `touch()`. |
| `source_metadata` | `dict` | Metadata copied from `MemoryInput`, plus `importance_model` provenance written by the core (hybrid: `type`, `model_weight`, `model_class`, `model_probabilities`, `model_score`, `small_talk_penalty`, `heuristic_score`, `model_sha256`, `encoder`; heuristic: `{"type": "heuristic"}`). |
| `features` | `dict[str, float]` | Extracted scoring features. |
| `expires_at` | `datetime | None` | Expiry hint for working/short-term memory. |
| `archive_after` | `datetime | None` | Archive hint for long-term memory. |

## 8. Current Data Flow

Input format:

```python
MemoryInput(
    content="I prefer concise technical summaries.",
    user_id="user_001",
    session_id="session_001",
    metadata={"interaction_score": 0.5},
)
```

Intermediate transformations:

- The classifier maps content to one category.
- A `MemoryRecord` is created from the input.
- Feature extraction calculates category flags, recency, entity density, task/preference indicators, sentiment strength, access frequency, and word-count normalization.
- Importance scoring applies heuristic weights and, by default, blends them with the trained model: `round(0.25 × model_score + 0.75 × heuristic_score, 4)`.
- Lifecycle assignment chooses the memory tier and timestamp hints.

Output format:

```python
{
    "id": "generated-uuid",
    "content": "I prefer concise technical summaries.",
    "user_id": "user_001",
    "session_id": "session_001",
    "category": "preference",
    "tier": "long_term",
    "importance_score": 0.6737,  # MemoryCore.with_heuristic() on a just-received message; the default also blends in the model
    "features": {"category_preference": 1.0, "...": "..."},
    "source_metadata": {"interaction_score": 0.5, "importance_model": {"type": "heuristic"}},
    "expires_at": None,
    "archive_after": "future UTC timestamp"
}
```

## 9. Current Model Design

The default importance scorer is hybrid: `HybridImportanceScorer` (`main/ensemble.py`) combines the csv5k "Ensemble + thresholds" three-class (low/mid/high) model with the unchanged rule-based `ImportanceScorer`, giving the model 25 % of the score. `COGNIMEM_SCORER=heuristic`, `--scorer heuristic`, or `MemoryCore.with_heuristic()` opt out; the API builds one local-only BGE encoder (cache `data/rag/models`, no download) shared by retrieval and the scorer, reports the scorer on `/health/ready`, and stops at start-up with the download command and the `COGNIMEM_SCORER=heuristic` opt-out if that cache is missing; with the opt-out it starts without BGE. On the csv5k test split the model reaches macro F1 0.8912 (95 % sentence-bootstrap interval 0.809-0.949) and 89.09 % accuracy, against 0.2529 for the most-common-class baseline and 0.9230 for the oracle; see [the ensemble report](evaluation/reports/retention_ensemble.md). At system level (16 Oct 2026, snapshots `hybrid` and `heuristic_control`) the hybrid matches the heuristic on LongMemEval user-fact retention (90.55 %) and classification accuracy, and the model's score does not separate answer-bearing turns (AUC 0.482 vs 0.582 for the heuristic); see [README "Importance scoring (default)"](README.md#importance-scoring-default).

Current classifier:

- Type: deterministic rule-based classifier.
- Inputs: raw content from `MemoryInput`.
- Output: one `MemoryCategory`.
- Method: keyword and regular-expression matching.

Current importance model:

- Type: default hybrid scorer (trained three-class ensemble plus heuristic), or the heuristic scorer alone.
- Inputs: extracted feature dictionary for the heuristic. The model computes its own inputs from the text: 13 text features plus 512 hashed word counts for XGBoost, and the frozen `BAAI/bge-small-en-v1.5` embedding (384 values) plus the 13 numeric features for the deep head.
- Output: float score from `0.0` to `1.0`, with provenance in `source_metadata["importance_model"]`.
- Method: a temperature-calibrated XGBoost model (525 features, max depth 3, min child weight 3, 304 boosting rounds) is blended 0.5/0.5 with the average of three seeds (42, 43, 44) of a deep head (text branch 384→192→128, numeric 13→64→32, joint 160→128→64→3; LayerNorm/GELU/dropout 0.2), each seed temperature-calibrated and the average calibrated again; class weights 1 / 0.7 / 1 pick low/mid/high. The predicted class's value (low 0.3102, mid 0.5792, high 0.8372, the mean csv5k score of that class's rows among the 3,992 fit and validation rows, test split excluded; × 0.25 for `temporary` small talk) is `model_score`, and the final score is `round(0.25 × model_score + 0.75 × heuristic_score, 4)`.

Current lifecycle model:

- Type: deterministic rule-based tier assignment.
- Inputs: `MemoryRecord` and importance score.
- Output: one `MemoryTier`.
- Method: threshold and category checks. Thresholds are unchanged by the hybrid default: working below 0.25 (or `temporary`), long-term at >= 0.42 for durable categories, archive rule >= 0.42; balanced cleanup archives rather than deletes expired records at >= 0.65 (`ARCHIVE_IMPORTANCE` in `main/storage/cleanup.py`).

The offline training workflow (`python -m training.ensemble.run`, package `training/ensemble/`) splits csv5k by sentence, chooses temperatures, blend weight, and class weights on validation only, and evaluates once on the test split. The first run on a fresh clone needs `--csv /path/to/cognimem_memory_importance_5000.csv`, which copies the file to git-ignored `data/experiments/csv5k.csv`; later runs need no flag. It reproduces the model files (`xgboost.ubj` and the three ONNX files, whose SHA-256 values `metadata.json` lists under `files`) bit for bit; `metadata.json` carries a `created_at` timestamp, so its own hash (`model_sha256` in provenance and on `/health/ready`) changes on every retrain. Working files go to git-ignored `artifacts/ensemble_training/`. The serving bundle `artifacts/importance_ensemble/` (`xgboost.ubj`, `deep_seed_42/43/44.onnx`, and `metadata.json` with sha256 checks and golden vectors) is tracked in git. At load `EnsembleModel` checks the checksums, schema, boosting rounds, ONNX interfaces, temperatures, weights, and encoder id; the golden vectors are checked by the test suite (`tests/test_hybrid_importance.py`), not at load. A missing model raises an error naming the training command and the heuristic opt-out; a missing encoder, one naming the download command and the opt-out. Scoring costs roughly 14 ms per user message and about 100 ms per long assistant message, dominated by the BGE embedding. There is no model registry.

## 10. Current Storage Design

Current storage has two levels:

- In-memory Python objects returned by `MemoryCore.process()`.
- Local JSONL persistence through `LocalMemoryStore`, with the default CLI path `memory_store/memories.jsonl`.

| Data type | Current storage |
| --- | --- |
| Raw datasets | Research documents plus the csv5k source (`cognimem_memory_importance_5000.csv`), copied to `data/experiments/csv5k.csv` for training. |
| Processed datasets | Features and the sentence-grouped split computed by `training/ensemble/`; working files under git-ignored `artifacts/ensemble_training/`. |
| Features | Stored inside returned `MemoryRecord.features` and persisted in JSONL when saved. |
| Trained models | XGBoost model, three deep-head ONNX seeds, and `metadata.json` (sha256 checks, golden vectors) under git-tracked `artifacts/importance_ensemble/`; the BGE encoder is cached in `data/rag/models`. |
| Experiment results | Importance-model test metrics in `evaluation/reports/retention_ensemble.{json,md}`, summarised into `retention_summary.json`; dated `<tag>_summary.json` snapshots keep older numbers as history. |
| Logs | Not implemented. |
| Configuration files | Not implemented. |
| Memory records | In-memory object and optional JSONL records via `LocalMemoryStore`. |

JSONL remains the local source log with cooperative cross-process locking on macOS/Linux. Rebuildable SQLite indexes provide vector search and temporal graph traversal.

## 11. Current Evaluation

Current evaluation includes automated tests, the csv5k importance-model test split ([report](evaluation/reports/retention_ensemble.md)), and a 48-case developer-authored conversational check. `scripts/measure_all.py --scorer-steps-only --scorer hybrid|heuristic` re-runs only the steps that use the scorer (retention, classification, lifecycle, and the summary); those runners record which scorer ran, and classification and lifecycle pin the clock (`evaluation/clock.py`). The system-level measurement of the hybrid default is finished (16 Oct 2026): LongMemEval user-fact retention (90.55 %), classification accuracy, and the lifecycle checks (16/16) match the heuristic, importance-range agreement rises from 50.0 % to 58.3 % (a constant 0.5 scores 83.3 %), and the LongMemEval replay took 581 s instead of 14 s; see [README "Importance scoring (default)"](README.md#importance-scoring-default). Human review remains pending; see `evaluation/README.md`.

The tests verify:

- Classification for semantic, episodic, procedural, preference, task, and temporary memory.
- Durable memories score higher than greetings.
- Lifecycle tier assignment.
- Serialization/deserialization stability.
- Local JSONL save/list/get/search behavior.
- CLI process/list/get behavior.
- Absence of Neo4j/graph database dependency in the Memory Core.

Run tests:

```bash
python3 -m unittest -v
```

## 12. Current Data Flow Diagram

```mermaid
flowchart LR
    A[MemoryInput content user_id session_id metadata] --> B[Rule-based classification]
    B --> C[MemoryRecord skeleton]
    C --> D[Feature extraction]
    D --> E[Hybrid importance score model plus heuristic]
    E --> F[Tier assignment]
    F --> G[Serializable MemoryRecord]
    G --> H[Optional JSONL save]
    H --> I[CLI list search get stats]
```

## 13. Complete Capstone System Overview

The complete CogniMem system will implement a production-oriented hybrid memory engine for personalized LLM agents. It will combine rule-based and ML-based memory decisions, persistent stores, vector retrieval, graph-based temporal reasoning, consolidation, conflict resolution, and evaluation.

The planned system is supported by the capstone proposal and base-paper analysis. The proposal lists working memory, short-term memory, long-term memory, archive memory, RAG, knowledge graphs, and machine learning as core architecture elements.

## 14. Complete Capstone Architecture

```mermaid
flowchart TD
    A[Conversation events] --> B[Input ingestion]
    B --> C[Safety and preprocessing]
    C --> D[Memory classification]
    D --> E[Feature extraction]
    E --> F[ML importance prediction]
    F --> G[Lifecycle decision engine]
    G --> H[Working memory]
    G --> I[Short-term memory]
    G --> J[Long-term memory]
    G --> K[Archive memory]
    J --> L[Vector index]
    J --> M[Temporal knowledge graph]
    I --> L
    M --> N[Conflict detection]
    N --> O[Conflict resolution]
    J --> P[Memory consolidation]
    P --> J
    H --> Q[Hybrid retrieval engine]
    I --> Q
    J --> Q
    K --> Q
    L --> Q
    M --> Q
    Q --> R[LLM context builder]
    R --> S[Personalized response]
    Q --> T[Retrieval feedback]
    T --> F
    T --> G
```

Planned complete architecture responsibilities:

- Ingest conversation events with metadata.
- Preprocess and validate input.
- Classify memories into supported memory categories.
- Extract features for ML scoring.
- Predict importance with a trained model (implemented: the XGBoost plus deep-head csv5k ensemble, blended with the heuristic as the default scorer).
- Store memories in appropriate lifecycle tiers.
- Build vector indexes for semantic retrieval.
- Build a temporal knowledge graph for relationships and chronology.
- Detect and resolve contradictory memories.
- Consolidate repeated observations into stronger memories.
- Forget or archive low-value/stale memories.
- Retrieve memory using vector similarity, graph traversal, timestamps, importance score, and user context.
- Evaluate against traditional RAG and existing memory-system baselines.

## 15. Complete System Components

| Component | Planned responsibility | Current status |
| --- | --- | --- |
| Ingestion layer | Accept user/assistant messages and metadata. | Partially implemented through `MemoryInput`. |
| Preprocessing layer | Clean, normalize, validate, and optionally redact input. | Basic validation only. |
| Memory classifier | Classify memory type. | Rule-based version implemented. |
| Feature extractor | Generate scoring features. | Baseline implemented. |
| Importance predictor | Decide storage value using ML. | Hybrid default implemented (trained csv5k ensemble at 25 %, heuristic at 75 %); heuristic-only opt-out. Conversational validation pending. |
| Lifecycle manager | Assign tier, expiry, archive behavior. | Deadlines, inactivity archival, and expiry visibility implemented. |
| Memory store | Persist memory records. | Local JSONL implemented; production storage planned. |
| Vector store | Store embeddings for retrieval. | SQLite normalized vectors with exact cosine retrieval (one numpy matrix product) by default; random-hyperplane LSH ANN retrieval is opt-in (`COGNIMEM_ANN_ENABLED`, threshold 5,000 memories) and experimental. |
| Temporal knowledge graph | Store entities, relationships, and timestamps. | SQLite graph with bitemporal edges, aliases, evidence, and bounded traversal implemented. |
| Conflict resolver | Detect contradictory memories and pick retained fact. | Implemented for supported assertions, with automatic policy, explicit selection, and history. |
| Consolidation engine | Merge repeated observations into useful summaries. | Equivalent claims and exact durable duplicates consolidate with source evidence. Generalized knowledge inference is not implemented. |
| Forgetting engine | Expire/archive memories based on value and age. | Read visibility, archive views, physical expiry cleanup, and lossless compaction implemented. |
| Hybrid retriever | Combine vector, graph, temporal, importance, and context signals. | Weighted rank fusion builds a 100-candidate pool (`COGNIMEM_CANDIDATE_POOL`); a local cross-encoder reranks it, then graph paths and relation validity reorder the reranked list. Importance/category/tier/recency enter the keyword channel. |
| CLI | Developer access surface for process/list/search/get/stats. | Implemented. |
| REST API | HTTP access surface for backend/UI/agent integration. | FastAPI implementation with JWT/RBAC, privacy endpoints, validation, and standardized responses. |
| Evaluation pipeline | Compare retrieval accuracy, personalization, efficiency, coherence. | External retrieval/retention reports implemented; human judgment and generated-answer quality pending. |

## 16. Complete Training Pipeline

The current training pipeline (`training/ensemble/`, run with `python -m training.ensemble.run`, plus `--csv /path/to/cognimem_memory_importance_5000.csv` on the first run of a fresh clone) uses csv5k (`cognimem_memory_importance_5000.csv`): 5,000 AI-generated memory sentences (443 distinct after removing a " (example N)" suffix) with synthetic importance scores 0.20-1.00, binned into low (< 0.4), mid, and high (>= 0.7). The split is grouped by sentence with seed 42: 3,200 / 792 / 1,008 rows (286 / 71 / 86 sentences). It does not provide human numeric importance ratings or lifecycle tier labels.

```mermaid
flowchart LR
    A[csv5k memory sentences] --> B[Text features hashed words and BGE embedding]
    B --> C[Train XGBoost and three deep-head seeds]
    C --> D[Calibrate blend and weight classes on validation]
    D --> E[Save artifacts importance_ensemble]
    E --> F[HybridImportanceScorer default]
```

Planned training data fields:

- `content`
- `category`
- `importance_score` or `importance_label`
- `tier`
- `recency`
- `access_frequency`
- `sentiment_strength`
- `entity_density`
- `interaction_signal`
- `retrieval_success`

csv5k supplies the current low/mid/high importance target, which differs from the invalid/short-term/long-term target of the removed XGBoost-only and neural scorers. Its limits: the sentences and scores are AI-generated, not human ratings; the test split has only 86 distinct sentences; and conversation turns are far longer than the csv5k sentences (at most 13 words, LongMemEval user turns median 42 words), so the model runs out of distribution, a reason to keep its share small. Its 25 % weight is a fixed, user-chosen setting that was not tuned or calibrated. Before the small-talk penalty it rates greetings mid or high. Human-rated numeric importance, task, greeting, and lifecycle tier labels are still needed.

## 17. Complete Retrieval Pipeline

```mermaid
flowchart TD
    A[User query] --> B[Query preprocessing]
    B --> C[Keyword/category filters]
    B --> D[Vector similarity retrieval]
    B --> E[Temporal graph traversal]
    C --> F[Candidate memories]
    D --> F
    E --> F
    F --> G[Rank by relevance importance recency confidence]
    G --> H[Context builder]
    H --> I[LLM prompt context]
```

The final retrieval strategy should combine:

- vector similarity
- graph traversal
- timestamp/recency
- importance score
- confidence
- user/session context
- memory tier

The local implementation uses keyword/dense/graph rank fusion, local cross-encoder
reranking, a character-budgeted context, and cited Gemini responses. See
implementation details; production scale and broader
extraction remain future work.

## 18. APIs and Interfaces

Current public Python interface:

```python
from main import LocalMemoryStore, MemoryCore, MemoryInput

record = MemoryCore().process(
    MemoryInput(
        content="I prefer concise technical summaries.",
        user_id="user_001",
        session_id="session_001",
    )
)

LocalMemoryStore().save(record)
```

Current CLI interface:

```bash
python3 -m main process --user-id user_001 --session-id session_001 "I prefer concise summaries"
python3 -m main process --scorer heuristic --user-id user_001 --session-id session_001 "I prefer concise summaries"
python3 -m main list --user-id user_001 --pretty
python3 -m main search "technical summaries" --user-id user_001 --pretty
python3 -m main get MEMORY_ID --pretty
python3 -m main stats --pretty
```

Implemented REST API examples:

```text
POST /api/v1/auth/signup
POST /api/v1/auth/login
POST /api/v1/memories
GET  /api/v1/memories
GET  /api/v1/memories/{id}
POST /api/v1/memories/search
GET  /api/v1/me/export?format=json|csv
DELETE /api/v1/me
GET /health/live
GET /health/ready
GET /metrics
```

## 19. Security and Privacy Considerations

Current state:

- Local JSONL persistence remains authoritative and may contain sensitive user memory content.
- Argon2 password hashing and short-lived JWT access tokens are implemented.
- Protected routes enforce user/admin roles and derive ownership from the authenticated subject.
- Structured JSON logging, request IDs, Prometheus metrics, and health probes are implemented.
- JSON/CSV export and account deletion are implemented.

Future requirements:

- User memory must be isolated by `user_id`.
- Sensitive content should be minimized and redacted where possible.
- Logs should not leak raw private memory content.
- Any graph/vector/database layer must enforce user-level data boundaries.
- Retrieval must avoid mixing memories between users or sessions.

## 20. Operational Readiness

Operational signals now include HTTP request counts/latency, authentication events, memory-ingestion counts, JSON structured request logs, and liveness/readiness checks.

Future operational signals:

| Signal | Purpose |
| --- | --- |
| Processing success rate | Confirm memory events are processed reliably. |
| Classification distribution | Detect classifier drift or rule mistakes. |
| Importance score distribution | Detect scoring bias or threshold problems. |
| Retrieval precision/recall | Evaluate memory usefulness. |
| Storage growth | Monitor JSONL and future database memory bloat. |
| Conflict rate | Track contradictory memories. |
| Consolidation rate | Measure reduction in repeated memories. |
| Forgetting/archive rate | Confirm lifecycle cleanup works. |

## 21. Alternatives Considered

| Alternative | Why considered | Why not selected as the current architecture |
| --- | --- | --- |
| Vector-only RAG memory | Simple and common for conversational memory. | The proposal identifies limitations in chronology, personalization, lifecycle management, and conflict handling. |
| Graph-only memory | Strong for relationships and temporal reasoning. | Does not solve semantic similarity retrieval alone; implemented as one hybrid retrieval channel. |
| LLM-only memory extraction | Flexible and semantically rich. | Adds cost, latency, nondeterminism, and external dependencies before the core is stable. |
| ML-first classifier/scorer | Better long-term adaptability. | Human-labeled conversational data does not exist yet; the trained model (synthetic csv5k data) supplies only 25 % of the default hybrid score, and classification stays rule-based. |
| Full-stack implementation first | Would show an end-to-end demo earlier. | Higher risk because core contracts, memory schema, and lifecycle behavior need to stabilize first. |

## 22. Open Questions

- Should the next production storage backend be SQLite, Postgres, or a document database?
- Which independent, conversation-disjoint dataset can validate the hybrid importance scorer and lifecycle decisions?
- How should the local LSH ANN configuration be tuned against workload size and recall?
- How should paragraph-level extraction extend the implemented typed bitemporal graph?
- Which independent human judgments and full-system baselines should extend LoCoMo/LongMemEval?
- How should privacy, deletion, and export be handled for user memories?

## 23. Recommended Next Steps

1. Configure the implemented conservative expiry cleanup worker; broader forgetting policies remain future work.
2. Complete production storage and deployment as a separate infrastructure track.
3. Improve extraction coverage and evaluate hosted answer quality.
4. Extend external retrieval/retention evidence with independent human conflict/personalization judgments.
5. Add background reminder delivery and the human feedback loop as separate capstone tracks.
6. Tune the implemented LSH ANN configuration when workloads require it.

ML modeling and its datasets are a separate research track.

## 24. Current Milestone

The current milestone is:

```text
Structured Memory Core, ranked keyword retrieval, local JSONL store, and CLI implemented.
```

The project has implemented memory structure, processing decisions, local JSONL storage, ranked keyword search, and CLI access. It now includes local vector/hybrid retrieval, cited RAG, temporal graph indexes, and external evaluation. Production concurrent storage, deployment, broader extraction, background reminder delivery, human feedback, and human quality evaluation remain. ML research is separate.

## 25. Conversational Task Updates

The heuristic recognizes indirect preferences and retains their original negation.
`process_many()` splits explicit independent clauses outside quoted spans. Task
state changes are recognized separately from classification; questions, reported
speech, hypotheticals, and negated completion cannot resolve tasks.

`ingest()` identifies an active task belonging to the observation's user, preferring
an exact normalized object and then a unique noun subset. Numeric identifiers
must match. Bare pronouns are restricted to one candidate in the same session.
Ambiguity is returned with candidate IDs. `task_id` supplies an explicit reference.
Updates preserve the task ID and append a revision plus an observation. Duplicate
event IDs are idempotent. Event chronology is enforced against `updated_at`.

Rescheduling parses the replacement date separately from the old date, updates
`due_at`, and recalculates one-off expiry. Relative postponement uses the current
deadline. Unknown/alternative dates do not mutate state. A narrow “not due X but Y”
correction is supported without treating general negation as completion.

Recurring tasks store `recurrence` (unit, interval, anchor, index), `next_due_at`,
and `task_occurrences`. `due_at` remains null for the standing series. The first
anchor comes from the input date/time or explicit `due_at`. Supported schedules
are daily, weekly, monthly, yearly, one named weekday, or integer intervals.
Calendar month/year arithmetic clamps invalid days against the original anchor,
so January 31 → February 28 → March 31 does not drift. Anchor offsets are fixed;
named timezone/DST schedules and combined/exclusion schedules are unsupported.

Each completion/cancellation advances one occurrence and leaves the series active.
The cursor tracks the oldest unresolved occurrence; reads never advance it. An
unqualified completion must refer to an occurrence due that local day, otherwise
`occurrence_at` is required. That field accepts its original or rescheduled ISO
date/time. Already resolved occurrences are recognized. Occurrence rescheduling
keeps the cadence and cannot cross the next occurrence. `task_scope=series` resets
the anchor or closes the series; re-anchoring cannot overlap resolved history.

See current behavior and verification and
[CLI examples](README.md#layer-2-score-retain-and-manage-change).

## 26. Assertion Reconciliation and Consolidation

`main/domain/claims.py` extracts a narrow single assertion: subject, predicate, normalized
value, polarity, and exclusivity. The default classifier recognizes those forms;
the hybrid scorer's model keeps its own legacy classification preprocessing. Ingestion
recomputes claims from the original content, including older records without
claim fields. Current user assertions with the same subject/predicate are checked
for opposing polarity or different values on an exclusive attribute. Compatible
likes and negative claims about different values remain independent.

`main/domain/reconciliation.py` compares source priority, confidence, observation time,
then importance. The best actual supporting observation determines a group's
rank. A rejected assertion does not supersede compatible existing facts. An exact
tie keeps an existing assertion. Source priority/confidence are caller-supplied,
not authenticated source identity or learned certainty. Automatic decisions,
manual selections, and conflict links are persisted as revisions. `last_confirmed_at` makes explicit reaffirmation participate in observation ordering; it is separate from access time.

`memory_status` is separate from lifecycle tier: active, superseded, consolidated.
Superseded records point to their replacement; consolidated observations point to
a canonical record. That record retains original content plus a summary,
evidence IDs/session IDs, and observation range. Fresh evidence renews retention
and retrieval recency without raising confidence/importance by repetition.
Evidence must exist, belong to the same user, and support the same assertion.

Only durable user facts/preferences/procedures consolidate, using equivalent
claims or exact normalized text. Events and tasks retain separate identities.
A change followed by a return to an old value creates a new assertion episode.
No facts are invented from repeated mentions. Historical, uncertain, reported,
and compound statements are not interpreted as unconditional replacements.
Unsupported language remains stored; geographic and entity aliases are not
resolved by the extractor.

List/search omit superseded and consolidated records unless `include_history`
is enabled. Search also indexes summary text; session filters match evidence
sessions. `get`/`all` retain administrative visibility. `history` returns all
revisions of a specific user-owned ID. Explicit `resolve_conflict` restores a
supported root and supersedes its active contradictions. `reconcile_memories`
replays current durable roots in memory and appends only changed records, making
bulk processing idempotent. It does not rewrite prior JSONL rows or lock writers.

See [README usage](README.md#layer-2-score-retain-and-manage-change) and
verification.

## 28. Retrieval and temporal graph implementation

The implementation contract, SQLite schema behavior, query examples, hosted
configuration, time semantics, extraction bounds, and citation checks are in
docs/retrieval_graph.md. External benchmark results and
limits are in reports/retrieval_graph_report.md.

## Scheduled cleanup and evaluation operations

See cleanup policy, locking, failure recovery, and scheduling,
unified external evaluation, and
[blind independent review instructions](evaluation/README.md). Human
personalization and conflict-policy scores remain pending actual reviews.
