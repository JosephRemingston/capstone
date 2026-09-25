# CogniMem Capstone

CogniMem is a Cognitive Hybrid Memory Architecture for long-term personalized LLM agents.

This repository currently implements the first foundation layer: the **Memory Core**. The Memory Core defines how raw conversation content is converted into a structured memory object. It classifies the memory, extracts scoring signals, estimates importance, assigns a memory tier, and returns a serializable record that can later be stored in a database or used by retrieval systems.

This phase does **not** implement a database server, RAG pipeline, vector search, graph database, API server. It now includes a local JSONL memory store and a CLI for processing and inspecting records.

## Current Implementation Status

Implemented:

- Memory input structure
- Memory record structure
- Memory categories
- Memory lifecycle tiers
- Rule-based memory classification
- Feature extraction for importance scoring
- Heuristic importance scoring
- Optional trained XGBoost importance scorer, evaluated on Hippocorpus
- Reproducible importance-model training, held-out metrics, and native model artifacts
- Lifecycle tier assignment
- Expiry/archive hints
- Serialization and deserialization
- Local JSONL memory store
- Advanced ranked keyword retrieval using relevance, category, tier, recency, and importance
- CLI interface
- Placeholder graph adapter interface only
- Unit tests for the Memory Core behavior

Not implemented yet:

- REST API
- RAG retrieval
- Vector database
- Embeddings
- Production database storage
- Graph database layer
- Neo4j integration
- Temporal knowledge graph construction
- Conflict resolution
- Memory consolidation
- Controlled forgetting job
- Evaluation dashboard or experiment runner

## What This Layer Does

The Memory Core takes an incoming user or assistant message and converts it into a normalized memory record.

For example, this input:

```text
I prefer concise technical summaries.
```

can become a structured memory like:

```text
category: preference
tier: long_term
importance_score: calculated by heuristic scoring
expires_at: none
archive_after: future timestamp
```

The goal is to build the internal memory representation first. Later layers can decide where to store it and how to retrieve it.

## Data Flow

The current processing flow is:

```text
Raw message
  -> MemoryInput
  -> MemoryClassifier
  -> MemoryRecord
  -> FeatureExtractor
  -> ImportanceScorer
  -> LifecycleManager
  -> Final MemoryRecord
```

Step by step:

1. `MemoryInput` receives the raw content and metadata.
2. `MemoryClassifier` assigns one memory category.
3. `MemoryRecord.from_input()` creates the normalized memory object.
4. `FeatureExtractor` generates numeric features from the content and metadata.
5. `ImportanceScorer` calculates an importance score from `0.0` to `1.0`.
6. `LifecycleManager` assigns the memory to a lifecycle tier.
7. The final `MemoryRecord` is returned and can be serialized with `to_dict()`.

## Main Interfaces

The public interfaces are exposed from the `main` package.

```python
from main import MemoryCore, MemoryInput

core = MemoryCore()

record = core.process(
    MemoryInput(
        content="I prefer concise technical summaries.",
        user_id="user_001",
        session_id="session_001",
        metadata={"interaction_score": 0.5},
    )
)

print(record.to_dict())
```

## MemoryInput

`MemoryInput` represents raw incoming interaction data.

Fields:

- `content`: raw message text
- `user_id`: user identifier
- `session_id`: conversation/session identifier
- `role`: message role, default is `user`
- `timestamp`: timestamp for the message
- `metadata`: optional caller-provided metadata

Example:

```python
MemoryInput(
    content="Remind me to submit the capstone report tomorrow.",
    user_id="user_001",
    session_id="session_001",
    role="user",
    metadata={"interaction_score": 0.7},
)
```

## MemoryRecord

`MemoryRecord` is the normalized memory object created by the Memory Core.

Fields:

- `id`: generated memory ID
- `content`: cleaned memory content
- `user_id`: user identifier
- `session_id`: session identifier
- `role`: source role
- `category`: memory category
- `tier`: lifecycle tier
- `created_at`: creation timestamp
- `updated_at`: latest update timestamp
- `confidence`: confidence score, currently defaulted to `1.0`
- `importance_score`: score from the baseline importance scorer
- `access_count`: number of times the memory has been touched/accessed
- `source_metadata`: metadata copied from the input
- `features`: extracted numeric scoring features
- `expires_at`: expiry hint for working or short-term memory
- `archive_after`: archive hint for long-term memory

Records can be converted to plain dictionaries:

```python
payload = record.to_dict()
restored = MemoryRecord.from_dict(payload)
```

This is useful for future JSON, JSONL, SQLite, API, or database storage.

## Memory Categories

The system currently supports six memory categories.

| Category | Meaning | Example |
| --- | --- | --- |
| `semantic` | Durable factual knowledge | `My project guide is Prof. Karnam Balaji.` |
| `episodic` | Time-bound events or experiences | `Yesterday I submitted the proposal.` |
| `procedural` | Steps, workflows, or repeated process knowledge | `First run tests, then build Docker.` |
| `preference` | User likes, dislikes, style, or personal choices | `I prefer short technical explanations.` |
| `task` | Future action, reminder, todo, or deadline | `Remind me to submit the report tomorrow.` |
| `temporary` | Low-value transient chat | `Thanks`, `ok`, `hello` |

Classification is currently rule-based. It uses keyword and pattern matching, not a trained model.

## Memory Tiers

The system currently supports four lifecycle tiers.

| Tier | Meaning |
| --- | --- |
| `working` | Very short-lived memory for immediate context |
| `short_term` | Useful recent memory, but not necessarily permanent |
| `long_term` | Durable user facts, preferences, tasks, or procedures |
| `archive` | Old but historically useful memory |

Current lifecycle behavior:

- Temporary messages usually become `working` memory.
- Low-score memories become `working` memory.
- Durable high-score semantic, procedural, and preference memories become `long_term`.
- One-off tasks become `short_term`; explicitly recurring tasks can qualify as `long_term`.
- Other useful memories become `short_term`.
- Stale but useful memories can become `archive`.

## Conversational Heuristic Improvements

The default pipeline now uses whole-word keyword matching, recognizes explicit
preferences/constraints and selected event updates, and keeps short one-off tasks
in short-term memory. Mentions of today/tomorrow count as deadline signals only
in task context. Fresh episodic memories receive a category weight of 0.20.
Short-term expiry remains a fixed 14-day hint; deadlines are not parsed or enforced.

See [the 15-sentence before/after experiment](reports/heuristic_improvements.md).
These are developer-authored regression examples, not human-labeled validation or
training data. No statistical threshold calibration or model retraining occurred.
The optional Hippocorpus model retains its legacy classification/feature inputs;
new lifecycle rules apply to both pipelines. Existing records are unchanged.

## Extracted Features

The feature extractor currently produces simple numeric signals:

- `access_frequency`
- category flags such as `category_preference`, `category_task`, etc.
- `entity_density`
- `has_deadline`
- `interaction_signal`
- `preference_signal`
- `recency`
- `sentiment_strength`
- `task_signal`
- `word_count_norm`

These features are used by the heuristic importance scorer. They are also designed to become the input features for a future LightGBM/XGBoost model.

## Importance Scoring

The current importance scorer is heuristic. It applies configured weights to extracted features and returns a score between `0.0` and `1.0`.

An optional XGBoost scorer is now trained and integrated. The heuristic remains the default because the Hippocorpus personal-event target has not been validated for conversational retention. See **ML Importance Model** below.

The scorer is intentionally designed behind this simple interface:

```python
score = ImportanceScorer().score(features)
```

Later, the implementation can be replaced with a trained model without changing the rest of the Memory Core pipeline.

## How Data Is Stored Right Now

There is no database server yet.

The system now supports a local JSONL-backed store. By default, CLI commands write to:

```text
memory_store/memories.jsonl
```

This path is ignored by Git because it is local runtime data.

Each line is one serialized `MemoryRecord` dictionary. Records can still be used in memory:

```python
record = core.process(memory_input)
payload = record.to_dict()
```

The local store can save and load records:

```python
from main import LocalMemoryStore

store = LocalMemoryStore()
store.save(record)
records = store.list(user_id="user_001")
matches = store.search("technical summaries", user_id="user_001")
```

The local store is intentionally simple. It does not provide database indexes, concurrent write guarantees, vector search, graph traversal, or server-side querying.

## Ranked Retrieval

`LocalMemoryStore.search()` and the CLI `search` command rank matching records
using `MemoryRanker`. The return format remains a list of memory records.
User, session, category, and tier filters are applied before ranking; the limit
is applied afterward. A zero limit returns no results; negative limits are rejected.

Queries and content are split into case-insensitive whole-word tokens. Punctuation
separates tokens. At least one query token must match; unrelated records are never
included merely because they are important. Duplicate query terms and repeated
content words do not boost relevance. There is no stemming, synonym expansion,
or semantic matching: `python` does not match `pythonic`.

| Signal | Default weight | Calculation |
| --- | --- | --- |
| Keyword relevance | 0.65 | Fraction of unique query tokens present in the content. |
| Category | 0.10 | Semantic/preference: 1.0; procedural/task: 0.9; episodic: 0.6; temporary: 0.1. |
| Tier | 0.05 | Long-term: 1.0; short-term: 0.7; working: 0.4; archive: 0.2. |
| Recency | 0.10 | Exponential decay from creation time, with a 30-day half-life. Future timestamps receive 1.0. |
| Importance | 0.10 | Stored importance score clamped to 0–1; nonfinite values contribute 0. |

The final score is the weighted sum divided by the total weight. Category and tier
values are fixed usefulness preferences, not predictions of query intent. Ties
are resolved by newest creation time, then ascending memory ID. Search does not
modify records, increment access counts, or enforce expiry/archive hints.

Weights and recency half-life can be configured through the Python API:

```python
from main import LocalMemoryStore, MemoryRanker

store = LocalMemoryStore(
    ranker=MemoryRanker(
        weights={
            "keyword_relevance": 0.65,
            "category": 0.10,
            "tier": 0.05,
            "recency": 0.10,
            "importance": 0.10,
        },
        recency_half_life_days=30.0,
    )
)
matches = store.search("Python tests", user_id="user_001", limit=10)
```

All five weights must be finite and nonnegative with a positive finite total.
The half-life must be finite and positive. Ranking remains an in-process heuristic
that scans the local store; vector retrieval and a trained ranker are future work.

## What Still Needs To Be Implemented

The remaining work should be implemented in phases. The current system already has memory structuring, lifecycle decisions, local JSONL persistence, and CLI access.

| Priority | Component | What needs to be implemented | Why it matters |
| --- | --- | --- | --- |
| 1 | In-domain training data | Add conversational importance/category/tier labels beyond Hippocorpus. | Validate transfer to actual agent memories. |
| 2 | ML validation and calibration | Improve the experimental XGBoost model, evaluate conversational retention, and calibrate tier thresholds. | Required before making ML the default. |
| 3 | Conflict detection | Detect contradictory memories for the same user, entity, or preference. | Prevents the system from keeping outdated or incompatible facts as equally valid. |
| 4 | Conflict resolution | Resolve contradictions using recency, confidence, importance score, and source metadata. | Supports consistent long-term personalization. |
| 5 | Memory consolidation | Merge repeated memories into higher-level long-term memories. | Reduces memory bloat and turns repeated events into useful durable knowledge. |
| 6 | Controlled forgetting | Expire low-value working/short-term memories and archive useful stale memories. | Keeps storage efficient and prevents irrelevant context buildup. |
| 7 | REST API | Expose memory processing, listing, lookup, and search through HTTP endpoints. | Makes the memory system usable by a backend, UI, or LLM agent. |
| 8 | Vector retrieval/RAG | Add embeddings, vector storage, retrieval, context building, and later LLM prompt integration. | Enables semantic retrieval instead of only keyword matching. |
| 9 | Evaluation pipeline | Measure classification accuracy, retrieval quality, memory efficiency, and personalization quality. | Needed for capstone validation and comparison with baseline systems. |
| 10 | Graph database layer | Add the temporal knowledge graph after the non-graph pipeline is stable. | Enables relationship-aware and time-aware reasoning, but is intentionally deferred. |
| 11 | Monitoring/logging | Add structured logs, metrics, and store health checks. | Required before treating the system as production-ready. |
| 12 | Privacy/security controls | Add redaction, deletion/export, user isolation checks, and safe logging rules. | Important because long-term memory may contain sensitive user information. |

## ML Importance Model

An XGBoost regression model has been trained on the existing **Hippocorpus**
importance ratings, normalized from 1–5 to 0–1. It uses text-derived features and
hashed word counts, with author/story-family-disjoint train, validation, and test
sets. No synthetic labels were generated.

| Held-out test metric | XGBoost | Mean baseline |
| --- | ---: | ---: |
| MAE | 0.2291 | 0.2424 |
| RMSE | 0.2847 | 0.2947 |
| R² | 0.0311 | -0.0381 |
| Spearman | 0.2367 | Undefined (constant) |

Training used 4,697 stories, validation 1,006, and test 1,007. The improvement is
modest. This model predicts personal-event significance; conversational retention
quality and lifecycle thresholds remain unvalidated. ML therefore replaces the
heuristic only when explicitly selected.

Install optional dependencies and use the trained model:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-ml.txt
.venv/bin/python -m main process "Yesterday I celebrated my graduation." \
  --user-id user_001 --session-id session_001 --scorer xgboost --no-save --pretty
```

Python integration:

```python
from main import MemoryCore, MemoryInput

core = MemoryCore.with_ml()
record = core.process(MemoryInput(
    content="Yesterday I celebrated my graduation.",
    user_id="user_001", session_id="session_001",
))
print(record.importance_score)
```

`MLImportanceScorer.score(features)` implements the same scoring contract as the
heuristic. Use `MLFeatureExtractor` with it; `MemoryCore.with_ml()` configures both.
`--model-dir PATH` or `MemoryCore.with_ml(PATH)` selects a custom artifact directory.
The score flows through existing tier assignment, JSONL persistence, and ranking.
Each ML record includes model provenance in its source metadata.

Retrain and reproduce metrics with `.venv/bin/python -m training.train_importance`.
It downloads the official Microsoft archive into ignored `data/hippocorpus/`.
See [experiment report](reports/importance_report.md),
[full metrics](reports/importance_metrics.json), and
[model metadata](artifacts/importance/metadata.json).

No learned classifier, retrieval ranker, consolidation model, or conflict detection
model is trained. In-domain conversational importance labels and evaluation remain
future work.

## Graph Database Status

The graph database layer is intentionally not implemented.

The file `main/interfaces.py` only defines a future contract:

```python
class GraphMemoryAdapter(Protocol):
    def index(self, record: MemoryRecord) -> None:
        ...
```

This is only a placeholder interface. It does not connect to Neo4j, create nodes, create relationships, build a graph schema, perform graph traversal, or persist anything.

## RAG Status

RAG is not implemented yet.

There is currently no:

- chunking
- embedding model
- vector database
- retriever
- reranker
- LLM prompt construction
- answer generation
- source citation
- hybrid retrieval

The current Memory Core can produce structured memory records that a future RAG layer may store and retrieve.

## CLI

The CLI is available through:

```bash
python3 -m main --help
```

Process and save one memory:

```bash
python3 -m main process \
  --user-id user_001 \
  --session-id session_001 \
  --interaction-score 0.5 \
  --pretty \
  "I prefer concise technical summaries."
```

Process without saving:

```bash
python3 -m main process \
  --user-id user_001 \
  --session-id session_001 \
  --no-save \
  --pretty \
  "Remind me to submit the capstone report tomorrow."
```

List stored memories:

```bash
python3 -m main list --user-id user_001 --pretty
```

Search stored memories with ranked keyword retrieval:

```bash
python3 -m main search "technical summaries" --user-id user_001 --pretty
```

Get one memory by ID:

```bash
python3 -m main get MEMORY_ID --pretty
```

Show local store statistics:

```bash
python3 -m main stats --pretty
```

Use a custom store path:

```bash
python3 -m main --store /tmp/cognimem.jsonl list --pretty
```

The Memory Core can also be used through Python:

```bash
python3 - <<'PY'
from main import LocalMemoryStore, MemoryCore, MemoryInput

core = MemoryCore()
record = core.process(
    MemoryInput(
        content="I prefer concise technical summaries.",
        user_id="user_001",
        session_id="session_001",
    )
)

LocalMemoryStore().save(record)
print(record.to_dict())
PY
```

## Tests

Run the test suite:

```bash
python3 -m unittest -v
```

The tests cover:

- classification for all memory categories
- importance scoring for durable vs temporary memories
- lifecycle tier assignment
- record serialization/deserialization
- local JSONL storage
- ranked retrieval signals, whole-word matching, filters, limits, deterministic ordering, and CLI search
- CLI process/list/get behavior
- graph database isolation

## Recommended Next Implementation Steps

Recommended order:

1. Add in-domain conversational labels
   - Create labeled examples for category, importance, and tier
   - Export CSV for ML experiments

2. Improve and validate the trained ML importance model
   - Evaluate transfer from Hippocorpus to conversations
   - Calibrate lifecycle thresholds before changing the default

3. Add conflict detection
   - Detect incompatible memories
   - Prefer newer or higher-confidence facts

4. Add memory consolidation
   - Merge repeated observations into stronger long-term memories
   - Example: repeated coffee mentions become `User prefers coffee`

5. Add controlled forgetting
   - Periodically expire low-value working/short-term memories
   - Archive useful old memories

6. Add RAG/vector retrieval
   - Generate embeddings
   - Store vectors
   - Retrieve relevant memories for prompts

7. Add graph database layer later
   - Temporal knowledge graph
   - Entity relationships
   - Conflict-aware graph updates
   - Graph traversal as one retrieval signal

## Current Milestone

The current milestone is:

```text
Structured Memory Core, ranked keyword retrieval, optional trained XGBoost scoring, local JSONL store, and CLI implemented.
```

In other words, we have implemented the memory representation, decision pipeline, local JSONL storage, and CLI access. We have not yet implemented production database storage, in-domain ML validation, RAG/vector retrieval, graph, API, or deployment layers.
