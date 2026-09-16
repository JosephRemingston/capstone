# CogniMem Capstone

CogniMem is a Cognitive Hybrid Memory Architecture for long-term personalized LLM agents.

This repository currently implements the first foundation layer: the **Memory Core**. The Memory Core defines how raw conversation content is converted into a structured memory object. It classifies the memory, extracts scoring signals, estimates importance, assigns a memory tier, and returns a serializable record that can later be stored in a database or used by retrieval systems.

This phase does **not** implement a database server, RAG pipeline, vector search, graph database, trained ML model, or API server. It now includes a local JSONL memory store and a CLI for processing and inspecting records.

## Current Implementation Status

Implemented:

- Memory input structure
- Memory record structure
- Memory categories
- Memory lifecycle tiers
- Rule-based memory classification
- Feature extraction for importance scoring
- Heuristic importance scoring
- Lifecycle tier assignment
- Expiry/archive hints
- Serialization and deserialization
- Local JSONL memory store
- CLI interface
- Placeholder graph adapter interface only
- Unit tests for the Memory Core behavior

Not implemented yet:

- Advanced ranked retrieval
- REST API
- RAG retrieval
- Vector database
- Embeddings
- Production database storage
- Graph database layer
- Neo4j integration
- Temporal knowledge graph construction
- LightGBM/XGBoost training
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
- Durable high-score semantic, procedural, preference, and task memories become `long_term`.
- Other useful memories become `short_term`.
- Stale but useful memories can become `archive`.

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

This is not machine learning yet.

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

## What Still Needs To Be Implemented

The remaining work should be implemented in phases. The current system already has memory structuring, lifecycle decisions, local JSONL persistence, and CLI access.

| Priority | Component | What needs to be implemented | Why it matters |
| --- | --- | --- | --- |
| 1 | Improved retrieval | Add better ranking over stored memories using category, tier, recency, importance score, and keyword relevance. | Makes stored memories actually useful for context recall before vector search is added. |
| 2 | Training dataset generation | Create/export labeled examples with content, category, features, importance labels, and tier labels. | Required before replacing heuristic scoring with ML. |
| 3 | ML importance model | Train LightGBM/XGBoost on extracted features and plug it behind the existing scorer interface. | Moves the project from rule/heuristic scoring toward the proposed ML-based memory importance predictor. |
| 4 | Conflict detection | Detect contradictory memories for the same user, entity, or preference. | Prevents the system from keeping outdated or incompatible facts as equally valid. |
| 5 | Conflict resolution | Resolve contradictions using recency, confidence, importance score, and source metadata. | Supports consistent long-term personalization. |
| 6 | Memory consolidation | Merge repeated memories into higher-level long-term memories. | Reduces memory bloat and turns repeated events into useful durable knowledge. |
| 7 | Controlled forgetting | Expire low-value working/short-term memories and archive useful stale memories. | Keeps storage efficient and prevents irrelevant context buildup. |
| 8 | REST API | Expose memory processing, listing, lookup, and search through HTTP endpoints. | Makes the memory system usable by a backend, UI, or LLM agent. |
| 9 | Vector retrieval/RAG | Add embeddings, vector storage, retrieval, context building, and later LLM prompt integration. | Enables semantic retrieval instead of only keyword matching. |
| 10 | Evaluation pipeline | Measure classification accuracy, retrieval quality, memory efficiency, and personalization quality. | Needed for capstone validation and comparison with baseline systems. |
| 11 | Graph database layer | Add the temporal knowledge graph after the non-graph pipeline is stable. | Enables relationship-aware and time-aware reasoning, but is intentionally deferred. |
| 12 | Monitoring/logging | Add structured logs, metrics, and store health checks. | Required before treating the system as production-ready. |
| 13 | Privacy/security controls | Add redaction, deletion/export, user isolation checks, and safe logging rules. | Important because long-term memory may contain sensitive user information. |

## Training Data Status

No training data is currently used.

The system does not currently train:

- a classifier
- an importance model
- a retrieval ranker
- a consolidation model
- a conflict detection model

Future training data should likely contain examples like:

```csv
content,category,importance_score,tier
"I prefer Python examples",preference,0.90,long_term
"hello",temporary,0.05,working
"Yesterday I met my guide",episodic,0.50,short_term
"Remind me to submit the report tomorrow",task,0.85,long_term
```

That dataset can later be used to train a LightGBM/XGBoost importance predictor or a learned classifier.

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

Search stored memories by simple keyword matching:

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
- CLI process/list/get behavior
- graph database isolation

## Recommended Next Implementation Steps

Recommended order:

1. Improve retrieval
   - Retrieve by user ID
   - Filter by category and tier
   - Current keyword search can be expanded into ranked retrieval and later vector retrieval

2. Add training dataset generation
   - Create labeled examples for category, importance, and tier
   - Export CSV for ML experiments

3. Add ML importance model
   - Train LightGBM/XGBoost using extracted features
   - Keep the same `score(features)` interface

4. Add conflict detection
   - Detect incompatible memories
   - Prefer newer or higher-confidence facts

5. Add memory consolidation
   - Merge repeated observations into stronger long-term memories
   - Example: repeated coffee mentions become `User prefers coffee`

6. Add controlled forgetting
   - Periodically expire low-value working/short-term memories
   - Archive useful old memories

7. Add RAG/vector retrieval
   - Generate embeddings
   - Store vectors
   - Retrieve relevant memories for prompts

8. Add graph database layer later
   - Temporal knowledge graph
   - Entity relationships
   - Conflict-aware graph updates
   - Graph traversal as one retrieval signal

## Current Milestone

The current milestone is:

```text
Structured Memory Core, local JSONL store, and CLI implemented.
```

In other words, we have implemented the memory representation, decision pipeline, local JSONL storage, and CLI access. We have not yet implemented production database storage, advanced retrieval, ML training, RAG, graph, API, or deployment layers.
