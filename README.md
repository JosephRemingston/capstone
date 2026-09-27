# CogniMem Capstone

CogniMem is a Cognitive Hybrid Memory Architecture for long-term personalized LLM agents.

This repository implements the **Memory Core**, local semantic/hybrid retrieval, cited RAG, and a temporal knowledge graph. The Memory Core defines how raw conversation content is converted into a structured memory object. It classifies the memory, extracts scoring signals, estimates importance, assigns a memory tier, and returns a serializable record that can later be stored in a database or used by retrieval systems.

JSONL stores authoritative memory history; SQLite stores rebuildable vector and temporal graph indexes. Hosted answers use **LangChain with Gemini 2.5 Flash**, configured through `.env`. See [setup, examples, and limits](docs/retrieval_graph.md). Production database servers and REST APIs remain future work.

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
- Deadline-aware task expiry, inactivity archive views, and expiry-aware retrieval
- Task completion, cancellation, rescheduling, and recurring occurrences with revision history
- Supported fact/preference conflict detection and resolution
- Durable memory consolidation with summaries and source evidence
- History inspection, explicit conflict selection, and bulk reconciliation
- Optional multi-memory segmentation
- Conversational developer evaluation and a human-review template
- Serialization and deserialization
- Local JSONL memory store
- Advanced ranked keyword retrieval using relevance, category, tier, recency, and importance
- CLI interface
- Local neural embeddings, persistent vector storage, and hybrid retrieval
- Bounded RAG context, Gemini prompt integration, and checked citations
- Typed temporal graph, aliases, relationship history, and multi-hop evidence paths
- External LoCoMo/LongMemEval evaluation and baseline reports
- Unit tests for the Memory Core behavior

Not implemented yet:

- REST API
- Production database storage
- Neo4j integration
- Controlled forgetting job
- Evaluation dashboard, independent human conflict/personalization labels, and live Gemini quality evaluation

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
- `confidence`: caller-provided assertion confidence (0–1), default `1.0`
- `importance_score`: score from the baseline importance scorer
- `access_count`: number of times the memory has been touched/accessed
- `source_metadata`: metadata copied from the input
- `features`: extracted numeric scoring features
- `expires_at`: expiry hint for working or short-term memory
- `archive_after`: archive hint for long-term memory
- `due_at`: parsed/explicit task deadline
- `task_status`: active, completed, or cancelled for tasks
- `related_task_id`: task changed by this observation
- `last_accessed_at`: explicit access timestamp used for inactivity
- `recurrence`: cadence, interval, calendar anchor, and occurrence index
- `next_due_at`: deadline of the current unresolved recurring occurrence
- `task_occurrences`: completed/cancelled occurrences with dates and source event IDs
- `memory_status`: `active`, `superseded`, or `consolidated` (separate from lifecycle tier)
- `claim`: extracted subject, attribute, value, polarity, and exclusivity
- `superseded_by`: current replacement for a conflicting assertion
- `consolidated_into`: canonical record for a repeated observation
- `conflict_ids` / `conflict_resolution`: detected conflicts and the recorded decision/reason
- `evidence_ids` / `evidence_session_ids`: supporting observations and their sessions
- `summary`: faithful summary of a consolidated group
- `first_observed_at` / `last_observed_at`: observation range; explicit selection also reaffirms the last observation time
- `last_confirmed_at`: explicit selection timestamp used to order later assertions

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
preferences/constraints, indirect preferences, and selected event updates, and keeps short one-off tasks
in short-term memory. Mentions of today/tomorrow count as deadline signals only
in task context. Fresh episodic memories receive a category weight of 0.20.
Dated tasks expire 24 hours after their deadline; undated tasks use 14 days.
List/search enforce expiry visibility without deleting records.

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

The JSONL source store is intentionally simple and lacks concurrent writer guarantees or server-side querying. The optional retrieval layer adds SQLite vector/graph indexes; see [retrieval design](docs/retrieval_graph.md).

## Ranked Retrieval

`LocalMemoryStore.search()` and the CLI `search` command rank matching records
using `MemoryRanker`. The return format remains a list of memory records.
User, session, category, and tier filters are applied before ranking; consolidated groups match their supporting sessions; the limit
is applied afterward. A zero limit returns no results; negative limits are rejected.

Queries, original content, and consolidated summaries are split into case-insensitive whole-word tokens. Punctuation
separates tokens. At least one query token must match; unrelated records are never
included merely because they are important. Duplicate query terms and repeated
content words do not boost relevance. There is no stemming, synonym expansion,
or semantic matching: `python` does not match `pythonic`.

| Signal | Default weight | Calculation |
| --- | --- | --- |
| Keyword relevance | 0.65 | Fraction of unique query tokens present in the content. |
| Category | 0.10 | Semantic/preference: 1.0; procedural/task: 0.9; episodic: 0.6; temporary: 0.1. |
| Tier | 0.05 | Long-term: 1.0; short-term: 0.7; working: 0.4; archive: 0.2. |
| Recency | 0.10 | Exponential decay from the latest supporting observation (creation time if absent), with a 30-day half-life. Future timestamps receive 1.0. |
| Importance | 0.10 | Stored importance score clamped to 0–1; nonfinite values contribute 0. |

The final score is the weighted sum divided by the total weight. Category and tier
values are fixed usefulness preferences, not predictions of query intent. Ties
are resolved by newest creation time, then ascending memory ID. Search does not
modify records or increment access counts. Expired records and resolved tasks are
omitted by default, and inactive long-term memories are returned as archive views.
Use `--include-expired`, `--include-resolved`, and `--include-history` to include the corresponding hidden records. Superseded facts and consolidated duplicates are omitted by default.

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
that scans the local store. Semantic/graph/hybrid modes are implemented separately; a learned reranker remains future work.

## What Still Needs To Be Implemented

The remaining work should be implemented in phases. The current system already has memory structuring, lifecycle decisions, local JSONL persistence, and CLI access.

| Priority | Component | What needs to be implemented | Why it matters |
| --- | --- | --- | --- |
| 1 | Persistent cleanup | Add scheduled physical cleanup/compaction beyond the implemented expiry filtering and archive views. | Reclaims storage without losing required history. |
| 2 | REST API | Expose memory processing, listing, lookup, and search through HTTP endpoints. | Makes the memory system usable by a backend, UI, or LLM agent. |
| 3 | Retrieval quality | Improve paragraph-level fact extraction and assess Gemini answers after credentials are configured. | Current bounded graph extraction has low coverage on free-form conversations. |
| 4 | Evaluation quality | Add independent human conflict/personalization labels and full memory-system baselines. | External retrieval/retention runners and recency/keyword/vector/graph/hybrid baselines now exist. |
| 5 | Scale storage | Add an ANN vector backend and production concurrent storage when needed. | Current exact vector scans and local SQLite graph target local workloads. |
| 6 | Monitoring/logging | Add structured logs, metrics, and store health checks. | Required before treating the system as production-ready. |
| 7 | Privacy/security controls | Add redaction, deletion/export, user isolation checks, and safe logging rules. | Important because long-term memory may contain sensitive user information. |

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

## Graph and RAG Status

Implemented: local BGE embeddings, chunked SQLite vector storage, hybrid ranking,
context budgets, LangChain Gemini 2.5 Flash answers with checked citations, and a
temporal SQLite graph with typed entities, aliases, provenance, valid/recorded
time, task relationships, and bounded traversal. The local `.env` has blank key
configuration; fill `GOOGLE_API_KEY` when ready. No hosted call has been made.

See [full setup and examples](docs/retrieval_graph.md) and
[external evaluation results](reports/retrieval_graph_report.md). Automatic graph
extraction is limited to supported assertions; broad paragraph understanding is
not solved. Citation checks verify source references/quotes, not entailment.

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

## Deadlines, Task Updates, and Multiple Memories

```bash
# Separate a standing preference and an actionable reminder.
python3 -m main process "I prefer Python, and remind me to submit the report tomorrow." \
  --user-id u1 --session-id s1 --split --pretty

# Resolves a unique matching active task for this user, or records ambiguity.
python3 -m main process "I submitted the report." --user-id u1 --session-id s1 --pretty

# Explicit deadline / explicit task resolution when needed.
python3 -m main process "Submit the application" --user-id u1 --session-id s1 \
  --due-at "2026-12-01T17:00:00+05:30"
python3 -m main process "I completed it." --user-id u1 --session-id s1 --task-id TASK_ID

# Include historical records omitted from normal retrieval.
python3 -m main list --user-id u1 --include-expired --include-resolved --pretty
```

Python: call `core.process_many(input)` for segmentation, then `store.ingest(record)`
for each record to enable task updates. `process()` returns one record and `save()`
is raw persistence. `--no-save` previews processing without resolving stored tasks.

Deadline parsing supports ISO dates, today/tomorrow/tonight, next week, weekdays,
and numeric durations, with optional times. Date-only deadlines mean end of day
in the input timestamp timezone (UTC by default). Unknown phrasing is not guessed.
Recurring tasks remain standing instructions with a separate occurrence cursor.
Task updates support rescheduling, changed deadlines, completion, and cancellation.
Exact object matches take priority; a unique noun subset supports “that report”
for “budget report.” Numbers must agree. Bare “it” requires exactly one active
candidate in the same session. Ambiguity returns candidate IDs for `--task-id`.
Only user messages update tasks, and older observations cannot revise newer state.

```bash
python3 -m main process "Move the report from Monday to Friday at 5 pm" \
  --user-id u1 --session-id s1
python3 -m main process "Postpone the report by 2 days" --user-id u1 --session-id s1
python3 -m main process "The report is not due Monday but Tuesday" --user-id u1 --session-id s1

# --due-at sets the first occurrence when creating a recurring task.
python3 -m main process "Remind me to pay rent monthly" \
  --user-id u1 --session-id s1 --due-at "2026-10-01T09:00:00+05:30"
python3 -m main process "I paid rent" --user-id u1 --session-id s1 \
  --task-id TASK_ID --occurrence-at "2026-10-01"
python3 -m main process "Cancel rent" --user-id u1 --session-id s1 \
  --task-id TASK_ID --task-scope series
```

Supported recurrence: daily, weekly, monthly, yearly, every named weekday, and
“every N days/weeks/months/years” (including “every other week”). A date/time
anchor preserves month ends and leap days. Without an explicit time, the deadline
is end of day; “morning/evening” does not invent a particular hour. Recurrence uses
the anchor's fixed UTC offset; named timezones and daylight-saving rules are not
implemented. Combined schedules/exclusions return `unsupported_schedule`.

Completing or skipping the current occurrence records its outcome and advances
`next_due_at`; the series stays active. Plain completion applies automatically
only when the current occurrence is due on the observation's local date. For an
early or overdue completion, pass `--occurrence-at` with the current occurrence's
scheduled or rescheduled date/time. Occurrences are processed in order; reads do
not skip overdue occurrences. Reusing an event ID is idempotent, and explicitly
repeating a resolved occurrence returns `already_resolved`.

Rescheduling defaults to the current occurrence and keeps the original cadence.
Moving it to or past the next occurrence returns `occurrence_overlaps_next`.
Use `--task-scope series` to re-anchor the series, or to cancel it entirely.
“Stop reminding me to pay rent” also cancels the series. A new series anchor
cannot overlap completed occurrence history. Unrecognized replacement dates
return `missing_deadline` without changing the task. These statuses appear in
`source_metadata.task_resolution`; applied changes include `task_change`.

Preferences keep their original wording, including negation. The rules recognize
“I'd rather…”, “not a fan of…”, “I don't dislike…”, and “works better for me.”
Uncertain or reported completions do not change tasks. `--split` handles explicit
independent clauses outside quoted spans, while preserving conditional scope,
procedures, and object lists. Arbitrary paraphrases, sarcasm, and general language
understanding remain outside these deterministic rules.

Storage is now an append-only revision log: reads use the latest row per ID.
`get()` and `all()` include historical visibility; list/search hide expired and
resolved tasks by default. Long-term memories are viewed as archived after 90
inactive days, using the latest explicit access, supporting observation, or creation time. No cleanup job or
concurrent-write guarantees are provided.

Run `python3 -m evaluation.evaluate` for the 48-case developer check. Its current
category/tier agreement is 48/48, not independently reviewed accuracy. The blank
[review template](evaluation/review_template.jsonl) and [review guidance](evaluation/README.md)
are ready; no human-reviewed labels have been collected.
See [implementation and limits](reports/lifecycle_improvements.md).

## Conflicts, Current Memories, and Consolidation

CLI processing and `store.ingest(record)` reconcile durable user assertions.
`store.save(record)` remains raw persistence. `process()` / `--no-save` extract
supported claims without consulting or changing stored records.

| Inputs | Result |
| --- | --- |
| “I live in Chennai.” → “I live in Bengaluru.” | Bengaluru becomes current by default; Chennai remains in history. |
| “I like tea.” + “I like coffee.” | Both remain current because the preferences are compatible. |
| “I like coffee.” → “I don't like coffee.” | The newer opposite preference becomes current. |
| “I prefer short answers.” → “I prefer detailed answers.” | The explicit response-length preference changes. |
| “I live in Chennai.” + “I reside in Chennai.” | One retrievable summary, “User lives in Chennai.”, with both source IDs. |

Supported assertions include residence, employer, occupation, name, home city,
hometown, timezone, a named entity's residence/employer, a pet/entity's name,
favorites, likes/dislikes, allergies, response length, and explicit “X over Y”
choices. Subjects and attributes must match. Different positive values conflict
only for attributes treated as single-valued; opposite polarities conflict only
for the same value. Multiple likes and multiple allergies can coexist.

The automatic policy compares **source priority → confidence → observation
timestamp → importance**, in that order. Priority defaults to 0 and confidence
to 1, so later assertions normally win. Exact ties keep an existing assertion.
`metadata.source_priority` / `--source-priority` accepts 0–100;
`metadata.confidence` / `--confidence` accepts 0–1. These are caller-assigned
values, not authenticated trust or automatically calibrated probabilities.
Every decision records its reason and the replacement ID. A consolidated group
uses its strongest actual supporting observation; repetition alone does not
increase confidence or importance. Access timestamps do not decide truth. Explicit selection records a confirmation timestamp, so equal-quality late data does not undo that selection.

Consolidation groups equivalent supported claims, or exact normalized duplicates
of other durable facts, preferences, and procedures. It keeps the original
content and source observations, adds a summary and evidence IDs, and hides
redundant observations from normal retrieval. Counts are `len(evidence_ids)`.
Fresh support renews retention and retrieval recency. Returning to an older value
after a conflicting change starts a new evidence group. Tasks, transient chat,
and separate episodic events are not merged as repeated facts.

```bash
# New assertions automatically reconcile when saved.
python3 -m main process "I live in Chennai." --user-id u1 --session-id s1
python3 -m main process "I live in Bengaluru." --user-id u1 --session-id s2

# Include superseded/consolidated records, or inspect every revision of one ID.
python3 -m main list --user-id u1 --include-history --pretty
python3 -m main history MEMORY_ID --user-id u1 --pretty

# Explicitly choose a supported original/canonical assertion in an active conflict.
python3 -m main resolve --keep MEMORY_ID --user-id u1 --pretty

# Apply reconciliation to existing records, including older/raw-saved memories.
python3 -m main reconcile --user-id u1 --pretty
```

Python equivalents: `store.history(id, user_id=...)`,
`store.resolve_conflict(id, user_id=...)`, and
`store.reconcile_memories(user_id=...)`. Bulk reconciliation appends only changed
rows and is idempotent. Explicit resolution can restore a superseded root,
reaffirm it now, and supersede its current contradictions; subsequent ingestion
still follows the normal policy. All these operations preserve original rows.

These rules do not infer preferences from mere mentions, generalize repeated
events into new facts, resolve arbitrary prose, or recognize geographical/name
aliases. Questions, uncertain/reported statements, historical wording, and
ambiguous compound assertions do not automatically replace supported facts.
Use `--split` for supported independent clauses. Unrecognized content remains
stored without an inferred conflict. The JSONL store still needs production
concurrency controls and physical cleanup.

See [behavior and verification](reports/memory_reconciliation.md).

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
- conflict resolution, duplicate summaries, provenance, history visibility, and bulk reconciliation

## Recommended Next Implementation Steps

Recommended order:

1. Add scheduled cleanup and JSONL compaction.
2. Add REST API endpoints and production storage.
3. Improve free-form entity/relation extraction and measure hosted answer quality.
4. Extend external retrieval/retention evaluation with independent human conflict and personalization judgments.
5. Add monitoring, deletion/export, and privacy controls.
6. Scale vector/graph storage beyond local workloads as needed.

ML research is tracked separately in **ML Importance Model** above.

## Current Milestone

The current milestone is:

```text
Structured Memory Core, ranked keyword retrieval, optional trained XGBoost scoring, local JSONL store, and CLI implemented.
```

In other words, we have implemented the memory representation, decision pipeline, local JSONL storage, and CLI access. Semantic/hybrid retrieval, cited RAG, temporal graph storage, and external evaluation are now implemented. Production concurrent storage, APIs, deployment, broader extraction quality, and independent human answer/conflict/personalization validation remain. ML research is separate.
