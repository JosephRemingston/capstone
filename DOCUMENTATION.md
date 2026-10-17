# CogniMem — Complete Documentation

## Purpose

CogniMem is a local-first memory system for personalized AI agents. It turns raw
conversation messages into structured memories, decides how useful they are,
keeps revision history, retrieves relevant evidence later, and can generate
cited answers from that evidence.

It is designed to answer questions a plain transcript cannot answer well:
which messages should persist, which fact is current after an update, whether a
task remains open, which evidence supports an answer, and what the system knew
at a past time. It is a Python memory-core implementation with a local authenticated HTTP/demo surface; production database migration and deployment remain separate concerns.

## What it builds

CogniMem provides these connected capabilities:

| Capability | What it does |
| --- | --- |
| Memory core | Converts a message into a normalized, serializable memory record. |
| Classification | Labels content as temporary, semantic, preference, episodic, procedural, or task. |
| Importance and lifecycle | Scores usefulness and assigns working, short-term, long-term, or archive tiers. |
| Task memory | Tracks creation, completion, cancellation, rescheduling, deadlines, and recurrence. |
| Reconciliation | Detects supported contradictory claims and preserves current and historical evidence. |
| Consolidation | Reduces exact/compatible repeated memories while retaining provenance. |
| Local store | Uses append-only JSONL as the authoritative memory history. |
| Retrieval | Supports keyword, semantic, graph, and hybrid retrieval. |
| RAG | Builds bounded, cited evidence context; can answer extractively or through Gemini. |
| Knowledge graph | Stores typed, time-aware relationships with source evidence. |
| Cleanup | Safely removes eligible expired memories and derived index data. |
| Trained importance model | Supplies 25% of the default importance score (csv5k ensemble); the rule-based heuristic supplies the other 75%. |

Not currently provided: production database migration, automated scheduler installation, general-purpose relation extraction, or independently human-rated answer quality.

## Architecture

```mermaid
flowchart TD
  A[CLI or Python caller] --> B[MemoryCore]
  B --> C[Classifier]
  B --> D[Feature Extractor]
  D --> E[Importance Scorer]
  E --> F[Lifecycle Manager]
  F --> G[MemoryRecord]
  G --> H[LocalMemoryStore]
  H --> I[Authoritative JSONL revisions]
  H --> J[Tasks, claims, reconciliation]
  I --> K[MemoryRAG sync]
  K --> L[SQLite vector index]
  K --> M[Temporal graph]
  L --> N[Keyword / semantic retrieval]
  M --> N
  N --> O[Bounded cited context]
  O --> P[Extractive answer or Gemini]
  H --> Q[Cleanup and compaction]
  Q --> I
  Q --> L
  Q --> M
```

The key invariant is that JSONL revisions are authoritative. SQLite vectors and
graph data are rebuildable derived indexes. An index failure must never be the
only loss of a memory.

## Source layout

```text
main/
  __main__.py              CLI
  application/             MemoryCore and maintenance worker
  domain/                  models, rules, lifecycle, tasks, reconciliation
  storage/                 JSONL store, SQLite indexes, cleanup, locks
  retrieval/               ranker, embeddings, RAG, generation
  graph/                   temporal graph and graph contract
  ensemble.py              default hybrid importance scorer (trained model + heuristic)
  ml.py                    importance feature schema shared with the model
training/                  importance-model training (training/ensemble/)
artifacts/                 trained importance model (importance_ensemble/)
tests/                     unit and integration tests
docs/                      focused guides and reports
data/                      datasets, caches, and local artifacts
```

Use new module paths directly: `main.application.core`,
`main.domain.models`, `main.storage.store`, `main.retrieval.rag`, and
`main.graph.temporal`. Old flat compatibility source modules were removed.

## Install and configure

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

For Gemini, create a repository-root `.env`:

```dotenv
GOOGLE_API_KEY=your_key_here
COGNIMEM_GEMINI_MODEL=gemini-2.5-flash
```

`GEMINI_API_KEY` is accepted as a fallback. Environment variables take
precedence over `.env`. The key is ignored by Git. Local FastEmbed embeddings
do not send memory text to an embedding provider; Gemini receives only the
question and selected context when hosted generation is intentionally used.

The default importance scorer also needs the trained model in
`artifacts/importance_ensemble/` (tracked in Git) and the BGE encoder cached in
`data/rag/models`. The scorer never downloads the encoder, in the CLI, Python, or
the API; fetch it once with
`.venv/bin/python -c "from main.retrieval.embeddings import FastEmbedder; FastEmbedder()"`
from the repository root, or set `COGNIMEM_SCORER=heuristic` to run without the model.
The API builds one local-only BGE encoder shared by retrieval and the scorer, so a
missing encoder cache stops it at start-up with that command and the opt-out; with
`COGNIMEM_SCORER=heuristic` the API starts without BGE as before.

Runtime locations:

- JSONL store: `memory_store/memories.jsonl`
- SQLite index: beside the store, normally `memories.index.sqlite3`
- embedding cache (retrieval and importance scorer): `data/rag/models`
- importance model: `artifacts/importance_ensemble/`
- reports: `docs/reports/`
- local configuration: `.env`

## Basic use

```bash
python3 -m main process "I prefer concise Python explanations" \
  --user-id joseph --session-id s1 --pretty

python3 -m main list --user-id joseph --pretty
python3 -m main search "Python explanation style" --user-id joseph --pretty
python3 -m main get MEMORY_ID --pretty
python3 -m main history MEMORY_ID --user-id joseph --pretty
python3 -m main stats --pretty
```

Use `--no-save` with `process` to inspect a result without writing JSONL.
Normal reads hide expired, superseded, consolidated, completed, and cancelled
records when appropriate. Use `--include-expired`, `--include-resolved`, or
`--include-history` for deliberate inspection.

## Real-world use and day-to-day integration

CogniMem is most useful when an assistant, internal tool, or workflow repeatedly
works with the same person, project, or team. It should run beside the agent
that receives messages; it is not meant to replace a company system of record.
For each integration, choose a stable `user_id` and a session ID for each chat,
meeting, or work thread.

| Work setting | Memories to retain | How CogniMem helps day to day |
| --- | --- | --- |
| Personal work assistant | Preferred writing style, active projects, reminders, constraints. | Starts each request with relevant preferences and open tasks rather than asking again. |
| Engineering copilot | Repository conventions, deployment steps, incidents, owners, project dependencies. | Retrieves prior decisions and shows graph-backed evidence for “who owns this?” questions. |
| Meeting assistant | Decisions, actions, deadlines, attendees, and later corrections. | Turns follow-up messages into task revisions and preserves the original decision trail. |
| Customer-success assistant | Customer preferences, stated goals, product constraints, and commitments. | Keeps customer-specific context isolated by `user_id` and supports cited answers to account questions. |
| Research assistant | Research interests, claims, sources, projects, and relationships. | Uses hybrid retrieval for prior notes and graph paths for supported people/project relationships. |
| Personal productivity tool | Recurring routines, single reminders, completed work, and changing deadlines. | Maintains active task state instead of repeatedly resurfacing completed work. |

### Recommended integration loop

For every incoming message, create a `MemoryInput`, process it, and ingest it.
Before generating an assistant response, retrieve a small evidence set for the
current question. Use extractive context during development and Gemini only when
the answer requires synthesis.

```python
from pathlib import Path

from main.application.core import MemoryCore
from main.domain.models import MemoryInput
from main.storage.store import LocalMemoryStore
from main.retrieval.rag import MemoryRAG

store = LocalMemoryStore(Path("memory_store/memories.jsonl"))
core = MemoryCore()

incoming = MemoryInput(
    content="I prefer brief weekly project updates.",
    user_id="user-42",
    session_id="chat-2026-09-28",
    role="user",
)
record = core.process(incoming)
store.ingest(record)

rag = MemoryRAG(store)
context = rag.context(
    "How should I format this week's project update?",
    user_id="user-42",
    mode="hybrid",
)
# Give context to your existing application prompt, or call the built-in answer generator.
```

The host application remains responsible for authentication, user consent,
session creation, displaying citations, and deciding which messages are allowed
to enter memory. A useful production pattern is to store only user messages and
explicitly approved assistant decisions, then present retrieved evidence to the
user when an answer relies on remembered information.

### Practical workflows

**Daily planning:** ingest “remind me,” “I finished,” and “move the deadline”
messages as they occur. At the start of the day, query open task records and use
`history` when a task status needs explanation.

**Project continuity:** ingest stable facts such as project ownership, technology
choices, and review preferences. Use hybrid retrieval before drafting a status
update; use graph metadata for stable relationships that need multi-hop queries.

**Meeting follow-up:** process the meeting notes in separate, clear statements.
Use `--split` only for independent clauses. Record a task for each action and
send the resulting record IDs to the workflow that owns reminders.

**Memory correction:** when a user changes a durable fact, ingest the new clear
statement. Reconciliation preserves the old evidence and identifies the current
one; use explicit `resolve` only when automatic evidence ordering is insufficient.

**Retention hygiene:** run cleanup as a preview first, review the JSON response,
then schedule the maintenance worker externally after retention policy is agreed.
Do not use automatic cleanup as a substitute for user consent or backup policy.

## Data model

### Categories

| Category | Meaning | Example |
| --- | --- | --- |
| temporary | Transient chat. | “Thanks!” |
| semantic | Durable fact. | “I live in Chennai.” |
| preference | Standing choice or constraint. | “Never include peanuts.” |
| episodic | Event or change. | “Yesterday I met my guide.” |
| procedural | Repeatable workflow. | “First test, then deploy.” |
| task | Action/reminder. | “Remind me to call Sam.” |

### Tiers

| Tier | Meaning |
| --- | --- |
| working | Brief or low-value memory; normally expires after one hour. |
| short_term | Limited-period memory; normally expires after fourteen days. |
| long_term | Durable, important memory. |
| archive | Inactive historical memory, normally omitted from active retrieval. |

### Input and record

`MemoryInput` requires nonempty `content`, `user_id`, and `session_id`.
It accepts `role` and structured `metadata`.

`MemoryRecord` adds an ID, category, importance score, tier, timestamps,
expiry/archive hints, access count, source metadata, task state, claim details,
conflict/consolidation state, confidence, and evidence IDs. Times are
timezone-aware ISO values. A record is serializable to JSON.

## Processing pipeline

`MemoryCore.process()` performs these steps:

1. Validate and normalize the input.
2. Classify the content.
3. Create a provisional record and timestamps.
4. Extract numeric importance features.
5. Score importance from 0 to 1 (by default 25% trained model, 75% heuristic).
6. Assign lifecycle tier, expiry, and archive hints.
7. Parse supported task, deadline, recurrence, claim, and graph metadata.
8. Return the record; `LocalMemoryStore.ingest()` persists it and applies
   task/reconciliation revisions when required.

`process_many()` can conservatively split independent clauses. Ambiguous mixed
messages are intentionally not allowed to mutate several tasks silently.

## Conversational rules

The default classifier is deterministic and English-specific. Priority matters:
a completion or cancellation is an episodic update before it can be mistaken for
a new task.

- Exact greetings such as hello, thanks, and okay are temporary.
- Supported task completion, cancellation, and rescheduling are episodic updates.
- Explicit reminder/action requests are tasks.
- Supported claims, user facts, and constraints become semantic or preference.
- “Never include peanuts” is a preference constraint.
- Events are episodic; workflows use phrases such as how to, first, then, and
  procedure.
- Uncertain, quoted, conditional, or mixed statements are handled conservatively.

This is explainable and testable, but does not fully understand sarcasm,
pronouns, arbitrary negation, indirect preferences, or long mixed prose.

## Importance and lifecycle

The default scorer is a hybrid (`HybridImportanceScorer` in `main/ensemble.py`):
`score = round(0.25 × model_score + 0.75 × heuristic_score, 4)`.

The heuristic part (75%) is the rule-based `ImportanceScorer` in
`main/domain/scoring.py`, a transparent weighted sum clamped to 0–1. Semantic,
preference, and task category signals each weigh 0.38; procedural weighs 0.34;
episodic weighs 0.20. It also uses task/preference language, deadline, access
frequency, entity density, interaction metadata, recency, word count, and
sentiment. Temporary memories receive a 0.25 multiplier.

The model part (25%) is the value of the trained model's predicted class: low
0.3102, mid 0.5792, high 0.8372 (the mean csv5k importance score of the fit and
validation rows in that class; 3,992 rows, the test split excluded). For temporary records it is multiplied by the same 0.25
small-talk penalty. The record's category still comes from the normal classifier;
the model computes its own inputs from the text with the legacy feature path
(`MemoryClassifier(legacy=True)` plus `MLFeatureExtractor` in `main/ml.py`).

Every record stores provenance in `source_metadata["importance_model"]`. For the
hybrid it holds `type`, `model_weight`, `model_class`, `model_probabilities`,
`model_score`, `small_talk_penalty`, `heuristic_score`, `model_sha256`, and
`encoder`; for the heuristic it is `{"type": "heuristic"}`.

Lifecycle policy:

- score below 0.25 or temporary category: working;
- normal useful memory: short-term;
- semantic/preference/procedural/recurring task at score >= 0.42: long-term;
- completed/cancelled task or eligible stale useful memory: archive;
- one-off tasks remain short-term even if highly important;
- task deadlines expire after deadline plus 24 hours;
- recurring tasks have no automatic expiry;
- long-term archive view begins after 90 inactive days when eligible.

These thresholds did not change with the hybrid default; balanced cleanup still
archives rather than deletes expired records scoring >= 0.65 (`ARCHIVE_IMPORTANCE`
in `main/storage/cleanup.py`). They are product-policy thresholds, not
statistically calibrated probabilities. Measured on 16 October 2026 (snapshots
`hybrid` and `heuristic_control`), the hybrid matched the heuristic on LongMemEval
user-fact retention (90.55%), classification category and tier accuracy, and
lifecycle checks (16/16), and its model score does not separate answer-bearing
turns; see
[Importance scoring (default)](README.md#importance-scoring-default).

### Trained model (25% of the default score)

The model is the csv5k "Ensemble + thresholds" bundle in
`artifacts/importance_ensemble/` (`xgboost.ubj`, `deep_seed_42/43/44.onnx`, and
`metadata.json` with SHA-256 checksums and golden vectors). At load,
`EnsembleModel` checks the checksums, schema, boosting rounds, ONNX interfaces,
temperatures, weights, and encoder id; the golden vectors are checked by the test
suite (`tests/test_hybrid_importance.py`), not at load time. A fit-only XGBoost model on 525 features (13 text features + 512
hashed word counts; max depth 3, min child weight 3, 304 boosting rounds) is
temperature-calibrated and blended 0.5/0.5 with the average of three seeds (42,
43, 44) of a deep head on the frozen `BAAI/bge-small-en-v1.5` embedding (384
values) plus the 13 numeric features (text branch 384→192→128, numeric 13→64→32,
joint 160→128→64→3, LayerNorm/GELU/dropout 0.2). Each seed is
temperature-calibrated and the average is calibrated again; class weights
1 / 0.7 / 1 then pick low, mid, or high. Serving uses xgboost, ONNX Runtime, and
the cached local BGE encoder (the API builds one local-only encoder shared with
retrieval); PyTorch is needed only for training. Cost is roughly 14 ms per user
message and about 100 ms per long assistant message, mostly the BGE embedding. A
missing model raises a clear error naming the training command and the opt-out; a
missing encoder cache, one naming the download command and the opt-out.

Training data is csv5k (`cognimem_memory_importance_5000.csv`): 5,000
AI-generated memory sentences (443 distinct after removing a " (example N)"
suffix) with synthetic importance scores from 0.20 to 1.00, binned as
low < 0.4 <= mid < 0.7 <= high. The split is grouped by sentence with seed 42:
3,200 / 792 / 1,008 rows (286 / 71 / 86 sentences). Temperatures, blend weight,
and class weights were chosen on validation only. On the test split, macro F1 is
0.8912 (95% sentence-bootstrap interval 0.809–0.949) and accuracy 89.09%; the
always-most-common-class baseline has 0.2529 macro F1 and the best possible
(oracle) 0.9230. The [ensemble report](evaluation/reports/retention_ensemble.md)
has the full results.

Retrain with the package `training/ensemble/`. The first run on a fresh clone needs
the CSV: `python -m training.ensemble.run --csv /path/to/cognimem_memory_importance_5000.csv`
copies it to the Git-ignored `data/experiments/csv5k.csv`, and later runs need no
flag (the bare command fails on a fresh clone). Working files go to the Git-ignored
`artifacts/ensemble_training/`. Training reproduces the model files (`xgboost.ubj`
and the three ONNX files, whose SHA-256 values are listed in the metadata `files`)
bit for bit; `metadata.json` carries a `created_at` timestamp, so its hash (shown as
`model_sha256` in provenance and `/health/ready`) changes on every retrain.

Select the scorer with `COGNIMEM_SCORER=hybrid` (default) or `heuristic`. CLI:
`python -m main process ... --scorer hybrid|heuristic` (`--model-dir` only with
`hybrid`). Python: `MemoryCore()` uses the default and
`MemoryCore.with_heuristic()` opts out. API: `Settings.scorer` reads
`COGNIMEM_SCORER`; the scorer is built at start-up and `/health/ready` reports it.

Limits: csv5k is synthetic (AI-generated sentences and scores, not human
ratings), and the test split has only 86 distinct sentences. Conversation turns
are far longer than the csv5k sentences (at most 13 words; LongMemEval
user turns median 42 words), so the model runs out of distribution. Its 25% weight
is a fixed, user-chosen setting that was not tuned or calibrated; this evidence is a
reason to keep it small, not where it came from. Before the small-talk penalty it rates greetings
mid or high. Its low/mid/high importance target differs from the
invalid/short-term/long-term target of the removed XGBoost-only and neural (MSC)
scorers, whose code, models, and reports are gone; dated snapshots
`evaluation/reports/<tag>_summary.json` keep their old numbers as history.

## Task management

Tasks are revisioned memories rather than mutable rows. Supported behavior:

- create tasks/reminders;
- parse explicit timezone-aware `--due-at` dates;
- complete, cancel, or reschedule supported task language;
- match task references with normalized action nouns;
- require explicit `--task-id` if a reference is ambiguous;
- record status changes as history;
- support recurring occurrences and occurrence-versus-series updates.

```bash
python3 -m main process "Remind me to submit the report tomorrow" \
  --user-id joseph --session-id s1
python3 -m main process "I submitted the report" \
  --user-id joseph --session-id s1
python3 -m main process "Reschedule the report submission to Friday" \
  --user-id joseph --session-id s1
```

## Claims, conflicts, and consolidation

Claim extraction intentionally supports a narrow set of facts and preferences:
subject, predicate, normalized value, polarity, exclusivity, and kind. Supported
examples include residence, employer, name, occupation, selected preferences,
and allergies.

A contradictory current claim with the same exclusive subject/predicate can be
resolved using source priority, confidence, observation time, then importance.
The losing fact remains historical evidence. For example, Bengaluru can replace
Chennai as a current residence without destroying the older observation.

```bash
python3 -m main resolve --keep MEMORY_ID --user-id joseph --pretty
python3 -m main reconcile --user-id joseph --pretty
```

Consolidation merges exact/compatible repetitions into a summary with evidence
IDs. It reduces duplicate retrieval but does not invent unsupported abstractions
or reclaim physical storage by itself.

## Storage, locks, and cleanup

`LocalMemoryStore` appends JSON objects to a JSONL revision log. Reads
materialize the latest relevant revision per memory ID. Participating ingestion,
reads, RAG synchronization, reconciliation, and cleanup use reentrant
process/file locks on macOS/Linux.

Cleanup physically removes eligible JSONL revisions and registered user vector
and graph data. Preview is the default:

```bash
python3 -m main cleanup --user-id joseph --pretty
python3 -m main cleanup --user-id joseph --apply --pretty
python3 -m main.application.maintenance \
  --store /absolute/path/memories.jsonl --user-id joseph --once
```

A memory is removable only when its latest revision is expired beyond the grace
period (seven days by default), belongs to the requested user, has no
`legal_hold`/ `retain` protection, is not an open task, and is not needed as
transitive evidence by retained records. Cleanup validates the full log, purges
derived indexes, writes a fsynced replacement in the same directory, atomically
renames it, and then records schedule progress. It removes active application
files, not backups, SSD snapshots, or forensic remnants.

Compaction removes only byte-equivalent repeated snapshots. Meaningful history
such as A → B → A and graph revision fingerprints remain intact.

## Retrieval and RAG

Keyword retrieval ranks whole-word matches with relevance, category, tier,
recency, and importance signals. Visibility/user/session/category/tier filters
apply before ranking.

Semantic retrieval uses local FastEmbed `BAAI/bge-small-en-v1.5` vectors
(384 dimensions). Documents are chunked at word boundaries into <=1,000
characters with 150-character overlap. SQLite stores normalized float32 vectors;
exact cosine candidates need similarity >=0.35. This is local persistent search,
not an approximate-nearest-neighbor service.

Hybrid mode combines keyword, semantic, and graph ranks using weighted reciprocal
rank fusion: keyword 0.3, semantic 0.5, graph 0.2, each divided by `60 + rank`.
The fusion score is an ordering signal, not a probability. The highest fused
candidates are reranked locally by `Xenova/ms-marco-MiniLM-L-6-v2`; the best of
up to three representative chunks supplies each memory's score and answer
snippet. The cross-encoder is enabled by default for hybrid mode, cached under
`data/rag/models`, and falls back to fusion ordering if it cannot run.

```bash
python3 -m main index --user-id joseph
python3 -m main search "project information" --user-id joseph --mode hybrid
python3 -m main search "project information" --user-id joseph --mode hybrid --no-rerank
python3 -m main ask "What project does Alice work on?" \
  --user-id joseph --generator extractive --pretty
python3 -m main ask "What project does Alice work on?" \
  --user-id joseph --preview --pretty
```

Context is capped at 10,000 characters. Whole chunks that do not fit are skipped.
No evidence yields an explicit abstention without a Gemini call.

Gemini uses LangChain with Gemini 2.5 Flash, temperature 0, a 4,096-token output
cap, 45-second timeout, and one retry. Each generated statement needs source IDs
and exact quotes from retrieved chunks. CogniMem rejects malformed output,
unknown sources, missing citations, and nonmatching quotes. This checks
provenance, not logical entailment or complete prompt-injection safety.
Extractive mode returns cited snippets without a hosted call.

## Temporal graph

The SQLite temporal graph stores typed nodes, aliases, signed directed edges,
source IDs, confidence, evidence IDs, source-revision fingerprints, and two
half-open intervals:

| Time | Meaning |
| --- | --- |
| valid time | When an assertion is true in the represented world. |
| known time | When the revision made it known to CogniMem. |

Use `--as-of` for valid time and `--known-at` for knowledge time. A fact
observed January 5 but recorded January 8 can be valid January 5 but must not
appear in a January 6 knowledge snapshot.

Node types: person, organization, project, location, task, concept, literal.
Automatic relationships include residence, employment, manager, project work,
ownership, membership, dependency, names, occupation, tastes, allergies, and
task status/due dates. Extraction requires complete supported assertions such as
“Alice reports to Bob.” It does not infer arbitrary relationships from paragraphs
or embedding similarity.

Traversal follows positive edges up to four hops and returns source evidence.
Negative facts can be direct evidence but are not traversed positively.

Structured metadata supports explicit relationships outside the narrow grammar:

```json
{
  "entities": [{"id": "person:alice", "type": "person", "label": "Alice"}],
  "relations": [{"subject": "person:alice", "predicate": "leads",
    "object": "self", "positive": true,
    "valid_from": "2026-09-01T00:00:00Z"}]
}
```

## Evaluation, limits, and results

The evaluation package was rebuilt on 12 October 2026 together with a fresh benchmark; the
[README](README.md#how-to-read-the-results) has the baseline, final and current metrics layer by layer and the
[evaluation guide](evaluation/README.md) the protocol. Headline numbers from the `final`
snapshot (`evaluation/reports/final_summary.json`, measured with the heuristic importance
scorer, the default before the hybrid; for the hybrid, measured on 16 October 2026 in
the `hybrid` and `heuristic_control` snapshots, see
[Importance scoring (default)](README.md#importance-scoring-default)):

| Evaluation | Result and interpretation |
| --- | --- |
| Controlled benchmark, test split (270 questions, AI-authored) | Hybrid + reranker Recall@5 97.5%, MRR 0.939, correct memory first 84.2% (baseline 93.0%, 0.904, 77.3%). |
| LoCoMo evidence retrieval (1,973 questions) | Hybrid + reranker Recall@5 66.8%, Hit@5 72.3% (baseline 55.8%, 61.1%); 66.8% on the 7 conversations not used to choose the pool size. |
| LongMemEval retention (457 examples) | 90.6% of answer turns still retrievable (baseline 21.0%). |
| Opt-in extractive abstention | Benchmark test: 17.4% of unanswerable questions refused, 99.6% of answerable kept; no useful gain on LongMemEval or LoCoMo. |
| Lifecycle checks | 16/16. |

These are evidence-retrieval and retention results with an extractive answerer, not
generated-answer accuracy. Run every measurement with
`.venv/bin/python scripts/measure_all.py --tag <name> --candidate-limit default --ann-arms`.
`--scorer hybrid|heuristic` picks the importance scorer, and `--scorer-steps-only` runs only
the steps that use it (retention, classification, lifecycle, and the summary; ablation and
LoCoMo never call it). The classification and lifecycle runners pin the clock
(`evaluation/clock.py`), every scorer-dependent runner records which scorer ran,
`evaluation/report.py` adds the importance-model summary from
`evaluation/reports/retention_ensemble.json` to `retention_summary.json`, and
`scripts/report_delta.py` also compares cleanup counts (`longmemeval._cleanup.*`).

Independent human labels for conversational importance/tier/category, conflict
decisions, personalization, and live Gemini answer quality are still required.

## Testing

```bash
.venv/bin/python -m pytest -q
python3 -m compileall -q main training tests
```

`python -m unittest discover -t . -s tests -q` is an alternative; without `-t .`
discovery does not run `tests/__init__.py`.

Hybrid-scorer tests are skipped when the trained model in
`artifacts/importance_ensemble/` or the cached BGE encoder is unavailable, and
training tests when PyTorch or xgboost is not installed. Tests cover the core pipeline,
rules, lifecycle, tasks, revisions, ranking, conflicts, graph/RAG, local reranking,
cleanup, locks, and CLI behavior.

## Security and privacy

- Memory files and indexes are local by default.
- `user_id` is a primary retrieval/index/graph isolation boundary.
- Embeddings run locally.
- Hosted generation receives only selected evidence and the question.
- API keys remain in ignored environment files.
- Cleanup removes active application data but does not manage backups.
- Cooperative file locks are not a replacement for server authentication or
  production-grade concurrent storage.

## Development priorities

1. Build independently labeled, consented conversational evaluation data.
2. Improve evidence retention on unseen conversation benchmarks.
3. Extend graph extraction through constrained, reviewable schemas.
4. Add independent human review of Gemini answer quality and citations.
5. Add API, authentication, and production storage only after local policies and
   evaluation are stable.

Focused supporting material remains in `README.md`, `system_design.md`,
`docs/retrieval_graph.md`, `docs/cleanup.md`, and `docs/reports/`. This
file is the complete single-document reference for the system.

## 14. HTTP API and security layer

The HTTP surface is implemented in `api/` without changing the Memory Core's
ownership boundaries. `MemoryCore` still owns processing and privacy redaction;
`LocalMemoryStore` remains the authoritative JSONL observation log; SQLite remains
used for rebuildable indexes. The API layer supplies transport validation,
identity, authorization, privacy endpoints, and operational instrumentation.

### Endpoints

| Method | Endpoint | Purpose | Auth |
| --- | --- | --- | --- |
| GET | `/health/live` | Liveness probe | No |
| GET | `/health/ready` | Readiness checks for auth, memory, and index stores; reports the importance scorer | No |
| GET | `/metrics` | Prometheus exposition | No |
| POST | `/api/v1/auth/signup` | Create a user account | No |
| POST | `/api/v1/auth/login` | Issue a short-lived access token | No |
| GET | `/api/v1/me` | Return the authenticated account | Bearer token |
| POST | `/api/v1/memories` | Process and persist a memory | Bearer token |
| GET | `/api/v1/memories` | List the caller's memories | Bearer token |
| GET | `/api/v1/memories/{id}` | Read one caller-owned memory | Bearer token |
| POST | `/api/v1/memories/search` | Keyword, semantic, hybrid, graph, or recency retrieval | Bearer token |
| GET | `/api/v1/memories/{id}/history` | Every revision of one of the caller's memories, oldest first; 404 if missing or another user's | Bearer token |
| GET | `/api/v1/graph` | The caller's temporal relations (`as_of`, `known_at`, `predicate`); with `query`, matched entities and paths up to `hops` (1–3) | Bearer token |
| GET | `/api/v1/evaluation/snapshots` | Tracked measurement snapshots (`evaluation/reports/<tag>_summary.json`, override the folder with `COGNIMEM_REPORTS_DIR`), oldest first | Bearer token |
| GET | `/api/v1/evaluation/snapshots/{tag}` | One snapshot's numeric metrics and provenance; 404 for unknown tags | Bearer token |
| GET | `/api/v1/evaluation/model` | The importance model's test report | Bearer token |
| POST | `/api/v1/memories/ask` | Cited answer from the caller's memories; `generator` is `extractive` (default) or `gemini` (503 when unavailable); optional `abstain`, `as_of`, `known_at` | Bearer token |
| GET | `/api/v1/me/export?format=json\|csv` | Export account and memory data | Bearer token |
| DELETE | `/api/v1/me` | Delete the account and its persisted memory/index data | Bearer token |
| GET | `/demo/` | Lightweight browser demo | No |
| GET | `/app/` | React web app (served when `frontend/dist` is built; client-side routes fall back to `index.html`) | No (the app signs in through the API) |

All JSON API responses use a common envelope containing `success`, `data`,
`error`, and `request_id`. FastAPI/Pydantic validation failures use the same
error envelope rather than leaking framework internals.

### Authentication and authorization

Passwords are hashed with Argon2id through `argon2-cffi`; plaintext passwords,
password hashes, bearer tokens, and secrets are never returned by the API.
Access tokens are signed HS256 JWTs with `sub`, `role`, `iat`, `exp`, and `typ`
claims. The secret must contain at least 32 bytes and is supplied through the
environment. Protected resources use a role dependency (`user` or `admin`) and
always derive the memory owner from the authenticated token. A caller cannot
supply another user's `user_id` to read or search their data.

### Privacy operations

JSON export contains the non-secret account fields plus every persisted memory
revision owned by the caller. CSV export provides a tabular memory export. Account
deletion removes the caller's observations from the authoritative JSONL log and
purges their vector/graph/index rows before removing the identity record. No
password hash or authentication secret is included in exports.

### Observability

HTTP requests emit JSON structured logs with request ID, method, route, status,
and duration. Sensitive request headers and credential fields are excluded from
logs. Prometheus counters/histograms cover request traffic and latency, while
separate counters track authentication events and memory ingestion. Readiness
checks verify the auth database, authoritative memory store, and rebuildable
index database; the response also names the importance scorer built at start-up
(for the hybrid, the SHA-256 of `metadata.json`, which changes on every retrain). A
missing model stops the app at start-up. For the hybrid the API builds one
local-only BGE encoder (cache `data/rag/models`, no download) shared by retrieval
and the scorer, so a missing encoder cache also stops it at start-up with the
download command and the `COGNIMEM_SCORER=heuristic` opt-out; with
`COGNIMEM_SCORER=heuristic` the API starts without BGE as before.

### Web app

`frontend/` is a React + TypeScript client (Vite, Tailwind, shadcn/ui, TanStack Query) over the API above; it holds
no business logic. `pnpm build` writes `frontend/dist`, which the API serves at `/app` (`COGNIMEM_WEB_DIR` overrides
the folder; without a build `/app` is simply absent). Screens: Chat, Memories, Memory detail, Search & Ask, Graph and
System, plus Evaluation, which renders the README's per-layer baseline → final → current tables, the importance model report and a snapshot comparison from the tracked reports. The session token lives in `sessionStorage`; a 401 or the token's expiry signs the user out, because the API
has no refresh endpoint. Answers are extractive by default, and switching to Gemini asks for confirmation first.
`pnpm test` runs the Vitest unit tests and `pnpm e2e` a Playwright smoke test against a fresh API; see
[`frontend/README.md`](frontend/README.md).

### Approximate-nearest-neighbor retrieval

`ApproximateVectorIndex` adds deterministic random-hyperplane locality-sensitive
hashing (multi-table, multi-probe) on top of the existing normalized cosine
vectors. It is disabled by default: exact search is the shipped path. When
enabled and the user's corpus reaches the configured threshold (default 5000),
semantic retrieval uses LSH buckets to produce a candidate set, widening the
Hamming radius until it has enough candidates and otherwise falling back to the
exact scan. The authoritative
vector table is unchanged, so the ANN index is rebuildable and does not replace
the existing source/index separation. The threshold is configured by
`COGNIMEM_ANN_EXACT_THRESHOLD` and ANN is enabled with
`COGNIMEM_ANN_ENABLED=true` (CLI: `python -m main --ann ...`). ANN is
experimental: on real conversational BGE vectors at about 5.9k chunks its top-20
agreed with exact search only about 41% of the time.

### Scope boundary

This implementation intentionally does **not** implement the other remaining
capstone items: production database migration, deployment setup, background
reminder delivery, broader entity/coreference/relationship extraction, or the
human feedback loop. Those remain available for the other contributor.
