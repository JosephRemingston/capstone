# Retrieval, Gemini RAG, and temporal graph

The JSONL revision log remains authoritative. `MemoryRAG` maintains rebuildable
SQLite vector and graph indexes next to it (`memories.index.sqlite3`). Indexing
and queries require a user ID. This is a local, single-writer design; it does not
add authentication or concurrent JSONL writer guarantees.

## Setup

```bash
uv pip install --python .venv/bin/python -r requirements-rag.txt
# A local .env was created with blank credentials. Fill it locally:
# GOOGLE_API_KEY=your_key
# COGNIMEM_GEMINI_MODEL=gemini-2.5-flash
```

`.env` is ignored by Git. `.env.example` documents configuration. Existing process
environment variables take precedence over the file; `GEMINI_API_KEY` is accepted
as a fallback. `ask --env-file PATH` selects a different file. No key is bundled.
The installed LangChain adapter and structured schema were checked without a
network request. Live Gemini generation has not been tested because the user
will configure credentials later.

Embeddings run locally with FastEmbed's `BAAI/bge-small-en-v1.5` (384 dimensions).
The first semantic query downloads the public embedding model into
`data/rag/models`; later queries reuse it. Memory text is not sent to an embedding
service. Gemini receives the question and selected evidence when `ask` runs with
its default `--generator gemini`. `--preview` and `--generator extractive` do not
call Gemini.

## CLI

```bash
.venv/bin/python -m main process 'Alice reports to Bob' --user-id demo --session-id s1
.venv/bin/python -m main process 'Bob works on Atlas' --user-id demo --session-id s1
.venv/bin/python -m main process 'Atlas is owned by Acme' --user-id demo --session-id s1
.venv/bin/python -m main index --user-id demo
.venv/bin/python -m main search 'Alice project' --user-id demo --mode hybrid --pretty
.venv/bin/python -m main graph --user-id demo --query Alice --hops 3 --pretty
.venv/bin/python -m main ask 'Which project does Alice’s manager work on?' --user-id demo --preview --pretty
.venv/bin/python -m main ask 'Which project does Alice’s manager work on?' --user-id demo --pretty
.venv/bin/python -m main ask 'Alice' --user-id demo --mode graph --generator extractive --pretty
.venv/bin/python -m main graph --user-id demo --as-of 2026-09-01T12:00:00Z --known-at 2026-09-10T12:00:00Z
```

The existing keyword search CLI and return format are unchanged. Other search
modes return hits with source record, chunk text, fusion score, channel ranks,
and graph evidence paths. Modes are `keyword`, `semantic`, `graph`, and `hybrid`;
`recency` is also available through the Python API for evaluation.
`index --graph-only` works with standard-library Python.

## Retrieval and citations

- Chunk documents at word boundaries into at most 1,000 characters with 150
  characters overlap. Store offsets, document hashes, model identity, normalized
  float32 vectors, and source IDs in SQLite. Changed chunks are re-embedded;
  removed/ineligible current records are evicted when the current index syncs.
- Exact cosine search scans the selected user's vectors. This is persistent
  vector storage, **not an approximate nearest-neighbor server**. Dense candidates
  need cosine similarity of at least 0.35. Large deployments need an ANN backend.
- Hybrid search combines ranked keyword, semantic, and graph candidates with
  weighted reciprocal rank fusion: `weight / (60 + rank)`, weights 0.3/0.5/0.2.
  A fusion score is not a probability. Keyword ranking retains the existing
  relevance/category/tier/recency/importance signals. No learned reranker is used.
- Visibility and user/session/category/tier filters apply before ranking. Index
  results are checked against allowed source IDs. Historical queries do not mix
  current vectors with old snapshots: they synchronize the requested snapshot.
- Context includes dated, identified source chunks. Its default limit is 10,000
  **characters**, not tokens. Whole chunks that cannot fit are skipped. A graph
  path is included only if every source supporting it fits. No transitive fact
  is asserted merely because nodes are connected.
- LangChain calls `gemini-2.5-flash` with structured output, temperature 0, a
  4,096-output-token cap, timeout, and one retry. Every generated statement requires citations and exact
  quotes from retrieved chunks. Unknown source IDs, absent quotes, malformed
  output, and uncited claims are rejected. This verifies provenance, **not that
  a quote logically entails the generated claim**. Human/semantic answer
  evaluation remains necessary. Prompt instructions treat memories as untrusted
  data; prompting alone cannot guarantee immunity to prompt injection.
- With no evidence, return an explicit abstention without an API call. Provider
  failure is an error rather than a silent switch to an extractive answer.
  Extractive mode returns cited snippets and is labeled as such.

## Temporal graph

`TemporalGraph` implements `GraphMemoryAdapter`. SQLite stores typed nodes,
aliases, and directed, signed relationships. Every edge carries a source memory,
confidence, evidence IDs, an exact source-revision fingerprint, and two half-open time intervals `[start, end)`:

- `valid_from` / `valid_to`: when the accepted assertion holds.
- `known_from` / `known_to`: when a persisted revision represents the system's
  knowledge. New JSONL writes stamp `recorded_at`; old logs approximate it using
  `updated_at`, and the first graph rebuild reports the fallback count.

A fact observed on January 5 but recorded January 8 is absent from a January 6
knowledge snapshot. Once recorded, a valid-time query can place it on January 5.
Superseding and explicitly restoring facts preserve earlier intervals. Graph
state is rebuilt transactionally from revisions when their fingerprint changes;
unchanged logs reuse the index. Task nodes include owner, status, and due date
relationships, with revision history.

Node types: person, organization, project, location, task, concept, literal.
Automatic relationships include residence, employer, occupation, names,
preferences, allergies, manager, project membership, organizational membership,
organization location, project ownership, dependencies, and acquaintances.
Automatic extraction is deliberately limited to supported complete assertions,
e.g. `Alice reports to Bob`, `Bob works on Atlas`, `Atlas is owned by Acme`.
It does **not** resolve arbitrary pronouns, extract every fact from paragraphs,
or infer relations from embedding similarity. On LoCoMo's long natural turns,
this conservative extractor produced no edges. The graph storage/traversal is
functional; broad language extraction is an outstanding quality limitation.

Typed identities are scoped per user. Explicit stable IDs prevent two people
with the same label from silently merging. Ambiguous aliases return all matching
entities. Alias registration is time-aware; future aliases do not appear in past
knowledge queries. Positive links support directed/undirected traversals up to
four hops, capped at 1,000 paths. Negative assertions are available as direct
retrieval evidence but are not traversed as positive relationships.

## Explicit entity/relationship ingestion

For language outside the automatic grammar, callers can supply structured
relationships through the existing `--metadata`/`MemoryInput.metadata` API:

```json
{
  "entities": [
    {"id": "person:alice", "type": "person", "label": "Alice", "aliases": ["Al"]},
    {"id": "project:atlas", "type": "project", "label": "Atlas"}
  ],
  "relations": [
    {"subject": "person:alice", "predicate": "leads", "object": "project:atlas",
     "positive": true, "valid_from": "2026-09-01T00:00:00Z", "valid_to": null}
  ]
}
```

`self` is a reserved subject/object reference for the user. Relation endpoints
must be defined in that message or use `self`. Predicate names are snake_case;
dates need timezones and increasing intervals. Malformed metadata is rejected
before persistence. Entity IDs are stable within a user/type namespace. The
caller is responsible for the truth of supplied relationships. Custom predicates
can be many-valued; they are not automatically treated as exclusive or reconciled
against natural-language claims. Supply validity intervals for their changes.

Current RAG applies ordinary expiry/task/history visibility. Historical RAG
(`ask --as-of/--known-at`, or `MemoryRAG.search`) restricts candidates to graph
assertions with explicit validity and the revision known at that cutoff; it does
not guess historical validity for unstructured paragraphs. The `graph` command
is a knowledge-history inspection tool and can show retained graph assertions
whose underlying memory is expired for ordinary RAG.

## Evaluation

See [the report](../reports/retrieval_graph_report.md) and
[evaluation instructions](../evaluation/README.md). LoCoMo measures externally
annotated evidence retrieval with fixed baseline comparisons. LongMemEval checks
labeled evidence retention in knowledge-update/preference examples. These are
not claims of correct generated answers. Independent human judgments of conflict
policy and recommendation personalization remain outstanding. ML importance
training is a separate research track and is not required by this layer.

Primary documentation: [FastEmbed retrieval](https://qdrant.github.io/fastembed/qdrant/Retrieval_with_FastEmbed/),
[LangChain Gemini integration](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai),
[Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash).
