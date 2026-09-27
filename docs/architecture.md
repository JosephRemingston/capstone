# Repository Architecture

## Entry points

- `python -m main` — CogniMem command-line interface.
- `python -m main.maintenance` — scheduled cleanup worker.
- `python -m training.train_importance` — optional importance-model training.
- `python -m evaluation.{evaluate,benchmark,longmemeval,conflicts,personalization,suite}` — evaluation commands.

## Runtime flow

```text
CLI
  -> application.MemoryCore
  -> domain classification / features / scoring / lifecycle
  -> storage.LocalMemoryStore (JSONL revision log)
  -> reconciliation and task updates

MemoryRAG
  -> storage SQLite indexes
  -> temporal graph + local embeddings
  -> bounded context
  -> Gemini or extractive answer

maintenance worker
  -> storage cleanup / compaction
  -> SQLite index invalidation
```

## Package layout

```text
main/
  application/  orchestration and maintenance-worker implementation
  domain/       memory models, classification, lifecycle, tasks, rules, reconciliation
  storage/      JSONL store, SQLite indexes, cleanup, locks
  retrieval/    ranker, embeddings, RAG, answer generation
  graph/        temporal graph and graph integration contract
  ml.py         optional importance-model integration
  __main__.py   unchanged CLI entry point

evaluation/     benchmark, human-review, and reporting runners
training/       optional importance-model training runner
tests/          unit and integration tests
docs/           system documentation and generated reports
data/           datasets, research inputs, and local model/evaluation artifacts
```

The thin `main.<module>` compatibility modules retain the prior public Python
import paths, including `main.core`, `main.models`, `main.store`, `main.rag`,
and `main.cleanup`. They contain no application logic.

## Runtime path assumptions

Commands continue to resolve relative paths from the repository working
directory. The JSONL default remains `memory_store/memories.jsonl`; data stays
under `data/`; reports stay under `docs/reports/`; and `.env` remains at the
repository root. No environment-variable names, file formats, or database table
names changed during this reorganization.
