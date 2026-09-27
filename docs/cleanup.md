# Scheduled cleanup and safe compaction

Cleanup physically removes **all revisions** of eligible expired memories from
the JSONL store and invalidates the user's registered SQLite indexes. It does
not merely hide records during retrieval. This feature is implemented for local
macOS/Linux stores; a scheduler has not been installed on your account.

## Preview, apply, and schedule

Run from the repository root, substituting your actual user ID and store path:

```bash
# Preview is the default; no memory data is removed.
.venv/bin/python -m main --store memory_store/memories.jsonl cleanup --user-id USER --pretty

# Apply the same policy (default: seven days beyond expiry).
.venv/bin/python -m main --store memory_store/memories.jsonl cleanup --user-id USER --apply --pretty

# One scheduler tick, suitable for cron or another supervisor.
.venv/bin/python -m main.maintenance --store /absolute/path/memories.jsonl --user-id USER --once

# Long-running worker; attempts a due run immediately, then checks every minute.
.venv/bin/python -m main.maintenance --store /absolute/path/memories.jsonl --user-id USER --interval-hours 24
```

For unattended operation, run the worker under a supervisor, or invoke `--once`
from cron/launchd. For example, this cron template checks hourly while persisted
state limits successful cleanup to once per 24 hours:

```cron
0 * * * * cd /absolute/path/capstone && /absolute/path/capstone/.venv/bin/python -m main.maintenance --store /absolute/path/memories.jsonl --user-id USER --once >> /absolute/path/cleanup.log 2>&1
```

The worker handles SIGINT/SIGTERM, exits nonzero on a failed `--once` run, and
retries failed continuous runs without advancing the schedule. Schedule state
is stored next to the JSONL file, keyed by user, grace period, and interval.
A restart resumes the persisted schedule; it does not restart the waiting period.
Changing the policy creates a new schedule. `cleanup --scheduled --apply` offers
the same due-check directly through the CLI.

## Eligibility and retained evidence

A memory can be removed only when its latest revision has `expires_at` at least
`--grace-days` in the past (default 7), belongs to the requested user, and is not
protected. Nonexpiring records are not eligible just because they are archived
or superseded. Open tasks are retained even if their deadline has expired.
Completed tasks need an explicit expired `expires_at` to qualify.

A `legal_hold` or `retain` flag in any revision's source metadata protects that
memory. This conservative policy has no automatic hold-release mechanism.
References from **all revisions** of retained memories protect their dependencies
transitively, including consolidated evidence, conflict winners/observations,
task updates, recurring occurrences, and ambiguous task candidates. References
across users also protect targets. An entirely expired dependency cycle may be
removed together if no retained record depends on it.

Compaction removes only byte-equivalent normalized snapshots repeated for the
same ID. It preserves meaningful transitions, including A → B → A, and keeps
recording timestamps and exact source-revision fingerprints for historical graph
queries. It does not collapse every memory to its latest state. JSON whitespace
is compacted too; reported reclaimed bytes include this serialization saving.

## Failure behavior and derived copies

Store reads, ingestion, conflict resolution, RAG synchronization/search, and
cleanup cooperate through a reentrant process/file lock. This prevents cleanup
from replacing a log while a participating writer appends to it. Direct manual
file edits and low-level index APIs bypass that contract and must not run during
maintenance.

Cleanup parses the complete store before mutation and fails on corruption or
unknown top-level fields. It validates the source hash before replacement.
The replacement file is staged in the same directory, fsynced, atomically renamed,
and the directory fsynced. No raw-history backup is left behind by cleanup.

Before deleting source records, cleanup purges the selected user's vectors,
graph edges, aliases, nodes, and index state from the default and registered
custom SQLite indexes, using secure-delete, checkpointing, and vacuum. All
`MemoryRAG` instances register their index path automatically. An index-purge
failure leaves source history intact; partially cleared caches can be rebuilt.
A crash after log replacement but before schedule persistence is safe to retry.
Old custom indexes never registered through `MemoryRAG` are outside this registry
and must be removed or registered separately.

Atomic file replacement is physical removal from the active application files,
not a guarantee of forensic erasure from SSD snapshots, external backups, or OS
caches. Existing backups are not managed by this feature.

## Verification

Tests cover previews, grace periods, other users, reference closure, retention
holds, open tasks, dependency cycles, meaningful history, default/custom index
purging, source-write/index failures, schedule retries, corrupt input, invalid
policies, and a concurrent writer. The external evaluation applies cleanup only
to temporary replay stores and compares visible records before/after. See
[the generated evaluation report](reports/independent_evaluation.md).
