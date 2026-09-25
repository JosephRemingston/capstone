# Lifecycle and conversational processing changes

## Implemented behavior

1. **Deadlines.** Tasks have `due_at`. English today/tonight/tomorrow, next week,
   named weekdays, numeric relative durations, and ISO dates/datetimes are parsed.
   Optional `at/by` times use 24h or am/pm notation. Date-only deadlines use
   23:59:59 in the input timestamp's timezone; timestamps default to UTC. “Next
   Monday” is the next Monday strictly after today. Unknown or explicitly negated/
   alternative wording returns no inferred date. Invalid recognized dates raise
   an input error. Explicit `metadata.due_at` overrides inference. Dated tasks
   expire 24 hours after their deadline; undated one-off tasks retain the 14-day
   policy. Recurring tasks are standing instructions without one inferred expiry.
   Explicit single deadlines on recurring tasks are rejected.
2. **Task updates.** `LocalMemoryStore.ingest()` and CLI processing recognize
   affirmative user completion/cancellation. A unique exact normalized object
   match for the same user resolves the previous task; ambiguous matches leave
   tasks active and expose candidate IDs. `metadata.task_id`/`--task-id` selects
   explicitly and enforces ownership, active status, and event chronology.
   Negation, questions, hypotheses, assistant statements and ambiguous mixed
   clauses do not auto-close tasks. Recurring series are not auto-closed by one
   completion. This is conservative lexical matching, not general coreference.
3. **Negation.** Unfinished/submitted-negative statements remain tasks. Negated
   completions never close an earlier task. Disclaiming a need does not create
   a new positive task. Complex negation still needs richer language handling.
4. **Multiple memories.** `MemoryCore.process_many()` and CLI `--split` separate
   explicit independent clauses; each gets its own record and provenance.
   Simple conjunction lists and procedural sequences remain intact. Quoted text
   remains intact. Shared explicit task IDs/deadlines on split inputs are rejected.
   The original single-record `process()` contract remains available.
5. **Archive age.** Inactivity is measured from `last_accessed_at` (if present) or
   `created_at` to the current time, not from creation to the last metadata update.
   `touch()` records explicit access. Long-term memories become archive views after
   90 days of inactivity. Open tasks are excluded from inactivity archiving.
6. **Retrieval visibility.** List/search omit expired records at `expires_at <= now`
   and omit resolved tasks. Flags can include them for inspection. Long-term archive
   views are computed before tier filtering. Reads do not mutate the log or increment
   access counts. `get()` and `all()` are historical inspection APIs and include
   expired/resolved records; `all()` returns the latest revision per ID.
7. **Evaluation.** A 48-case developer evaluation and blank human-review template
   are supplied. Category/tier accuracy: 44/48; importance-range agreement: 44/48.
   This is not human validation or statistical threshold calibration. Four failures
   are included in the JSON report and human review remains outstanding.

## Persistence and compatibility

The JSONL file is an append-only revision log. Resolving a task appends its revised
state plus the observation; the old line is retained. Latest line for an ID wins.
Raw `save()` persists without automatic task matching; `ingest()` provides that
behavior. IDs may not change owners. Older rows deserialize with null defaults
for new fields. There is no database transaction, concurrent-writer guarantee,
or automatic disk cleanup. General contradictory facts and rescheduling are not
resolved by the task completion mechanism.

The optional Hippocorpus scorer still uses its frozen classifier/feature inputs.
The new lifecycle policy applies to both scorers; no model was retrained and no
claims are made about improving its regression metrics.

## Verification

40 automated tests pass with optional ML dependencies installed. Tests exercise
parsed dates/timezones, expiry boundaries, read-only archival, serialization,
completion/cancellation, ambiguous matches, cross-user and chronology protections,
negation, recurring series, numeric task identities, segmentation, CLI persistence,
and evaluation label provenance. Existing core/retrieval/ML tests also pass.

Results: `reports/conversational_evaluation.json`.
Review guidance: `evaluation/README.md`.
