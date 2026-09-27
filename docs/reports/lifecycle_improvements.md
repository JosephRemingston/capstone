# Conversational rules and task management

## Implemented behavior

- Indirect preferences include “I would rather…”, “I'm not a fan of…”, “works
  better for me”, habits such as taking tea without milk, and dietary constraints.
  Negative/double-negative preferences retain the original text; a category label
  does not turn “don't dislike” into “dislike”. Uncertain/reported assertions do
  not provide evidence for a task state change.
- `process_many()` / `--split` separates explicit independent clauses outside
  quoted spans. Object lists, procedural sequences, and conditional scope stay
  intact. Shared task IDs, deadlines, occurrence selectors, or scopes on split
  messages are rejected because they cannot safely target every clause.
- Completion recognizes submitted/completed/finished/paid/sent/bought, “wrapped
  up”, “taken care of”, and “done with”. Cancellation supports cancelled/called
  off, direct cancel/skip requests, and stopping recurring reminders.
- Rescheduling supports “move/reschedule/postpone … to …”, “postpone … by N days”,
  “is now due …”, changed-deadline statements, and “is not due X but Y”. Replacement
  dates are parsed separately from old dates. A relative delay is added to the
  stored deadline. The existing task ID is retained and expiry is recalculated.
- References prefer an exact normalized object, followed by a unique noun subset
  (“that report” → “budget report”). Numeric identifiers must agree. Bare pronouns
  require exactly one active candidate in the same session. `--task-id` resolves
  ambiguity while enforcing user ownership, active state, and update chronology.
- Recurrence supports daily, weekly, monthly, yearly, one named weekday, and
  integer day/week/month/year intervals. `--due-at` supplies the first occurrence.
  `recurrence` stores a stable calendar anchor and cursor; `next_due_at` stores the
  current deadline. Month/year calculations preserve the original day when it
  exists (January 31 → February 28 → March 31; February 29 returns in leap years).
- Completing/skipping an occurrence appends an entry to `task_occurrences` and
  advances the cursor without resolving the series. Unqualified completion must
  match the current occurrence's local date. `--occurrence-at` explicitly selects
  the current original/rescheduled date or datetime for early/overdue completion.
  Future occurrences cannot be processed out of order. Duplicate event IDs do not
  reapply updates; explicitly repeated resolved occurrences are identified.
- Rescheduling one occurrence preserves the cadence and must stay before its
  successor. `--task-scope series` re-anchors or closes the series; a new anchor
  cannot overlap resolved occurrence history. “Stop reminding me to…” closes it.
- One-off deadlines expire after 24 hours of grace; undated tasks retain the
  14-day policy. Series never expire because a single occurrence is overdue.
  List/search hide expired/resolved records by default. Reads never advance a
  recurrence or mutate the JSONL log. Inactivity/archive behavior is unchanged.

## Update results and limits

`source_metadata.task_resolution` records `applied`, `ambiguous`, `unmatched`,
`missing_deadline`, `occurrence_required`, `occurrence_not_current`,
`already_resolved`, `unsupported_schedule`, `occurrence_overlaps_next`, or
`series_overlaps_history`. Ambiguity includes candidate IDs; applied changes
include the target ID and a change summary. Invalid explicit input raises a
`ValueError` before appending a revision.

These are deterministic English rules. Arbitrary paraphrases, sarcasm, implicit
shared subjects, and general coreference are not guaranteed. Multiple weekday
schedules and recurrence exceptions are unsupported. Recurrence uses the anchor's
fixed timezone offset, without named timezones/DST. Unspecified times mean end
of day, including morning/evening wording. There is no notification scheduler.
Occurrences are handled in order; old unresolved occurrences are not silently
skipped. Use the documented explicit fields when a reference/date is unclear.

## Persistence and compatibility

The JSONL store is an append-only revision log. A successful update appends the
revised task plus its observation. Latest row per ID wins; prior rows remain.
`save()` provides raw persistence; `ingest()` applies task actions. IDs cannot
change owners. Old records load with empty recurrence/history defaults, and old
recurring tasks are upgraded when updated. Concurrency guarantees, database
transactions, cleanup, and general fact/preference conflict resolution remain
separate work.

## Verification

- `.venv/bin/python -m unittest -q`: **57 tests pass**, including the optional
  scorer compatibility tests. The new workflow tests cover indirect preferences,
  negation, mixed messages, references, rescheduling, recurrence, month ends,
  leap years, serialization/timezones, ambiguity, owner/chronology checks,
  invalid metadata, retries, old-record upgrades, and CLI integration.
- `python3 -m evaluation.evaluate`: **48/48 category matches**, **48/48 tier
  matches**, macro F1 **1.0**, and **48/48 importance-range matches**, up from
  44/48. Labels were not changed. These examples were used during development;
  this is regression coverage, not independent accuracy on unseen conversations.

Full evaluation: [conversational_evaluation.json](conversational_evaluation.json).
Usage: [README](../README.md#deadlines-task-updates-and-multiple-memories).
