"""Deterministic lifecycle and task-state evaluation without learned importance."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from main import LifecycleManager, LocalMemoryStore, MemoryCore, MemoryInput, MemoryTier

NOW = datetime(2026, 9, 26, 10, tzinfo=timezone.utc)


def _process(text, *, at=NOW, **metadata):
    return MemoryCore().process(MemoryInput(text, 'lifecycle-user', 'lifecycle-session',
                                           timestamp=at, metadata=metadata))


def evaluate():
    manager = LifecycleManager()
    cases = []

    def check(name, expected, actual):
        if isinstance(expected, datetime):
            expected = expected.isoformat()
        if isinstance(actual, datetime):
            actual = actual.isoformat()
        cases.append({'check': name, 'expected': expected, 'actual': actual,
                      'passed': expected == actual})

    check('working_tier', 'working', _process('Hello').tier.value)
    check('one_off_task_tier', 'short_term', _process('Submit the report tomorrow').tier.value)
    check('preference_tier', 'long_term', _process('I prefer concise technical summaries.').tier.value)
    check('recurring_task_tier', 'long_term', _process('Remind me to submit the report every Friday').tier.value)
    due = _process('Submit the report tomorrow')
    check('task_expiry', due.due_at + timedelta(hours=24), due.expires_at)
    old = replace(_process('My name is Joseph.', at=NOW - timedelta(days=120)),
                  tier=MemoryTier.LONG_TERM)
    check('inactive_long_term_archives', 'archive', manager.refresh(old, now=NOW).tier.value)

    with TemporaryDirectory() as directory:
        store = LocalMemoryStore(Path(directory) / 'lifecycle.jsonl')
        completed = store.ingest(_process('Submit the budget report tomorrow'))
        store.ingest(_process('I submitted the budget report.', at=NOW + timedelta(hours=1)))
        check('task_completion', 'completed', store.get(completed.id).task_status)
        check('completed_task_hidden', False, completed.id in {row.id for row in store.list(user_id='lifecycle-user')})
        check('completion_history_preserved', 2, len(store.history(completed.id, user_id='lifecycle-user')))

    with TemporaryDirectory() as directory:
        store = LocalMemoryStore(Path(directory) / 'lifecycle.jsonl')
        cancelled = store.ingest(_process('Schedule the planning meeting tomorrow'))
        store.ingest(_process('The planning meeting has been cancelled.', at=NOW + timedelta(hours=2)))
        check('task_cancellation', 'cancelled', store.get(cancelled.id).task_status)

        rescheduled = store.ingest(_process('Send the invoice tomorrow'))
        store.ingest(_process('Move the invoice to 2026-10-02 at 5 pm', at=NOW + timedelta(hours=3)))
        check('task_rescheduling', '2026-10-02T17:00:00+00:00', store.get(rescheduled.id).due_at.isoformat())

    with TemporaryDirectory() as directory:
        store = LocalMemoryStore(Path(directory) / 'lifecycle.jsonl')
        recurring = store.ingest(_process('Remind me to pay rent monthly', due_at='2026-09-30T17:00:00Z'))
        store.ingest(_process('I paid rent', at=NOW + timedelta(days=4),
                              task_id=recurring.id, occurrence_at='2026-09-30'))
        latest = store.get(recurring.id)
        check('recurring_task_remains_active', 'active', latest.task_status)
        check('recurring_occurrence_recorded', 1, len(latest.task_occurrences))
        check('recurring_next_occurrence', '2026-10-30', latest.next_due_at.date().isoformat())

    return {'importance_model': 'disabled', 'checks': len(cases),
            'passed': sum(row['passed'] for row in cases),
            'accuracy': sum(row['passed'] for row in cases) / len(cases), 'results': cases}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/reports/ablation/lifecycle.json'))
    args = parser.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'results'}, indent=2))


if __name__ == '__main__':
    main()
