"""Deterministic lifecycle, task-state, and supersession checks with a fixed clock.

    python -m evaluation.lifecycle
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from main import LifecycleManager, LocalMemoryStore, MemoryCore, MemoryInput, MemoryTier
from main.ensemble import describe_scorer

from .clock import pinned_clock

NOW = datetime(2026, 10, 12, 10, tzinfo=timezone.utc)
USER = 'lifecycle-user'


def process(text: str, *, at: datetime = NOW, **metadata):
    return MemoryCore().process(MemoryInput(text, USER, 'lifecycle-session', timestamp=at, metadata=metadata))


def evaluate() -> dict:
    with pinned_clock(NOW):   # tiers, archiving and visibility read the clock; the checks are written for NOW
        return _evaluate()


def _evaluate() -> dict:
    checks = []

    def check(name, expected, actual):
        expected = expected.isoformat() if isinstance(expected, datetime) else expected
        actual = actual.isoformat() if isinstance(actual, datetime) else actual
        checks.append({'check': name, 'expected': expected, 'actual': actual, 'passed': expected == actual})

    check('greeting_is_working_memory', 'working', process('Hey, good morning!').tier.value)
    check('one_off_task_is_short_term', 'short_term', process('Submit the expense report tomorrow').tier.value)
    check('preference_is_long_term', 'long_term', process('I prefer short technical summaries.').tier.value)
    check('recurring_task_is_long_term', 'long_term', process('Remind me to back up the laptop every Friday').tier.value)
    dated = process('Submit the expense report tomorrow')
    check('dated_task_expires_a_day_after_deadline', dated.due_at + timedelta(hours=24), dated.expires_at)
    stale = replace(process('My name is Joseph.', at=NOW - timedelta(days=120)), tier=MemoryTier.LONG_TERM)
    check('inactive_long_term_memory_archives', 'archive', LifecycleManager().refresh(stale, now=NOW).tier.value)

    with TemporaryDirectory() as directory:
        store = LocalMemoryStore(Path(directory) / 'lifecycle.jsonl')
        task = store.ingest(process('Submit the budget report tomorrow'))
        store.ingest(process('I submitted the budget report.', at=NOW + timedelta(hours=1)))
        check('completion_statement_completes_task', 'completed', store.get(task.id).task_status)
        check('completed_task_hidden_from_default_list', False, task.id in {r.id for r in store.list(user_id=USER)})
        check('completion_keeps_revision_history', 2, len(store.history(task.id, user_id=USER)))

    with TemporaryDirectory() as directory:
        store = LocalMemoryStore(Path(directory) / 'lifecycle.jsonl')
        meeting = store.ingest(process('Schedule the planning meeting tomorrow'))
        store.ingest(process('The planning meeting has been cancelled.', at=NOW + timedelta(hours=2)))
        check('cancellation_statement_cancels_task', 'cancelled', store.get(meeting.id).task_status)
        invoice = store.ingest(process('Send the invoice tomorrow'))
        store.ingest(process('Move the invoice to 2026-10-20 at 5 pm', at=NOW + timedelta(hours=3)))
        check('reschedule_statement_moves_deadline', '2026-10-20T17:00:00+00:00', store.get(invoice.id).due_at.isoformat())

    with TemporaryDirectory() as directory:
        store = LocalMemoryStore(Path(directory) / 'lifecycle.jsonl')
        rent = store.ingest(process('Remind me to pay rent monthly', due_at='2026-10-31T17:00:00Z'))
        store.ingest(process('I paid rent', at=NOW + timedelta(days=20), task_id=rent.id, occurrence_at='2026-10-31'))
        latest = store.get(rent.id)
        check('recurring_task_stays_active_after_payment', 'active', latest.task_status)
        check('recurring_occurrence_is_recorded', 1, len(latest.task_occurrences))
        check('recurring_task_advances_to_next_due_date', '2026-11-30', latest.next_due_at.date().isoformat())

    with TemporaryDirectory() as directory:
        store = LocalMemoryStore(Path(directory) / 'lifecycle.jsonl')
        old = store.ingest(process('I live in Chennai.', at=NOW - timedelta(days=30)))
        store.ingest(process('I live in Bengaluru.', at=NOW))
        visible = {r.content for r in store.list(user_id=USER)}
        check('superseded_fact_hidden_from_current_view', False, 'I live in Chennai.' in visible)
        check('superseded_fact_kept_in_history', 'superseded',
              store.get(old.id).memory_status if store.get(old.id) else None)

    return {'importance_model': describe_scorer(MemoryCore().importance_scorer), 'clock': NOW.isoformat(), 'checks': len(checks),
            'passed': sum(c['passed'] for c in checks), 'results': checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=Path('docs/reports/lifecycle.json'))
    args = parser.parse_args()
    report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({k: report[k] for k in ('checks', 'passed')}))
    for row in report['results']:
        if not row['passed']:
            print('FAILED', row)


if __name__ == '__main__':
    main()
