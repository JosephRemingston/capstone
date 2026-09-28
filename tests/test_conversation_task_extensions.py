"""Regression tests for conversational language and stateful task workflows."""
import io
import json
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from main import MemoryCore, MemoryInput, LocalMemoryStore, MemoryRecord
from main.__main__ import main as cli_main
from main.domain.recurrence import parse_recurrence, scheduled_at

NOW = datetime(2026, 9, 26, 10, tzinfo=timezone.utc)


def process(text, at=NOW, user='u', session='s', **metadata):
    return MemoryCore().process(MemoryInput(content=text, user_id=user, session_id=session,
                                           timestamp=at, metadata=metadata))


class ConversationExtensions(unittest.TestCase):
    def test_indirect_preferences_and_constraints(self):
        for text in ["I would rather receive short answers.", "I'd rather avoid spicy food.",
                     "I'm not a fan of noisy restaurants.", 'Short answers work better for me.',
                     'I usually take tea without milk.', "I don't really like emojis.",
                     'Please remember that my cat cannot eat tuna.',
                     "I don't mind longer answers.", "I do not dislike coffee.",
                     'I prefer the label "if needed".', "I wouldn't mind longer answers."]:
            with self.subTest(text=text):
                record = process(text, at=datetime.now(timezone.utc))
                self.assertEqual(record.category.value, 'preference')
                self.assertEqual(record.tier.value, 'long_term')
                self.assertEqual(record.content, text)
                self.assertEqual(record.features['preference_signal'], 1)

    def test_mixed_messages_quotes_and_procedure_scope(self):
        def split(text):
            return MemoryCore().process_many(MemoryInput(content=text, user_id='u', session_id='s'))
        records = split('I would rather have short answers, but remind me to send the invoice tomorrow.')
        self.assertEqual([r.category.value for r in records], ['preference', 'task'])
        quoted = split('She said "I finished it; I sent it."; I prefer concise answers.')
        self.assertEqual(len(quoted), 2)
        self.assertEqual(quoted[0].category.value, 'temporary')
        self.assertEqual(quoted[1].category.value, 'preference')
        self.assertEqual(len(split('First run tests, then build; finally deploy.')), 1)
        self.assertEqual(len(split('If I finish the report, I will call you; I submitted it.')), 1)
        self.assertEqual(len(split('Buy milk and eggs.')), 1)
        self.assertEqual(len(split('I prefer tea; buy milk tomorrow.')), 2)


class TaskWorkflowExtensions(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = LocalMemoryStore(Path(temp.name) / 'store.jsonl')

    def ingest(self, text, **kwargs):
        return self.store.ingest(process(text, **kwargs))

    def test_rescheduling_updates_same_task_and_expiry(self):
        task = self.ingest('Submit the budget report tomorrow')
        event = self.ingest('Move that report from tomorrow to 2026-10-02 at 5 pm')
        self.assertEqual(event.related_task_id, task.id)
        updated = self.store.get(task.id)
        self.assertEqual(updated.due_at, datetime(2026, 10, 2, 17, tzinfo=timezone.utc))
        self.assertEqual(updated.expires_at, updated.due_at + timedelta(hours=24))
        self.assertEqual(updated.task_status, 'active')
        self.assertEqual(len([r for r in self.store.all() if r.category.value == 'task']), 1)
        self.assertEqual(len(self.store.path.read_text().splitlines()), 3)
        self.ingest('The budget report is now due next Monday at 9 am')
        self.assertEqual(self.store.get(task.id).due_at, datetime(2026, 9, 28, 9, tzinfo=timezone.utc))
        self.ingest('The deadline for the budget report changed from Monday to Tuesday')
        self.assertEqual(self.store.get(task.id).due_at.date().isoformat(), '2026-09-29')

    def test_explicit_reschedule_missing_date_and_wrong_owner(self):
        task = self.ingest('Send the invoice tomorrow')
        unknown = self.ingest('Postpone the invoice to sometime later')
        self.assertEqual(unknown.source_metadata['task_resolution'], 'missing_deadline')
        self.assertEqual(self.store.get(task.id).due_at, task.due_at)
        updated = self.ingest('Reschedule it', task_id=task.id, due_at='2026-11-01T11:00:00+05:30')
        self.assertEqual(updated.related_task_id, task.id)
        self.assertEqual(self.store.get(task.id).due_at, datetime.fromisoformat('2026-11-01T11:00:00+05:30'))
        before = self.store.path.read_bytes()
        with self.assertRaises(ValueError):
            self.ingest('Move it to tomorrow', task_id=task.id, user='other')
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_date_corrections_and_relative_delay_use_current_deadline(self):
        task = self.ingest('Submit the report tomorrow at 5 pm')
        self.ingest('Postpone the report by 2 days')
        self.assertEqual(self.store.get(task.id).due_at, datetime(2026, 9, 29, 17, tzinfo=timezone.utc))
        self.ingest('The report is not due Tuesday but Friday at 9 am')
        self.assertEqual(self.store.get(task.id).due_at, datetime(2026, 10, 2, 9, tzinfo=timezone.utc))
        self.ingest('Change the deadline for the report to next Monday')
        self.assertEqual(self.store.get(task.id).due_at.date().isoformat(), '2026-09-28')

    def test_unsupported_schedule_is_visible_and_cannot_be_advanced(self):
        task = self.ingest('Remind me to send invoices every Monday and Friday')
        self.assertEqual(task.source_metadata['recurrence_resolution'], 'unsupported_schedule')
        event = self.ingest('I completed it', task_id=task.id)
        self.assertEqual(event.source_metadata['task_resolution'], 'unsupported_schedule')
        self.assertEqual(self.store.get(task.id).task_status, 'active')

    def test_legacy_recurring_record_is_upgraded_on_update(self):
        task = process('Remind me to submit report daily')
        task.recurrence = task.next_due_at = None
        self.store.save(task)
        self.ingest('I submitted report')
        updated = self.store.get(task.id)
        self.assertIsNotNone(updated.recurrence)
        self.assertEqual(len(updated.task_occurrences), 1)

    def test_stop_reminders_closes_series_and_bad_metadata_does_not_write(self):
        task = self.ingest('Remind me to pay rent monthly')
        before = self.store.path.read_bytes()
        for metadata in [{'task_scope': 'all'}, {'occurrence_at': []}, {'task_id': ''}]:
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                self.ingest('I paid rent', **metadata)
            self.assertEqual(self.store.path.read_bytes(), before)
        self.ingest('Stop reminding me to pay rent')
        self.assertEqual(self.store.get(task.id).task_status, 'cancelled')
        before = self.store.path.read_bytes()
        with self.assertRaises(ValueError):
            self.ingest('I finished the unknown task', occurrence_at='invalid')
        with self.assertRaises(ValueError):
            self.ingest('Move the unknown task to later', due_at='invalid')
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_reference_ambiguity_pronouns_and_session_boundaries(self):
        a = self.ingest('Submit the budget report tomorrow')
        b = self.ingest('Submit the expense report tomorrow')
        event = self.ingest('I finished that report')
        self.assertEqual(event.source_metadata['task_resolution'], 'ambiguous')
        self.assertEqual(set(event.source_metadata['candidate_task_ids']), {a.id, b.id})
        self.ingest("I've wrapped up the budget report")
        self.assertEqual(self.store.get(a.id).task_status, 'completed')
        foreign_session = self.ingest("I'm done with it", session='another')
        self.assertEqual(foreign_session.source_metadata['task_resolution'], 'unmatched')
        self.ingest("I'm done with it")
        self.assertEqual(self.store.get(b.id).task_status, 'completed')

    def test_negation_hypotheticals_and_mixed_actions_do_not_mutate(self):
        task = self.ingest('Submit the report tomorrow')
        for text in ["I didn't fail to submit the report", "I haven't quite finished the report",
                     "I'm not sure I completed the report", 'The report was not cancelled',
                     'If the report is finished, move it to Monday', 'Move the report to Monday or Tuesday',
                     'I completed the report but not the invoice', 'I did not say I finished the report',
                     'She said "I completed the report"', 'Did I complete the report?',
                     'Do not move the report to Monday', "I can't cancel the report"]:
            with self.subTest(text=text):
                self.ingest(text)
                updated = self.store.get(task.id)
                self.assertEqual(updated.task_status, 'active')
                self.assertEqual(updated.due_at, task.due_at)

    def test_recurrence_completion_advance_and_idempotency(self):
        task = self.ingest('Remind me to submit the report daily at 5 pm')
        self.assertEqual(task.next_due_at, NOW.replace(hour=17))
        self.assertIsNone(task.expires_at)
        event = self.ingest('I submitted the report')
        updated = self.store.get(task.id)
        self.assertEqual(updated.task_status, 'active')
        self.assertEqual(len(updated.task_occurrences), 1)
        self.assertEqual(updated.next_due_at, NOW.replace(hour=17) + timedelta(days=1))
        before = self.store.path.read_bytes()
        self.store.ingest(event)
        self.assertEqual(self.store.path.read_bytes(), before)
        duplicate = self.ingest('I submitted the report', occurrence_at='2026-09-26')
        self.assertEqual(duplicate.source_metadata['task_resolution'], 'already_resolved')
        early = self.ingest('I submitted the report')
        self.assertEqual(early.source_metadata['task_resolution'], 'occurrence_required')
        self.ingest('I submitted the report', at=NOW + timedelta(days=1))
        self.assertEqual(len(self.store.get(task.id).task_occurrences), 2)
        self.assertEqual(MemoryRecord.from_dict(updated.to_dict()), updated)

    def test_occurrence_and_series_cancellation(self):
        task = self.ingest('Remind me to pay rent monthly', due_at='2026-09-30T17:00:00Z')
        self.ingest('Skip rent', task_id=task.id)
        updated = self.store.get(task.id)
        self.assertEqual(updated.task_status, 'active')
        self.assertEqual(updated.next_due_at.date().isoformat(), '2026-10-30')
        self.assertEqual(updated.task_occurrences[0]['status'], 'cancelled')
        self.ingest('Cancel rent', task_id=task.id, task_scope='series')
        self.assertEqual(self.store.get(task.id).task_status, 'cancelled')
        self.assertIsNone(self.store.get(task.id).next_due_at)

    def test_recurring_reschedule_keeps_anchor_and_series_reanchors(self):
        task = self.ingest('Remind me to send the invoice every week', due_at='2026-09-28T09:00:00Z')
        self.ingest('Move the invoice to 2026-09-29 at 10 am')
        updated = self.store.get(task.id)
        self.assertEqual(updated.next_due_at, datetime(2026, 9, 29, 10, tzinfo=timezone.utc))
        self.ingest('I sent the invoice', occurrence_at='2026-09-29')
        self.assertEqual(self.store.get(task.id).next_due_at, datetime(2026, 10, 5, 9, tzinfo=timezone.utc))
        self.ingest('Move the invoice to 2026-10-07 at 11 am', task_scope='series')
        self.ingest('I sent the invoice', occurrence_at='2026-10-07')
        self.assertEqual(self.store.get(task.id).next_due_at, datetime(2026, 10, 14, 11, tzinfo=timezone.utc))

    def test_invalid_occurrences_and_out_of_order_events(self):
        task = self.ingest('Remind me to pay rent monthly', due_at='2026-09-30')
        event = self.ingest('I paid rent', occurrence_at='2026-10-30')
        self.assertEqual(event.source_metadata['task_resolution'], 'occurrence_not_current')
        event = self.ingest('Move rent to 2026-11-01')
        self.assertEqual(event.source_metadata['task_resolution'], 'occurrence_overlaps_next')
        with self.assertRaises(ValueError):
            self.ingest('I paid rent', occurrence_at='invalid')
        self.ingest('I paid rent', occurrence_at='2026-09-30', at=NOW + timedelta(days=4))
        old = self.ingest('Cancel rent')
        self.assertEqual(old.source_metadata['task_resolution'], 'unmatched')
        self.assertEqual(len(self.store.get(task.id).task_occurrences), 1)

    def test_cli_occurrence_and_explicit_rescheduling(self):
        def run(*args):
            output = io.StringIO()
            with redirect_stdout(output):
                code = cli_main(['--store', str(self.store.path), 'process', *args,
                                 '--user-id', 'u', '--session-id', 's'])
            self.assertEqual(code, 0, output.getvalue())
            return json.loads(output.getvalue())
        task = run('Remind me to pay rent monthly', '--due-at', '2026-10-01')
        run('Move it to tomorrow', '--task-id', task['id'], '--task-scope', 'series')
        run('I paid rent', '--task-id', task['id'], '--occurrence-at',
            self.store.get(task['id']).next_due_at.date().isoformat())
        self.assertEqual(len(self.store.get(task['id']).task_occurrences), 1)


class RecurrenceCalendarTests(unittest.TestCase):
    def test_month_end_leap_year_and_interval(self):
        rule = parse_recurrence('Remind me to pay rent monthly', NOW, '2026-01-31T10:00:00+05:30')
        self.assertEqual(scheduled_at(rule, 1).date().isoformat(), '2026-02-28')
        self.assertEqual(scheduled_at(rule, 2).date().isoformat(), '2026-03-31')
        yearly = parse_recurrence('Remind me yearly', NOW, '2024-02-29')
        self.assertEqual(scheduled_at(yearly, 1).date().isoformat(), '2025-02-28')
        self.assertEqual(scheduled_at(yearly, 4).date().isoformat(), '2028-02-29')
        fortnight = parse_recurrence('Remind me every 2 weeks', NOW)
        self.assertEqual(scheduled_at(fortnight, 1) - scheduled_at(fortnight, 0), timedelta(days=14))
        with self.assertRaises(ValueError):
            parse_recurrence('Remind me every 0 days', NOW)

    def test_timezone_roundtrip_does_not_move_occurrence_day(self):
        task = process('Remind me to send report daily', due_at='2026-09-27T00:30:00+05:30')
        restored = MemoryRecord.from_dict(task.to_dict())
        self.assertEqual(scheduled_at(restored.recurrence).hour, 0)
        self.assertEqual(restored.next_due_at, task.next_due_at)
