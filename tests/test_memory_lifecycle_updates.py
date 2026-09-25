"""Behavioral tests for deadlines, task state, segmentation and read visibility."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from main import MemoryCore, MemoryInput, LocalMemoryStore, LifecycleManager, MemoryCategory, MemoryRecord, MemoryTier
from main.__main__ import main as cli_main
from main.deadlines import parse_deadline
from main.tasks import task_outcome

NOW = datetime(2026, 9, 26, 10, 0, tzinfo=timezone.utc)


def process(text, user='u', timestamp=NOW, **metadata):
    return MemoryCore().process(MemoryInput(content=text, user_id=user, session_id='s',
                                           timestamp=timestamp, metadata=metadata))


class DeadlineTests(unittest.TestCase):
    def test_relative_and_absolute_deadlines(self):
        zone = timezone(timedelta(hours=5, minutes=30))
        reference = NOW.astimezone(zone)
        cases = {
            'Submit the report tomorrow at 5 pm': datetime(2026, 9, 27, 17, tzinfo=zone),
            'Buy groceries today': datetime(2026, 9, 26, 23, 59, 59, tzinfo=zone),
            'Send the report in 2 hours': reference + timedelta(hours=2),
            'Submit the report next Monday by 09:30': datetime(2026, 9, 28, 9, 30, tzinfo=zone),
            'Pay rent by 2026-11-01': datetime(2026, 11, 1, 23, 59, 59, tzinfo=zone),
            'Submit by 2026-10-01T17:00:00Z': datetime(2026, 10, 1, 17, tzinfo=timezone.utc),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_deadline(text, reference), expected)
        self.assertIsNone(parse_deadline('Do this sometime soon', reference))
        self.assertIsNone(parse_deadline('Not tomorrow', reference))
        self.assertIsNone(parse_deadline('Today or tomorrow', reference))
        with self.assertRaises(ValueError):
            parse_deadline('by 2026-02-30', reference)
        with self.assertRaises(ValueError):
            parse_deadline('tomorrow at 25:00', reference)

    def test_deadline_controls_expiry_and_roundtrip(self):
        record = process('Pay rent by 2026-11-01')
        self.assertEqual(record.expires_at, record.due_at + timedelta(hours=24))
        self.assertGreater(record.expires_at, NOW + timedelta(days=14))
        self.assertEqual(MemoryRecord.from_dict(record.to_dict()), record)
        plain = process('Buy groceries')
        self.assertEqual(plain.expires_at, NOW + timedelta(days=14))
        explicit = process('Submit the report', due_at='2026-10-01T09:00:00+05:30')
        self.assertEqual(explicit.due_at.utcoffset(), timedelta(hours=5, minutes=30))
        recurring = process('Remind me to buy groceries every Monday')
        self.assertIsNone(recurring.due_at)
        self.assertIsNone(recurring.expires_at)


class TaskUpdateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = LocalMemoryStore(Path(self.directory.name) / 'store.jsonl')

    def test_completion_updates_unique_task_and_preserves_history(self):
        original = self.store.ingest(process('Remind me to submit the report tomorrow'))
        event = self.store.ingest(process('I submitted the report.', timestamp=NOW + timedelta(hours=1)))
        self.assertEqual(event.related_task_id, original.id)
        latest = self.store.get(original.id)
        self.assertEqual(latest.task_status, 'completed')
        self.assertEqual(latest.tier, MemoryTier.ARCHIVE)
        self.assertEqual(len(self.store.all()), 2)
        self.assertEqual(len(self.store.path.read_text().splitlines()), 3)
        self.assertNotIn(original.id, [r.id for r in self.store.search('report', now=NOW)])
        self.assertIn(original.id, [r.id for r in self.store.search('report', include_resolved=True, now=NOW)])

    def test_cancellation_and_cross_user_isolation(self):
        task = self.store.ingest(process('Schedule the meeting tomorrow'))
        foreign = self.store.ingest(process('The meeting has been cancelled.', user='other'))
        self.assertIsNone(foreign.related_task_id)
        self.assertEqual(self.store.get(task.id).task_status, 'active')
        event = self.store.ingest(process('The meeting has been cancelled.'))
        self.assertEqual(event.related_task_id, task.id)
        self.assertEqual(self.store.get(task.id).task_status, 'cancelled')

    def test_ambiguous_matching_requires_id(self):
        a = self.store.ingest(process('Submit the report tomorrow'))
        b = self.store.ingest(process('Remind me to submit the report today'))
        event = self.store.ingest(process('I submitted the report.'))
        self.assertEqual(event.source_metadata['task_resolution'], 'ambiguous')
        self.assertEqual(self.store.get(a.id).task_status, 'active')
        explicit = self.store.ingest(process('I completed it.', task_id=a.id))
        self.assertEqual(explicit.related_task_id, a.id)
        self.assertEqual(self.store.get(b.id).task_status, 'active')
        before = self.store.path.read_bytes()
        with self.assertRaises(ValueError):
            self.store.ingest(process('I completed it.', user='other', task_id=b.id))
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_negation_questions_and_hypotheticals_do_not_resolve(self):
        original = self.store.ingest(process('Submit the report tomorrow'))
        for text in ["I haven't submitted the report.", 'I have not completed the report.',
                     'I never submitted the report.', 'Have I completed the report?',
                     'If I completed the report, would you check it?', 'I will finish the report.',
                     'Remind me to check whether I completed the report.', '"I completed the report."']:
            with self.subTest(text=text):
                self.assertIsNone(task_outcome(text))
                self.store.ingest(process(text))
                self.assertEqual(self.store.get(original.id).task_status, 'active')
        self.assertEqual(process("I haven't finished the report.").category, MemoryCategory.TASK)

    def test_out_of_order_update_does_not_close_newer_task(self):
        task = self.store.ingest(process('Submit the report tomorrow', timestamp=NOW + timedelta(days=1)))
        event = self.store.ingest(process('I submitted the report.'))
        self.assertIsNone(event.related_task_id)
        self.assertEqual(self.store.get(task.id).task_status, 'active')

    def test_recurring_numbered_and_assistant_messages_are_safe(self):
        recurring = self.store.ingest(process('Remind me to submit the report every Friday'))
        numbered = self.store.ingest(process('Submit report 1 tomorrow'))
        self.store.ingest(process('I submitted report 2.'))
        self.store.ingest(process('I submitted the report.'))
        self.store.ingest(replace(process('I submitted report 1.'), role='assistant'))
        self.assertEqual(self.store.get(recurring.id).task_status, 'active')
        self.assertEqual(self.store.get(numbered.id).task_status, 'active')
        before = self.store.path.read_bytes()
        with self.assertRaises(ValueError):
            self.store.save(replace(numbered, user_id='other'))
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_split_completion_and_negation_are_applied_separately(self):
        report = self.store.ingest(process('Submit the report tomorrow'))
        invoice = self.store.ingest(process('Send the invoice tomorrow'))
        incoming = MemoryInput(content="I submitted the report, but I haven't sent the invoice.",
                               user_id='u', session_id='s', timestamp=NOW)
        records = MemoryCore().process_many(incoming)
        self.assertEqual(len(records), 2)
        for record in records:
            self.store.ingest(record)
        self.assertEqual(self.store.get(report.id).task_status, 'completed')
        self.assertEqual(self.store.get(invoice.id).task_status, 'active')


class LifecycleVisibilityTests(unittest.TestCase):
    def test_archive_uses_inactivity_not_updated_minus_created(self):
        manager = LifecycleManager()
        old = process('My name is Joseph.', timestamp=NOW - timedelta(days=120))
        old = replace(old, tier=MemoryTier.LONG_TERM, updated_at=old.created_at)
        self.assertEqual(manager.refresh(old, now=NOW).tier, MemoryTier.ARCHIVE)
        old.touch(NOW - timedelta(days=1))
        self.assertEqual(manager.refresh(old, now=NOW).tier, MemoryTier.LONG_TERM)
        self.assertEqual(manager.archive_after_for(old), NOW + timedelta(days=89))
        self.assertEqual(MemoryRecord.from_dict(old.to_dict()).last_accessed_at, old.last_accessed_at)

    def test_expiry_boundary_and_archive_views_are_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalMemoryStore(Path(directory) / 'memories.jsonl')
            expired = store.save(process('Hello'))
            live = store.save(process('I prefer Python examples.'))
            old = store.save(replace(live, id='old', created_at=NOW - timedelta(days=100), last_accessed_at=None))
            original = store.path.read_bytes()
            when = expired.expires_at
            self.assertNotIn(expired.id, [r.id for r in store.list(now=when)])
            self.assertEqual(store.search('hello', now=when), [])
            self.assertEqual(store.search('hello', now=when, include_expired=True)[0].id, expired.id)
            self.assertEqual(store.list(tier='archive', now=when)[0].id, old.id)
            self.assertEqual(store.path.read_bytes(), original)
            self.assertEqual(store.get(expired.id), expired)

    def test_old_schema_defaults(self):
        payload = process('Hello').to_dict()
        for name in ('due_at', 'task_status', 'related_task_id', 'last_accessed_at'):
            payload.pop(name)
        restored = MemoryRecord.from_dict(payload)
        self.assertIsNone(restored.due_at)
        self.assertIsNone(restored.task_status)


class SegmentationTests(unittest.TestCase):
    def test_multiple_intents_and_no_split_for_list_or_procedure(self):
        core = MemoryCore()
        incoming = MemoryInput(content='I prefer Python, and remind me to call Alex tomorrow.', user_id='u', session_id='s')
        records = core.process_many(incoming)
        self.assertEqual([r.category.value for r in records], ['preference', 'task'])
        self.assertEqual([r.tier.value for r in records], ['long_term', 'short_term'])
        self.assertEqual(len({r.id for r in records}), 2)
        self.assertTrue(all(r.source_metadata['source_content'] == incoming.content for r in records))
        for content in ('I like apples and oranges.', 'First run tests, then build and deploy.'):
            self.assertEqual(len(core.process_many(replace(incoming, content=content))), 1)
        with self.assertRaises(ValueError):
            core.process_many(replace(incoming, metadata={'task_id': 'x'}))

    def test_cli_split_and_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            store = str(Path(directory) / 'store.jsonl')
            output = io.StringIO()
            with redirect_stdout(output):
                status = cli_main(['--store', store, 'process', 'I prefer Python, and remind me to submit the report tomorrow.',
                                   '--user-id', 'u', '--session-id', 's', '--split'])
            self.assertEqual(status, 0)
            records = json.loads(output.getvalue())
            self.assertEqual(len(records), 2)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(['--store', store, 'process', 'I submitted the report.',
                                          '--user-id', 'u', '--session-id', 's']), 0)
            self.assertEqual(LocalMemoryStore(store).get(records[1]['id']).task_status, 'completed')


if __name__ == '__main__':
    unittest.main()
