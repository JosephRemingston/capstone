from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from main.storage.cleanup import cleanup
from main.domain.models import MemoryCategory, MemoryRecord
from main.storage.store import LocalMemoryStore
from main.retrieval.rag import MemoryRAG
from main.storage.locking import store_lock

NOW = datetime(2026, 9, 27, tzinfo=timezone.utc)


def concurrent_writer(path, ready):
    ready.set()
    LocalMemoryStore(path).save(MemoryRecord('retained concurrent write', 'u', 's', MemoryCategory.SEMANTIC, id='concurrent'))


class CleanupFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = LocalMemoryStore(Path(self.tmp.name) / 'memories.jsonl')

    def add(self, identifier, **kwargs):
        record = MemoryRecord('sensitive ' + identifier, 'u', 's', MemoryCategory.TEMPORARY,
                              id=identifier, expires_at=NOW - timedelta(days=10), **kwargs)
        return self.store.save(record)


class CleanupTests(CleanupFixture):
    def test_preview_scope_grace_and_idempotency(self):
        self.add('expired')
        self.store.save(MemoryRecord('other', 'v', 's', MemoryCategory.TEMPORARY, id='other', expires_at=NOW-timedelta(days=20)))
        self.store.save(MemoryRecord('recent', 'u', 's', MemoryCategory.TEMPORARY, id='recent', expires_at=NOW-timedelta(days=1)))
        before = self.store.path.read_bytes()
        preview = cleanup(self.store, user_id='u', now=NOW)
        self.assertEqual(preview['removed_ids'], ['expired'])
        self.assertEqual(before, self.store.path.read_bytes())
        result = cleanup(self.store, user_id='u', now=NOW, apply=True)
        self.assertEqual(result['removed_memories'], 1)
        self.assertEqual({r.id for r in self.store.all()}, {'other', 'recent'})
        self.assertNotIn(b'sensitive expired', self.store.path.read_bytes())
        again = cleanup(self.store, user_id='u', now=NOW, apply=True)
        self.assertEqual(again['removed_memories'], 0)
        self.assertEqual(again['bytes_reclaimed'], 0)

    def test_reference_closure_hold_and_open_tasks(self):
        self.add('evidence')
        self.add('intermediate', evidence_ids=['evidence'])
        self.store.save(MemoryRecord('root', 'u', 's', MemoryCategory.SEMANTIC, id='root', evidence_ids=['intermediate']))
        self.add('held', source_metadata={'legal_hold': True})
        task = self.add('open')
        self.store.save(replace(task, category=MemoryCategory.TASK, task_status='active'))
        self.add('remove')
        report = cleanup(self.store, user_id='u', now=NOW, apply=True)
        self.assertEqual(report['removed_ids'], ['remove'])
        self.assertEqual(report['protected_referenced_expired'], 2)
        self.assertIsNotNone(self.store.get('evidence'))

    def test_lossless_revision_compaction_and_graph(self):
        record = self.store.save(MemoryRecord('Alice reports to Bob', 'u', 's', MemoryCategory.SEMANTIC, id='fact'))
        # An actual byte-identical retry is redundant.
        with self.store.path.open('ab') as handle:
            handle.write(self.store.path.read_bytes())
        self.store.save(replace(record, content='Alice reports to Carol', updated_at=NOW))
        self.store.save(replace(record, updated_at=NOW + timedelta(days=1)))
        before = [r.to_dict() for r in self.store.history('fact', user_id='u')]
        report = cleanup(self.store, user_id='u', now=NOW, apply=True)
        self.assertEqual(report['duplicate_revisions_removed'], 1)
        self.assertEqual([r.to_dict() for r in self.store.history('fact', user_id='u')], [before[0], before[2], before[3]])
        self.assertEqual(self.store.get('fact').content, record.content)

    def test_custom_and_default_indexes_are_purged(self):
        record = self.add('secret')
        for path in (None, Path(self.tmp.name) / 'custom.sqlite3'):
            rag = MemoryRAG(self.store, index_path=path)
            rag.sync(user_id='u', semantic=False)
            with rag.database.connect() as db:
                db.execute('INSERT INTO vectors VALUES (?,?,?,?,?,?,?,?,?,?)',
                           ('u','test','secret:0','secret',record.content,0,len(record.content),'x',1,b'0000'))
        result = cleanup(self.store, user_id='u', now=NOW, apply=True)
        self.assertEqual(len(result['indexes_invalidated']), 2)
        for name in result['indexes_invalidated']:
            self.assertNotIn(record.content.encode(), Path(name).read_bytes())
        self.assertEqual(self.store.all(), [])

    def test_failure_keeps_source_and_schedule_retries(self):
        self.add('expired')
        before = self.store.path.read_bytes()
        with patch('main.storage.cleanup.atomic_write', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                cleanup(self.store, user_id='u', now=NOW, apply=True, scheduled=True)
        self.assertEqual(before, self.store.path.read_bytes())
        self.assertFalse(Path(str(self.store.path) + '.cleanup-schedule.json').exists())
        result = cleanup(self.store, user_id='u', now=NOW, apply=True, scheduled=True)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(cleanup(self.store, user_id='u', now=NOW, apply=True, scheduled=True)['status'], 'not_due')
        self.assertEqual(cleanup(self.store, user_id='u', now=NOW+timedelta(days=1), apply=True, scheduled=True)['status'], 'completed')

    def test_corrupt_or_unknown_data_fails_closed(self):
        self.store.path.write_text('{broken\n')
        with self.assertRaises(ValueError):
            cleanup(self.store, user_id='u', now=NOW, apply=True)
        self.assertEqual(self.store.path.read_text(), '{broken\n')
        self.store.path.unlink()
        record = self.add('x')
        self.store.path.write_text(json.dumps({**record.to_dict(), 'future_field':'preserve'})+'\n')
        with self.assertRaisesRegex(ValueError, 'Unknown record fields'):
            cleanup(self.store, user_id='u', now=NOW, apply=True)

    def test_writer_waits_and_is_not_lost(self):
        self.add('expired')
        context = multiprocessing.get_context('spawn')
        ready = context.Event()
        process = context.Process(target=concurrent_writer, args=(str(self.store.path), ready))
        with store_lock(self.store.path):
            process.start()
            self.assertTrue(ready.wait(5))
            self.assertIsNone(self.store.get('concurrent'))
            cleanup(self.store, user_id='u', now=NOW, apply=True)
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join()
            self.fail('Writer did not finish')
        self.assertEqual(process.exitcode, 0)
        self.assertEqual([r.id for r in self.store.all()], ['concurrent'])

class CleanupFailureTests(CleanupFixture):
    def test_failed_index_purge_never_replaces_source(self):
        self.add('expired')
        before = self.store.path.read_bytes()
        with patch('main.storage.cleanup.purge_indexes', side_effect=ValueError('busy index')):
            with self.assertRaises(ValueError):
                cleanup(self.store, user_id='u', now=NOW, apply=True, scheduled=True)
        self.assertEqual(self.store.path.read_bytes(), before)
        self.assertFalse(Path(str(self.store.path) + '.cleanup-schedule.json').exists())

    def test_expired_cycle_removal_and_historical_reference_protection(self):
        self.add('a', evidence_ids=['b'])
        self.add('b', evidence_ids=['a'])
        self.add('old')
        root = self.store.save(MemoryRecord('root', 'u', 's', MemoryCategory.SEMANTIC, id='root', evidence_ids=['old']))
        self.store.save(replace(root, evidence_ids=[]))
        result = cleanup(self.store, user_id='u', now=NOW, apply=True)
        self.assertEqual(result['removed_ids'], ['a', 'b'])
        self.assertIsNotNone(self.store.get('old'))

    def test_invalid_registry_and_schedule_fail_without_deleting_history(self):
        self.add('expired')
        before = self.store.path.read_bytes()
        registry = Path(str(self.store.path.resolve()) + '.indexes.json')
        registry.write_text('{"invalid": "registry"}')
        with self.assertRaisesRegex(ValueError, 'registry'):
            cleanup(self.store, user_id='u', now=NOW, apply=True)
        self.assertEqual(self.store.path.read_bytes(), before)
        registry.unlink()
        state = Path(str(self.store.path.resolve()) + '.cleanup-schedule.json')
        state.write_text('[]')
        with self.assertRaisesRegex(ValueError, 'schedule'):
            cleanup(self.store, user_id='u', now=NOW, scheduled=True, apply=True)
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_invalid_policy_and_preview_do_not_advance_scheduler(self):
        self.add('expired')
        for value in (float('nan'), float('inf'), -1, 1e300):
            with self.assertRaises(ValueError):
                cleanup(self.store, user_id='u', now=NOW, grace_days=value, apply=True)
        cleanup(self.store, user_id='u', now=NOW, scheduled=True)
        self.assertFalse(Path(str(self.store.path) + '.cleanup-schedule.json').exists())

    def test_compaction_preserves_temporal_query_evidence(self):
        from main.application.core import MemoryCore
        from main.domain.models import MemoryInput
        first = NOW - timedelta(days=30)
        second = NOW - timedelta(days=20)
        for when, text in ((first, 'Alice reports to Bob'), (second, 'Alice reports to Carol')):
            record = MemoryCore().process(MemoryInput(text, 'u', 's', timestamp=when))
            record.expires_at = None
            with patch('main.storage.store.utc_now', return_value=when):
                self.store.ingest(record)
        raw = self.store.path.read_bytes().splitlines(keepends=True)
        with self.store.path.open('ab') as stream:
            stream.write(raw[-1])
        rag = MemoryRAG(self.store)
        def snapshot():
            rag.sync(user_id='u', semantic=False)
            return [[(e['subject_label'], e['predicate'], e['object_label'], e['source_revision'], e['evidence_ids'])
                     for e in rag.graph.relations(user_id='u', as_of=valid, known_at=NOW)]
                    for valid in (first + timedelta(days=1), second + timedelta(days=1))]
        before = snapshot()
        result = cleanup(self.store, user_id='u', now=NOW, apply=True)
        self.assertEqual(result['duplicate_revisions_removed'], 1)
        self.assertEqual(snapshot(), before)

    def test_worker_once_returns_failure_and_retries_sqlite_errors(self):
        import sqlite3
        from main.application.maintenance import main
        with patch('main.application.maintenance.signal.signal'), patch('sys.argv', ['maintenance', '--user-id', 'u', '--once']), \
             patch('main.application.maintenance.cleanup', side_effect=sqlite3.OperationalError('busy')), \
             patch('builtins.print'):
            self.assertEqual(main(), 1)
        with patch('main.application.maintenance.signal.signal'), patch('sys.argv', ['maintenance', '--user-id', 'u', '--once']), \
             patch('main.application.maintenance.cleanup', return_value={'status': 'completed'}), \
             patch('builtins.print'):
            self.assertEqual(main(), 0)


if __name__ == '__main__':
    unittest.main()
