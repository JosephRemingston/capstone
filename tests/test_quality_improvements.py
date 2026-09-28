from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from main import MemoryCore, MemoryInput, LocalMemoryStore
from main.domain.models import MemoryCategory, MemoryRecord, MemoryTier
from main.graph.temporal import TemporalGraph
from main.storage.indexes import IndexDatabase
from main.storage.cleanup import plan


NOW = datetime(2026, 9, 29, tzinfo=timezone.utc)


class QualityImprovementTests(unittest.TestCase):
    def process(self, text, **metadata):
        return MemoryCore().process(MemoryInput(
            content=text, user_id='u', session_id='s', timestamp=NOW, metadata=metadata))

    def test_useful_conversational_fact_is_retained(self):
        record = self.process('I am responsible for the Atlas migration.')
        self.assertEqual(record.category, MemoryCategory.SEMANTIC)
        self.assertEqual(record.tier, MemoryTier.LONG_TERM)
        self.assertIsNone(record.expires_at)

    def test_sensitive_values_are_redacted_before_record_creation(self):
        record = self.process('Email me at joseph@example.com; api_key=abcdefghijklmnop')
        self.assertNotIn('joseph@example.com', record.content)
        self.assertNotIn('abcdefghijklmnop', record.content)
        self.assertEqual(record.source_metadata['privacy']['types'], ['api_key', 'email'])
        parts = MemoryCore().process_many(MemoryInput(
            content='Email me at joseph@example.com. I live in Chennai.',
            user_id='u', session_id='s', timestamp=NOW))
        self.assertTrue(parts)
        self.assertTrue(all('joseph@example.com' not in str(part.source_metadata) for part in parts))

    def test_multi_sentence_conflict_can_be_processed_as_independent_observations(self):
        core = MemoryCore()
        incoming = MemoryInput(content='I live in Chennai. I live in Bengaluru.',
                               user_id='u', session_id='s', timestamp=NOW)
        records = core.process_many(incoming)
        self.assertEqual(len(records), 2)
        with tempfile.TemporaryDirectory() as directory:
            store = LocalMemoryStore(Path(directory) / 'store.jsonl')
            for record in records:
                store.ingest(record)
            self.assertEqual(store.list(user_id='u')[0].claim['value'], 'bengaluru')

    def test_explicit_transition_extracts_current_fact(self):
        record = self.process('I used to live in Chennai, but now I live in Bengaluru.')
        self.assertEqual(record.claim['value'], 'bengaluru')

    def test_paragraph_builds_multiple_graph_relationships(self):
        record = self.process('Alice works on Atlas. Atlas is owned by Acme.')
        with tempfile.TemporaryDirectory() as directory:
            graph = TemporalGraph(IndexDatabase(Path(directory) / 'index.sqlite3'))
            graph.sync([record], user_id='u')
            predicates = {edge['predicate'] for edge in graph.relations(user_id='u', as_of=NOW, known_at=NOW)}
            self.assertEqual(predicates, {'works_on', 'owned_by'})

    def test_related_wording_consolidates(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalMemoryStore(Path(directory) / 'store.jsonl')
            first = self.process('We use Python for data processing.')
            second = MemoryCore().process(MemoryInput(
                content='Our team uses Python for data processing.', user_id='u', session_id='s2',
                timestamp=NOW + timedelta(minutes=1)))
            store.ingest(first)
            store.ingest(second)
            self.assertEqual(second.consolidated_into, first.id)
            self.assertIn('Related observations:', store.get(first.id).summary)

    def test_balanced_cleanup_archives_useful_expired_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalMemoryStore(Path(directory) / 'store.jsonl')
            useful = MemoryRecord(content='Durable fact', user_id='u', session_id='s',
                                  category=MemoryCategory.SEMANTIC,
                                  tier=MemoryTier.SHORT_TERM, importance_score=.7,
                                  created_at=NOW - timedelta(days=30), updated_at=NOW - timedelta(days=30),
                                  expires_at=NOW - timedelta(days=20))
            disposable = MemoryRecord(content='hello', user_id='u', session_id='s',
                                      category=MemoryCategory.TEMPORARY,
                                      created_at=NOW - timedelta(days=30), updated_at=NOW - timedelta(days=30),
                                      expires_at=NOW - timedelta(days=20))
            store.save(useful)
            store.save(disposable)
            result, _ = plan(store, user_id='u', now=NOW)
            self.assertEqual(result['archived_ids'], [useful.id])
            self.assertEqual(result['removed_ids'], [disposable.id])


if __name__ == '__main__':
    unittest.main()
