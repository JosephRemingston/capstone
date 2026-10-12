from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation import ablation, classification, lifecycle, locomo, retention
from evaluation.schema import DATASET_PATH, GOLD_USER
from evaluation.systems import Protocol, resolve_pool
from main.domain.models import MemoryCategory, MemoryRecord, MemoryTier
from main.retrieval.rag import MemoryRAG
from main.storage.store import LocalMemoryStore

from tests._evaluation_fakes import HashEmbedder, OverlapReranker


class AblationRunnerTests(unittest.TestCase):
    def test_small_run_reports_every_arm_with_full_visibility(self):
        report = ablation.run(dataset=DATASET_PATH, split='dev', protocol=Protocol(candidate_limit=20),
                              arms=['vector', 'hybrid_full'], limit_queries=6,
                              embedder=HashEmbedder(), reranker=OverlapReranker())
        self.assertEqual(report['dataset']['visible_after_load'], report['dataset']['memories'])
        self.assertEqual(len(report['rows']), 12)
        self.assertTrue(all(row['pool_size'] <= 20 for row in report['rows']))
        self.assertEqual(report['protocol']['resolved_pool'], {'vector': 20, 'hybrid_full': 20})
        self.assertEqual(report['summary']['hybrid_full']['vector_paths'], {'exact': 6})
        self.assertFalse(report['protocol']['ann'])
        self.assertIn('Recall@5', ablation.markdown(report))
        hybrid = [row for row in report['rows'] if row['arm'] == 'hybrid_full']
        self.assertTrue(all(row['reranker_status'] in {'applied', 'empty'} for row in hybrid))

    def test_default_pool_resolution(self):
        self.assertEqual(resolve_pool(5, 20), 20)
        self.assertEqual(resolve_pool(5, None), 100)
        self.assertEqual(resolve_pool(5, None, rerank=False), 20)


class LocomoRunnerTests(unittest.TestCase):
    def test_only_text_turns_are_indexed(self):
        row = {'sample_id': 'conv-x', 'qa': [{'question': 'secret question?', 'answer': 'secret answer', 'evidence': ['D1:1']}],
               'observation': {'session_1': 'OBSERVATION LEAK'}, 'session_summary': {'session_1': 'SUMMARY LEAK'},
               'conversation': {'session_1': [{'dia_id': 'D1:1', 'speaker': 'A', 'text': 'I adopted a cat.'},
                                              {'dia_id': 'D1:2', 'speaker': 'B', 'img_url': ['x.jpg']}],
                                'session_1_date_time': '1:56 pm on 8 May, 2023', 'speaker_a': 'A', 'speaker_b': 'B'}}
        records, image_only = locomo.conversation_records(row)
        self.assertEqual([r.id for r in records], ['D1:1'])
        self.assertEqual(image_only, {'D1:2'})
        text = json.dumps([r.to_dict() for r in records])
        for leak in ('secret', 'LEAK'):
            self.assertNotIn(leak, text)

    def test_score_denominators(self):
        result = locomo.score(['a', 'b', 'a'], ['a', 'c'], ['a', 'b'])
        self.assertEqual(result['recall_at_5'], 0.5)
        self.assertEqual(result['mrr_at_5'], 1.0)
        self.assertEqual(result['pool_recall'], 0.5)
        self.assertIsNone(locomo.score(['a'], ['a'], None)['pool_hit'])


class RetentionRunnerTests(unittest.TestCase):
    ROW = {'question_id': 'q1', 'question_type': 'knowledge-update', 'question': 'Where do I live?',
           'question_date': '2023/06/01 (Thu) 10:00', 'haystack_dates': ['2023/05/01 (Mon) 09:00'],
           'haystack_session_ids': ['s1'],
           'haystack_sessions': [[{'role': 'user', 'content': 'I live in Lisbon.', 'has_answer': True},
                                  {'role': 'assistant', 'content': 'Noted!'}]]}

    def test_labels_score_but_never_enter_memory(self):
        result = retention.replay(self.ROW)
        self.assertEqual(result['gold_turns'], 1)
        self.assertEqual(result['keep_everything_retention'], 1.0)
        self.assertNotIn('has_answer', json.dumps(result))

    def test_future_sessions_are_rejected(self):
        future = {**self.ROW, 'haystack_dates': ['2023/07/01 (Sat) 09:00']}
        self.assertFalse(retention.chronology_valid(future))
        with self.assertRaises(ValueError):
            retention.replay(future)


class SmallRunnerTests(unittest.TestCase):
    def test_lifecycle_checks_all_pass(self):
        report = lifecycle.evaluate()
        self.assertEqual(report['passed'], report['checks'])

    def test_classification_cases_are_balanced(self):
        cases = classification.load_cases()
        self.assertEqual(len(cases), 60)
        self.assertEqual({c['expected_category'] for c in cases},
                         {'semantic', 'episodic', 'procedural', 'preference', 'task', 'temporary'})
        macro, _ = classification.macro_f1([('a', 'a'), ('b', 'b')])
        self.assertEqual(macro, 1.0)


class RecencyModeRegressionTests(unittest.TestCase):
    def test_recency_mode_returns_newest_first(self):
        with TemporaryDirectory() as directory:
            store = LocalMemoryStore(Path(directory) / 'store.jsonl')
            for day in (1, 3, 2):
                when = datetime(2026, 1, day, tzinfo=timezone.utc)
                store.save(MemoryRecord(id=f'm{day}', content=f'note {day}', user_id='u', session_id='s',
                                        category=MemoryCategory.SEMANTIC, tier=MemoryTier.LONG_TERM,
                                        created_at=when, updated_at=when))
            rag = MemoryRAG(store, embedder=HashEmbedder(), reranker=None, ann_enabled=False)
            hits = rag.search('anything', user_id='u', mode='recency', limit=3)
            self.assertEqual([hit['memory_id'] for hit in hits], ['m3', 'm2', 'm1'])
            with self.assertRaises(ValueError):
                rag.search('anything', user_id='u', mode='hybrid', channels=('recency',))


if __name__ == '__main__':
    unittest.main()
