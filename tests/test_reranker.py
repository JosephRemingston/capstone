from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import math
import unittest

from main.application.core import MemoryCore
from main.domain.models import MemoryInput, MemoryTier
from main.retrieval.rag import MemoryRAG, graph_predicates
from main.storage.store import LocalMemoryStore


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class Embedder:
    model_id = 'test-embedder'
    dimension = 2

    def passages(self, texts):
        return [self.query(text) for text in texts]

    def query(self, text):
        return [1.0, 1.0 if 'target' in text.lower() else 0.0]


class FakeReranker:
    model_id = 'test-reranker'

    def rerank(self, query, documents):
        return [10.0 if 'target' in text.lower() else float(index)
                for index, text in enumerate(documents)]


class BrokenReranker:
    model_id = 'broken-reranker'

    def rerank(self, query, documents):
        raise RuntimeError('private failure details')


class InvalidReranker:
    model_id = 'invalid-reranker'

    def rerank(self, query, documents):
        return [math.nan for _ in documents]


class EqualReranker:
    model_id = 'equal-reranker'

    def rerank(self, query, documents):
        return [1.0 for _ in documents]


class RerankerPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = LocalMemoryStore(Path(temporary.name) / 'memories.jsonl')

    def add(self, text, *, user='u', session='s', identifier=None):
        record = MemoryCore().process(MemoryInput(
            content=text, user_id=user, session_id=session, timestamp=NOW))
        record.tier = MemoryTier.LONG_TERM
        record.expires_at = None
        if identifier:
            record.id = identifier
        return self.store.ingest(record)

    def rag(self, reranker):
        return MemoryRAG(self.store, embedder=Embedder(), reranker=reranker)

    def test_reranker_changes_top_k_and_reports_candidate_pool(self):
        first = self.add('alpha ordinary evidence', identifier='a-first')
        target = self.add('alpha target evidence', identifier='z-target')
        self.add('alpha additional evidence', identifier='b-additional')
        rag = self.rag(FakeReranker())
        fused = rag.search('alpha', user_id='u', mode='hybrid', limit=1, rerank=False)
        reranked = rag.search('alpha', user_id='u', mode='hybrid', limit=1)
        self.assertEqual(reranked[0]['memory_id'], target.id)
        self.assertNotEqual(fused[0]['memory_id'], target.id)
        self.assertEqual(reranked[0]['candidate_pool_size'], 3)
        self.assertEqual(reranked[0]['reranker']['status'], 'applied')
        self.assertEqual(reranked[0]['reranker']['model'], 'test-reranker')
        self.assertEqual(reranked[0]['text'], target.content)
        self.assertEqual(reranked[0]['score'], reranked[0]['fusion_score'])
        self.assertIsNotNone(reranked[0]['reranker_score'])
        self.assertIn(first.id, {row.id for row in self.store.all()})

    def test_parallel_channels_and_graph_provenance_survive_reranking(self):
        relation = self.add('Alice reports to Bob')
        rag = self.rag(FakeReranker())
        hit = rag.search('Who does Alice report to?', user_id='u', mode='hybrid', limit=1)[0]
        self.assertEqual(hit['memory_id'], relation.id)
        self.assertEqual(set(hit['signals']), {'keyword', 'semantic', 'graph'})
        self.assertTrue(hit['paths'])

    def test_failure_falls_back_without_leaking_message(self):
        self.add('alpha one')
        self.add('alpha two')
        rag = self.rag(BrokenReranker())
        expected = rag.search('alpha', user_id='u', mode='hybrid', limit=2, rerank=False)
        actual = rag.search('alpha', user_id='u', mode='hybrid', limit=2)
        self.assertEqual([h['memory_id'] for h in actual], [h['memory_id'] for h in expected])
        self.assertTrue(all(h['reranker']['status'] == 'fallback' for h in actual))
        self.assertTrue(all(h['reranker']['error_type'] == 'RuntimeError' for h in actual))
        self.assertNotIn('private failure details', str(actual))

    def test_invalid_scores_fall_back_and_none_can_disable_model(self):
        self.add('alpha evidence')
        invalid = self.rag(InvalidReranker()).search('alpha', user_id='u', mode='hybrid', limit=1)
        self.assertEqual(invalid[0]['reranker']['status'], 'fallback')
        disabled = self.rag(None).search('alpha', user_id='u', mode='hybrid', limit=1, rerank=False)
        self.assertEqual(disabled[0]['reranker']['status'], 'disabled')

    def test_equal_reranker_scores_use_fusion_then_id_deterministically(self):
        self.add('alpha first evidence', identifier='z-memory')
        self.add('alpha second evidence', identifier='a-memory')
        rag = self.rag(EqualReranker())
        fused = rag.search('alpha', user_id='u', mode='hybrid', limit=2, rerank=False)
        first = rag.search('alpha', user_id='u', mode='hybrid', limit=2)
        second = rag.search('alpha', user_id='u', mode='hybrid', limit=2)
        expected = sorted(fused, key=lambda hit: (-hit['fusion_score'], hit['memory_id']))
        self.assertEqual([hit['memory_id'] for hit in first],
                         [hit['memory_id'] for hit in expected])
        self.assertEqual([hit['memory_id'] for hit in first],
                         [hit['memory_id'] for hit in second])

    def test_bounds_filters_and_nonhybrid_default(self):
        own = self.add('alpha evidence')
        self.add('alpha evidence', user='other')
        rag = self.rag(FakeReranker())
        for value in (0, 201, True, 0.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rag.search('alpha', user_id='u', mode='hybrid', limit=1, candidate_limit=value)
        with self.assertRaises(ValueError):
            rag.search('alpha', user_id='u', mode='hybrid', limit=5, candidate_limit=4)
        hit = rag.search('alpha', user_id='u', mode='keyword', limit=1)[0]
        self.assertEqual(hit['memory_id'], own.id)
        self.assertEqual(hit['reranker']['status'], 'disabled')

    def test_context_uses_reranked_winning_snippet(self):
        self.add('alpha ordinary evidence')
        target = self.add('alpha target evidence')
        context = self.rag(FakeReranker()).context(
            'alpha', user_id='u', mode='hybrid', limit=1, budget=512)
        self.assertEqual(list(context['sources']), [target.id])
        self.assertIn('target evidence', context['text'])

    def test_graph_query_predicates_do_not_treat_unrelated_questions_as_relational(self):
        self.assertEqual(graph_predicates('What milestone did Alice complete?'), set())
        self.assertEqual(graph_predicates('Where is the company Alice works for based?'),
                         {'works_at', 'based_in'})
        self.assertEqual(graph_predicates('Who does Alice report to?'), {'reports_to'})


if __name__ == '__main__':
    unittest.main()
