from pathlib import Path
import unittest

from evaluation.analysis.statistics import paired_comparison, percentiles
from evaluation.datasets.build_gold import QUERY_TYPES, build
from evaluation.human import score as score_human_reviews
from evaluation.lifecycle import evaluate as evaluate_lifecycle
from evaluation.retrieval.metrics import retrieval_metrics
from evaluation.run import load_dataset


class AblationEvaluationTests(unittest.TestCase):
    def test_frozen_dataset_schema_and_coverage(self):
        payload = load_dataset(Path('evaluation/datasets/gold_queries.json'))
        self.assertEqual(len(payload['queries']), 240)
        self.assertEqual(len(payload['memories']), 240)
        counts = {kind: sum(row['query_type'] == kind for row in payload['queries'])
                  for kind in QUERY_TYPES}
        self.assertEqual(set(counts.values()), {20})
        self.assertEqual(build(), payload)
        lifecycle = evaluate_lifecycle()
        self.assertEqual((lifecycle['passed'], lifecycle['checks']), (14, 14))

    def test_binary_and_graded_metrics(self):
        result = retrieval_metrics(['b', 'a', 'x'], {'a': 3, 'b': 1})
        self.assertEqual(result['recall_at_5'], 1.0)
        self.assertEqual(result['hit_at_10'], 1.0)
        self.assertEqual(result['mrr'], 1.0)
        self.assertLess(result['ndcg_at_5'], 1.0)
        empty = retrieval_metrics([], {})
        self.assertIsNone(empty['recall_at_5'])

    def test_latency_percentiles_and_paired_statistics(self):
        self.assertEqual(percentiles([1, 2, 3, 4])['p95'], 4)
        rows = [
            {'query_id': '1', 'system': 'a', 'score': 0.0},
            {'query_id': '1', 'system': 'b', 'score': 1.0},
            {'query_id': '2', 'system': 'a', 'score': 0.5},
            {'query_id': '2', 'system': 'b', 'score': 1.0},
        ]
        result = paired_comparison(rows, 'a', 'b', 'score', samples=100)
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['absolute_improvement'], .75)
        self.assertEqual(result, paired_comparison(list(reversed(rows)), 'a', 'b', 'score', samples=100))

    def test_human_reviews_require_real_complete_ratings(self):
        rows = [{'review_id': f'r{question}-{system}', 'query_id': f'q{question}',
                 'system_label': f'System {system}', 'reviewer': 'reviewer-1',
                 'reviewed_at': '2026-01-01T00:00:00+00:00',
                 'scores': {'correctness': 4, 'relevance': 3, 'faithfulness': 4,
                            'citation_usefulness': 3, 'temporal_correctness': 4}}
                for question in range(50) for system in 'ABCD']
        result = score_human_reviews(rows)
        self.assertEqual(result['means_0_to_4']['System A']['correctness'], 4)
        rows[0]['scores']['correctness'] = None
        with self.assertRaises(ValueError):
            score_human_reviews(rows)


if __name__ == '__main__':
    unittest.main()
