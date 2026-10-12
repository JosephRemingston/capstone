import unittest

from evaluation.metrics import (answerability, contains_phrase, context_metrics, correctness, normalize,
                                percentiles, pool_metrics, retrieval_metrics)
from evaluation.stats import cluster_bootstrap, paired_comparison


def row(**overrides):
    base = {'query_id': 'q', 'unanswerable': False, 'expected_answer': 'Porto', 'answer_aliases': ['Oporto'],
            'supporting_memory_ids': ['p01-home'], 'graded_relevance': {'p01-home': 3, 'p01-old': 1}}
    base.update(overrides)
    return base


class RetrievalMetricTests(unittest.TestCase):
    def test_binary_and_graded_metrics(self):
        result = retrieval_metrics(['p01-home', 'x', 'p01-old'], {'p01-home': 3, 'p01-old': 1})
        self.assertEqual(result['recall_at_5'], 1.0)
        self.assertEqual(result['hit_at_5'], 1.0)
        self.assertEqual(result['mrr'], 1.0)
        self.assertLess(result['ndcg_at_5'], 1.0)
        self.assertEqual(retrieval_metrics(['x', 'p01-home'], {'p01-home': 3})['mrr'], 0.5)

    def test_unanswerable_rows_have_no_retrieval_metrics(self):
        self.assertTrue(all(value is None for value in retrieval_metrics(['x'], {}).values()))
        self.assertEqual(pool_metrics(['x'], {}), {'pool_hit': None, 'pool_recall': None})

    def test_duplicate_predictions_count_once(self):
        self.assertEqual(retrieval_metrics(['a', 'a', 'a'], {'a': 1, 'b': 1})['recall_at_5'], 0.5)

    def test_pool_metrics(self):
        self.assertEqual(pool_metrics(['a', 'c'], {'a': 3, 'b': 1}), {'pool_hit': 1.0, 'pool_recall': 0.5})


class AnswerMetricTests(unittest.TestCase):
    def test_phrase_matching_is_whole_token(self):
        self.assertEqual(normalize("Don't own a CAR!"), 'don t own a car')
        self.assertTrue(contains_phrase('It moved to Thursdays at 3', 'Thursdays'))
        self.assertFalse(contains_phrase('It moved to Thursdays at 3', 'Thursday'))
        self.assertFalse(contains_phrase('anything', ''))

    def test_wrong_persona_snippet_does_not_count(self):
        sources = {'p02-home': {'text': 'speaker: Jonas\nWe moved to Porto last year.'}}
        self.assertEqual(correctness(row(), sources, None, False)['correct_grounded'], 0.0)

    def test_grounded_and_top1_correctness(self):
        sources = {'p01-old': {'text': 'I lived in Lisbon.'}, 'p01-home': {'text': 'We finally moved to Porto.'}}
        top = {'memory_id': 'p01-old', 'text': 'I lived in Lisbon.'}
        result = correctness(row(), sources, top, False)
        self.assertEqual(result, {'correct_grounded': 1.0, 'correct_top1': 0.0, 'correct_first': 0.0})
        reordered = {'p01-home': sources['p01-home'], 'p01-old': sources['p01-old']}
        self.assertEqual(correctness(row(), reordered, top, False)['correct_first'], 1.0)
        self.assertEqual(correctness(row(), sources, top, True)['correct_grounded'], 0.0)

    def test_unanswerable_correct_only_when_abstaining(self):
        unanswerable = row(unanswerable=True, expected_answer=None, answer_aliases=[], supporting_memory_ids=[],
                           graded_relevance={})
        self.assertEqual(correctness(unanswerable, {}, None, True)['correct_grounded'], 1.0)
        self.assertEqual(correctness(unanswerable, {'x': {'text': 'y'}}, None, False)['correct_grounded'], 0.0)

    def test_context_metrics(self):
        self.assertEqual(context_metrics(row(), ['p01-home', 'x']), {'context_precision': 0.5, 'context_recall': 0.5})

    def test_answerability_confusion(self):
        rows = [{'abstain': False, 'unanswerable': False}, {'abstain': True, 'unanswerable': False},
                {'abstain': True, 'unanswerable': True}, {'abstain': False, 'unanswerable': True},
                {'abstain': True, 'unanswerable': True}, {'abstain': False, 'unanswerable': False, 'status': 'failed'}]
        result = answerability(rows)
        self.assertEqual(result['answerable_accept_rate'], 0.5)
        self.assertAlmostEqual(result['unanswerable_refusal_rate'], 2 / 3)
        self.assertAlmostEqual(result['balanced_accuracy'], (0.5 + 2 / 3) / 2)
        self.assertEqual(result['generation_failures'], 1)


class StatisticsTests(unittest.TestCase):
    def test_paired_comparison_is_deterministic_and_ordered(self):
        left, right = [0, 0, 1, 0, 1, 0, 0, 1], [1, 0, 1, 1, 1, 0, 1, 1]
        first, second = paired_comparison(left, right), paired_comparison(left, right)
        self.assertEqual(first, second)
        self.assertEqual(first['mean_difference'], 0.375)
        self.assertLessEqual(first['ci95'][0], first['mean_difference'])
        self.assertLessEqual(first['mean_difference'], first['ci95'][1])

    def test_none_pairs_are_dropped_and_identical_lists_are_null(self):
        result = paired_comparison([None, 1, 0], [1, 1, 0])
        self.assertEqual(result['pairs'], 2)
        self.assertEqual(result['mean_difference'], 0)
        self.assertGreater(result['p_value'], 0.9)
        with self.assertRaises(ValueError):
            paired_comparison([1], [1, 2])

    def test_cluster_bootstrap_and_percentiles(self):
        low, high = cluster_bootstrap({'a': [0.0, 1.0], 'b': [1.0], 'c': [0.5]})
        self.assertLessEqual(low, high)
        self.assertEqual(percentiles([1, 2, 3, 4, 5])['p50'], 3)


if __name__ == '__main__':
    unittest.main()
