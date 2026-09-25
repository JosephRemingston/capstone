import unittest
from evaluation.evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def test_developer_labels_are_reported_without_claiming_human_review(self):
        report = evaluate([{'id':'example', 'content':'Hello', 'expected_category':'temporary',
                            'expected_tier':'working', 'importance_range':[0,.25],
                            'label_source':'developer_authored'}])
        self.assertEqual(report['label_sources'], {'developer_authored':1})
        self.assertEqual(report['category_accuracy'], 1)
        self.assertEqual(report['tier_accuracy'], 1)

    def test_empty_review_set_and_invalid_labels_fail(self):
        with self.assertRaisesRegex(ValueError, 'No eligible labels'):
            evaluate([])
        with self.assertRaises(ValueError):
            evaluate([{'id':'bad', 'content':'Hello', 'expected_category':'made-up',
                       'expected_tier':'working', 'importance_range':[0,1], 'label_source':'developer_authored'}])
