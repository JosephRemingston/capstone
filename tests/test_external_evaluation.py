import unittest
from evaluation.benchmark import metrics, conversation_records
from evaluation.longmemeval import evaluate


class ExternalEvaluationTests(unittest.TestCase):
    def test_metric_denominators_and_duplicate_predictions(self):
        result = metrics(['x', 'b', 'b', 'a'], ['a', 'b', 'c'])
        self.assertAlmostEqual(result['recall_at_5'], 2 / 3)
        self.assertEqual(result['mrr_at_5'], .5)
        self.assertEqual(result['hit_at_5'], 1)
        self.assertEqual(metrics([], ['a'])['ndcg_at_5'], 0)
        with self.assertRaises(ValueError):
            metrics(['a'], [])

    def test_no_qa_or_summary_leakage_into_documents(self):
        row = {'sample_id': 'test', 'qa': [{'question': 'secret question', 'answer': 'secret answer'}],
               'observation': 'secret observation', 'session_summary': 'secret summary',
               'conversation': {'session_1_date_time': '1:00 pm on 1 January, 2026',
                   'session_1': [{'speaker': 'A', 'dia_id': 'D1:1', 'text': 'I live in Chennai'}]}}
        records, _ = conversation_records(row)
        self.assertEqual(len(records), 1)
        self.assertNotIn('secret', str(records[0].to_dict()))

    def test_external_retention_labels_do_not_change_processing(self):
        row = {'question_id': 'test', 'question_type': 'knowledge-update',
               'question_date': '2026/01/02 (Fri) 12:00', 'haystack_dates': ['2026/01/01 (Thu) 12:00'],
               'haystack_session_ids': ['s'], 'haystack_sessions': [[
                   {'role': 'user', 'content': 'I live in Chennai', 'has_answer': True}]]}
        first = evaluate(row)
        row['haystack_sessions'][0][0]['has_answer'] = False
        second = evaluate(row)
        for key in ('visible_roots', 'visible_text_bytes', 'superseded', 'consolidated'):
            self.assertEqual(first[key], second[key])
        self.assertEqual(first['gold_turns'], 1)
        self.assertEqual(second['gold_turns'], 0)
