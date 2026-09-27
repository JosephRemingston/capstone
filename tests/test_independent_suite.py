from copy import deepcopy
from datetime import datetime, timezone
import unittest

from main.indexes import digest
from tests.evaluation.personalization import score_reviews, records_for, RUBRIC
from tests.evaluation.suite import report


class IndependentSuiteTests(unittest.TestCase):
    def fixture(self):
        review = {'answer_id': 'a', 'question': 'What should I drink?', 'reference_answer': 'Tea',
                  'response': {'answer': 'Tea'}, 'retrieved_evidence': {'s': {'text': 'I prefer tea'}},
                  'reviewer': 'independent-test-fixture', 'reviewed_at': '2026-01-01T00:00:00Z',
                  'independent': True, 'scores': {'preference_alignment': 2, 'groundedness': 2,
                                                'usefulness': 1, 'conflict_correctness': None}}
        review['binding'] = digest({'id':'a','question':review['question'],'reference':review['reference_answer'],
                                    'response':review['response'],'sources':review['retrieved_evidence']})
        report = {'results': [{'answer_id':'a','binding':review['binding'],'status':'answered',
                              'type':'single-session-preference','mode':'hybrid'}]}
        return report, review

    def test_blank_partial_stale_and_nonindependent_reviews_rejected(self):
        report, review = self.fixture()
        for mutate in (lambda r:r.update(reviewer=''), lambda r:r.update(independent=False),
                       lambda r:r.update(binding='stale'), lambda r:r.update(response={'answer':'Altered'}),
                       lambda r:r['scores'].update(usefulness=None),
                       lambda r:r['scores'].update(groundedness=True),
                       lambda r:r.update(reviewed_at='2099-01-01T00:00:00Z')):
            changed = deepcopy(review)
            mutate(changed)
            with self.assertRaises(ValueError):
                score_reviews(report, [changed])
        with self.assertRaises(ValueError):
            score_reviews(report, [])
        with self.assertRaises(ValueError):
            score_reviews(report, [review, review])

    def test_blind_score_aggregation(self):
        report, review = self.fixture()
        second = deepcopy(review)
        second['reviewer'] = 'second-test-fixture'
        second['scores']['usefulness'] = 2
        result = score_reviews(report, [review, second])
        self.assertEqual(result['mean_scores_0_to_2']['hybrid']['usefulness'], 1.5)
        self.assertEqual(result['disagreement_fraction'], 1/3)

    def test_labels_never_enter_retrieval_documents(self):
        row = {'question_id':'q','answer':'SECRET GOLD', 'question':'SECRET QUERY',
               'haystack_dates':['2026/01/01 (Thu) 12:00'], 'haystack_session_ids':['s'],
               'haystack_sessions':[[{'content':'I like tea','role':'user','has_answer':True}]]}
        records, gold = records_for(row)
        self.assertEqual(gold, {'s:0'})
        self.assertNotIn('SECRET', str(records[0].to_dict()))
        self.assertNotIn('has_answer', str(records[0].to_dict()))

    def test_future_sessions_are_rejected_before_scoring_or_context(self):
        from tests.evaluation.longmemeval import chronology_valid, evaluate
        row = {'question_id': 'q', 'question_date': '2026/01/01 (Thu) 12:00',
               'haystack_dates': ['2026/01/02 (Fri) 12:00']}
        self.assertFalse(chronology_valid(row))
        with self.assertRaises(ValueError):
            evaluate(row)
        with self.assertRaises(ValueError):
            records_for(row)

    def test_failed_stages_do_not_reuse_old_reports(self):
        result = report([{'name':'locomo_retrieval','status':'failed','seconds':0,'exit_code':1}])
        self.assertEqual(result['execution_status'], 'incomplete')
        self.assertNotIn('retrieval', result)
        self.assertEqual(result['files'], {})
        self.assertEqual(report([])['execution_status'], 'incomplete')
