from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from main.retrieval.generation import ABSTAIN_THRESHOLD, abstention_reason, answer, question_slot


def context(query, sources, *, status='applied'):
    """sources: list of (memory_id, text, reranker_logit[, speaker[, summary]])."""
    built, scores = {}, {}
    for rank, source in enumerate(sources, 1):
        key, text, logit = source[:3]
        speaker = source[3] if len(source) > 3 else 'Priya'
        summary = source[4] if len(source) > 4 else None
        body = f'speaker: {speaker}\nsession_date: 2024-01-01\n{text}' + (f'\nSummary: {summary}' if summary else '')
        built[key] = {'id': key, 'observed_at': '2024-01-01T00:00:00+00:00', 'text': body}
        scores[key] = {'rank': rank, 'reranker': logit, 'fusion': 0.01, 'cosine': 0.6, 'channels': ['semantic']}
    return {'query': query, 'sources': built, 'scores': scores,
            'reranker': {'status': status, 'model': 'test', 'error_type': None}, 'text': '', 'historical': False}


def decide(query, *texts, logit=3.0):
    return abstention_reason(context(query, [(f'm{i}', text, logit) for i, text in enumerate(texts)]))


class QuestionSlotTests(unittest.TestCase):
    def test_slot_questions(self):
        cases = {
            "What does Priya's brother do for a living?": 'occupation',
            "Do you know what job Amara's brother in Arusha has?": 'occupation',
            "What is Felix's profession by training?": 'occupation',
            'What is the name of my cat?': 'name',
            "Do you know what Priya's sister is called?": 'name',
            'What did Priya name her rescue dog?': 'name',
            'Which city does Priya live in?': 'entity',
            'How often does Priya swim?': 'frequency',
            'How much did the new bike cost?': 'money',
            'What annual salary does Ravi earn?': 'money',
            'How long does Priya knead the dough?': 'duration',
            'How many hours does Jonas cure the salmon?': 'duration',
            'How far is the forest loop?': 'distance',
        }
        for query, slot in cases.items():
            self.assertEqual(question_slot(query), slot, query)

    def test_questions_without_a_slot(self):
        for query in [
            'Is my dog called Biscuit?',                                   # yes/no, name given
            'Did I ever meet someone named Carlos?',                       # yes/no
            "Is Priya's manager at Tidewater called Dana Whitlock?",
            'What is the name of the dog called Biscuit?',                 # name already in the question
            "What is John's number one goal in his basketball career?",    # not asking for a job
            'How much do I like jazz?',                                    # degree, not an amount
            'How many days ago did I watch the Super Bowl?',               # date arithmetic
            'How many plants did I acquire in the last month?',            # counting
            'How many times has Joanna found new hiking trails?',          # counting
            'How long have I had my car?',                                 # computed from dates
            'For how many years has Jonas been building his record collection?',
            'How many months after moving to Oslo was the exam?',
            'How much did I spend on gifts in total?',                     # aggregation
            'Which city did Priya visit, Paris or Rome?',                  # choice
            'Which weekday is the planning call on?',
            'Why did Priya leave Brightline?',
        ]:
            self.assertIsNone(question_slot(query), query)


class SlotEvidenceTests(unittest.TestCase):
    def test_occupation_uses_whole_words(self):
        q = 'What does my brother do for a living?'
        for job in ('nurse', 'chef', 'vet', 'pilot', 'clerk', 'cook', 'architect', 'surgeon', 'midwife', 'barista', 'actuary'):
            self.assertIsNone(decide(q, f'My brother Arjun is a {job} in Leeds.'), job)
        self.assertIsNone(decide(q, 'My brother Arjun lives in Leeds.', 'Arjun works as a sound engineer.'))
        for distractor in ('My brother is a vegetarian.', 'My brother is a keen runner.', 'My mother is a keen runner.'):
            self.assertEqual(decide(q, distractor), 'unfilled_occupation_slot', distractor)
        # The job belongs to someone else: the sentence must be about the brother.
        self.assertEqual(decide(q, 'My brother lives in Leeds.', 'My wife is a nurse.'), 'unfilled_occupation_slot')

    def test_name_detection_is_unicode_and_skips_sentence_initial_words(self):
        q = 'What is the name of my cat?'
        self.assertIsNone(decide(q, 'Our cat Émilie knocked over my plant.'))
        self.assertIsNone(decide(q, 'We adopted a cat and called her Øyvinda.'))
        self.assertEqual(decide(q, 'Adopted a cat last week and she is lovely.'), 'unfilled_name_slot')
        self.assertEqual(decide(q, 'Yesterday the cat knocked over my plant.'), 'unfilled_name_slot')
        self.assertEqual(decide(q, 'Update on the cat: she ate the B2 exam notes.'), 'unfilled_name_slot')

    def test_metadata_and_speakers_never_fill_a_name(self):
        ctx = context('What is the name of my cat?', [('m1', 'The cat sleeps all day.', 3.0, 'Łukasz',
                                                        'Łukasz Nowak says the cat sleeps.')])
        self.assertEqual(abstention_reason(ctx), 'unfilled_name_slot')

    def test_name_relation_anchor(self):
        q = "What is the name of Priya's sister?"
        self.assertEqual(decide(q, "My sister's wedding is in Goa.", 'My brother Arjun lives in Leeds.'),
                         'unfilled_name_slot')
        self.assertIsNone(decide(q, 'My sister Ana lives in Goa.'))

    def test_entity_needs_a_new_proper_noun(self):
        q = 'Which Porto neighbourhood does Priya live in?'
        self.assertEqual(decide(q, 'We finally moved to Porto in March.'), 'unfilled_entity_slot')
        self.assertIsNone(decide(q, 'Our new flat is in Bonfim, close to Porto centre.'))
        self.assertEqual(decide('Which university did Priya attend before Brightline?',
                                "Work-wise I'm at Brightline Analytics."), 'unfilled_entity_slot')

    def test_numeric_slots_need_an_anchored_filler(self):
        self.assertEqual(decide("How long does Priya's metro commute take?",
                                'Finished the half marathon in 1:58.'), 'unfilled_duration_slot')
        self.assertIsNone(decide('How long did my trip to Japan last?', 'I was in Japan for two weeks in April.'))
        self.assertIsNone(decide('How long does Priya rest the dough?', 'I rest the dough for an hour.'))
        self.assertIsNone(decide('How often does Priya video call her brother?', 'We video call most Sundays.'))
        self.assertEqual(decide('How often does Priya swim?', 'I love swimming in the sea.'), 'unfilled_frequency_slot')
        self.assertIsNone(decide('How much rent does Priya pay?', 'Rent is 850 euros a month.'))
        self.assertEqual(decide('How much rent does Priya pay?', 'The rent keeps creeping up.'), 'unfilled_money_slot')

    def test_evidence_set_and_other_speakers(self):
        ctx = context('What name did Priya give her dog?', [('m1', 'The dog loves the beach.', 4.0),
                                                            ('m2', 'Our dog is called Biscuit.', 0.5),
                                                            ('m3', 'Dana Whitlock is my manager.', -3.0)])
        self.assertIsNone(abstention_reason(ctx))
        ctx['scores']['m2']['reranker'] = -1.0   # only a low-scored, non-best source could fill it now
        self.assertEqual(abstention_reason(ctx), 'unfilled_name_slot')
        other = context('How often does Priya swim?', [('m1', 'I swim twice a week.', 4.0, 'Sofia'),
                                                       ('m2', 'Priya hates cold water.', 1.0, 'Priya')])
        self.assertEqual(abstention_reason(other), 'unfilled_frequency_slot')


class AnswerIntegrationTests(unittest.TestCase):
    def test_abstain_payload_and_default_off(self):
        ctx = context('What does my brother do for a living?', [('m1', 'My brother lives in Leeds.', 3.0)])
        result = answer(ctx, generator='extractive', abstain_threshold=ABSTAIN_THRESHOLD)
        self.assertEqual((result['abstain'], result['reason'], result['sources']), (True, 'unfilled_occupation_slot', {}))
        self.assertFalse(answer(ctx, generator='extractive')['abstain'])   # library default: rule off

    def test_noop_without_reranker_scores(self):
        query, text = 'What name did Priya give her rescue dog?', 'The rescue dog is settling in.'
        for status in ('disabled', 'fallback'):
            self.assertFalse(answer(context(query, [('m1', text, None)], status=status), generator='extractive',
                                    abstain_threshold=ABSTAIN_THRESHOLD)['abstain'])
        legacy = context(query, [('m1', text, 3.0)])
        del legacy['scores'], legacy['reranker']
        self.assertFalse(answer(legacy, generator='extractive', abstain_threshold=ABSTAIN_THRESHOLD)['abstain'])

    def test_optional_floor_and_validation(self):
        ctx = context('Which city does Priya live in?', [('m1', 'I live in Lisbon.', -9.5)])
        self.assertIsNone(abstention_reason(ctx, 0.0))
        self.assertEqual(abstention_reason(ctx, 2e-4), 'low_reranker_confidence')
        with self.assertRaises(ValueError):
            answer(ctx, generator='extractive', abstain_threshold=1.5)

    def test_cli_flag_is_opt_in(self):
        from main.__main__ import build_parser
        parser = build_parser()
        self.assertFalse(parser.parse_args(['ask', 'q', '--user-id', 'u', '--generator', 'extractive']).abstain)
        self.assertTrue(parser.parse_args(['ask', 'q', '--user-id', 'u', '--generator', 'extractive', '--abstain']).abstain)

    def test_gemini_path_unchanged(self):
        ctx = context('What name did Priya give her rescue dog?', [('m1', 'The rescue dog is settling in.', 3.0)])
        client = Mock()
        client.with_structured_output.return_value.invoke.return_value = {'abstain': True, 'reason': 'x', 'statements': []}
        self.assertTrue(answer(ctx, client=client, abstain_threshold=ABSTAIN_THRESHOLD)['abstain'])
        client.with_structured_output.assert_called_once()


class CliAbstainTests(unittest.TestCase):
    """`ask --abstain` is experimental: rejected with Gemini, and it warns whenever it cannot take effect."""

    def run_cli(self, args, context_value=None):
        from main.__main__ import main
        with TemporaryDirectory() as directory:
            out, err = StringIO(), StringIO()
            with patch('sys.stdout', out), patch('sys.stderr', err):
                if context_value is None:
                    code = main(['--store', str(Path(directory) / 'cli.jsonl'), *args])
                else:
                    with patch('main.retrieval.rag.MemoryRAG.context', return_value=context_value):
                        code = main(['--store', str(Path(directory) / 'cli.jsonl'), *args])
        return code, json.loads(out.getvalue()), err.getvalue()

    def test_help_says_experimental(self):
        from main.__main__ import build_parser
        ask = build_parser()._subparsers._group_actions[0].choices['ask']
        self.assertIn('Experimental', next(a.help for a in ask._actions if a.dest == 'abstain'))

    def test_rejected_with_gemini(self):
        code, payload, _ = self.run_cli(['ask', 'What is my cat called?', '--user-id', 'u', '--abstain'])
        self.assertEqual(code, 2)
        self.assertEqual(payload['error'], 'invalid_input')
        self.assertIn('--generator extractive', payload['message'])

    def test_warns_when_reranker_not_applied(self):
        code, payload, err = self.run_cli(['ask', 'What is my cat called?', '--user-id', 'u', '--mode', 'graph',
                                           '--generator', 'extractive', '--abstain'])
        self.assertEqual(code, 0)
        self.assertEqual(err.count('warning:'), 1)
        self.assertIn('reranker was not applied', err)

    def test_warns_once_with_no_rerank(self):
        ctx = context('What is my cat called?', [('m1', 'My cat sleeps a lot.', None)], status='disabled')
        code, _, err = self.run_cli(['ask', 'What is my cat called?', '--user-id', 'u', '--no-rerank',
                                     '--generator', 'extractive', '--abstain'], context_value=ctx)
        self.assertEqual(code, 0)
        self.assertEqual(err.count('warning:'), 1)
        self.assertIn('--no-rerank', err)

    def test_warns_once_with_preview(self):
        ctx = context('What is my cat called?', [('m1', 'My cat sleeps a lot.', 3.0)])
        code, payload, err = self.run_cli(['ask', 'What is my cat called?', '--user-id', 'u', '--preview',
                                           '--generator', 'extractive', '--abstain'], context_value=ctx)
        self.assertEqual(code, 0)
        self.assertIn('prompt', payload)
        self.assertEqual(err.count('warning:'), 1)
        self.assertIn('--preview', err)

    def test_applied_reranker_abstains_without_warning(self):
        ctx = context('What does my brother do for a living?', [('m1', 'My brother lives in Leeds.', 3.0)])
        code, payload, err = self.run_cli(['ask', ctx['query'], '--user-id', 'u', '--generator', 'extractive',
                                           '--abstain'], context_value=ctx)
        self.assertEqual((code, err), (0, ''))
        self.assertEqual((payload['abstain'], payload['reason']), (True, 'unfilled_occupation_slot'))
        code, payload, err = self.run_cli(['ask', ctx['query'], '--user-id', 'u', '--generator', 'extractive'],
                                          context_value=ctx)
        self.assertEqual((code, err, payload['abstain']), (0, '', False))   # off without the flag


if __name__ == '__main__':
    unittest.main()
